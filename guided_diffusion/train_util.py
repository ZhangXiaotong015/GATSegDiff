import copy
import functools
import os
import numpy as np
import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from guided_diffusion import dist_util, logger
from guided_diffusion.fp16_util import MixedPrecisionTrainer
from guided_diffusion.nn import update_ema
from guided_diffusion.resample import LossAwareSampler, UniformSampler
from guided_diffusion.metrics import AverageMeter, precision, recall, fscore, accuracy
# from visdom import Visdom
# viz = Visdom(port=8850)
# loss_window = viz.line( Y=th.zeros((1)).cpu(), X=th.zeros((1)).cpu(), opts=dict(xlabel='epoch', ylabel='Loss', title='loss'))
# grad_window = viz.line(Y=th.zeros((1)).cpu(), X=th.zeros((1)).cpu(),
#                            opts=dict(xlabel='step', ylabel='amplitude', title='gradient'))


# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

class TrainLoop:
    def __init__(
        self,
        *,
        name,
        writer,
        model,
        classifier,
        diffusion,
        data,
        dataloader,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        output_root=None,
        fold_idx=None,
        logger=None,
        case_idx_record=[]
    ):
        self.name = name
        self.writer = writer
        self.model = model
        self.dataloader=dataloader
        self.classifier = classifier
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.output_root = output_root
        self.fold_idx = fold_idx
        self.logger = logger
        self.case_idx_record = case_idx_record

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            print('resume model')
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            logger.log("restart the training from step "+str(self.resume_step))
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        lossesMeter = AverageMeter(name='TrainMeter total loss ')
        lossesMeter_cond = AverageMeter(name='TrainMeter condition loss ')
        lossesMeter_diff = AverageMeter(name='TrainMeter diffusion loss ')
        # lossesMeter_diff_boundary = AverageMeter(name='TrainMeter boundary loss for diffUNet ')
        precisionMeter = AverageMeter(name='TrainMeter precision')
        recallMeter = AverageMeter(name='TrainMeter recall')
        fscoreMeter = AverageMeter(name='TrainMeter f1')
        accMeter = AverageMeter(name='TrainMeter accuracy')

        i = 0
        data_iter = iter(self.dataloader)
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):

            try:
                batch, cond, nodes_coord, edge_index, speed, path = next(data_iter) # CT and seg label
            except StopIteration:
                # StopIteration is thrown if dataset ends
                # reinitialize data loader
                data_iter = iter(self.dataloader)
                batch, cond, nodes_coord, edge_index, speed, path = next(data_iter)

            'Check the cases in the training set'
            for sub_path in path:
                if sub_path.split('_')[0] not in self.case_idx_record:
                    self.case_idx_record.append(sub_path.split('_')[0])
                    self.case_idx_record = sorted(self.case_idx_record)

            graph = {'nodes':nodes_coord.cuda(), 'edges':edge_index, 'speed':speed.cuda()}

            losses, sample, timestamp, cal = self.run_step(batch, cond, graph) # loss dictionary, sampled noise, current timestamp

            total_loss = (losses['mse_diff'] + losses['loss_cal'] ).mean(0) # formal implementation
            # total_loss = (losses['mse_diff'] + losses['loss_boundary'] + losses['loss_cal'] ).mean(0)
            cond_loss = losses['loss_cal'].mean(0)
            diff_loss = losses['mse_diff'].mean(0)
            # diff_boundary_loss = losses['loss_boundary']

            precision_val = precision(th.sigmoid(cal).view(*speed.shape), speed.type(th.int32).cuda())
            recall_val = recall(th.sigmoid(cal).view(*speed.shape), speed.type(th.int32).cuda())
            fscore_val = fscore(th.sigmoid(cal).view(*speed.shape), speed.type(th.int32).cuda())
            acc_val = accuracy(th.sigmoid(cal).view(*speed.shape), speed.type(th.int32).cuda())

            'save metrics scalars'
            lossesMeter.update(total_loss.item())
            lossesMeter_cond.update(cond_loss.item())
            lossesMeter_diff.update(diff_loss.item())
            # lossesMeter_diff_boundary.update(diff_boundary_loss.item())
            precisionMeter.update(precision_val.item())
            recallMeter.update(recall_val.item())
            fscoreMeter.update(fscore_val.item())
            accMeter.update(acc_val.item())

            if (self.step+self.resume_step+1) % 10000 == 0:
                self.logger.log('Training cases are '+str(self.case_idx_record)+'!')
                if self.fold_idx is None:
                    os.makedirs('{}/{}/{}/{}'.format('/data1/xzhang2/liver_vessel_segmentation/model/GATSegDiff/outputs/logger', self.name, 'pred_nodes_train', 'Iter_'+str(self.step+self.resume_step+1)), exist_ok=True)
                    nodes_save_root = os.path.join('/data1/xzhang2/liver_vessel_segmentation/model/GATSegDiff/outputs/logger', self.name, 'pred_nodes_train', 'Iter_'+str(self.step+self.resume_step+1))
                else:
                    os.makedirs('{}/{}/{}/{}/{}'.format('/data1/xzhang2/liver_vessel_segmentation/model/GATSegDiff/outputs/logger', self.name, 'fold_'+str(self.fold_idx), 'pred_nodes_train', 'Iter_'+str(self.step+self.resume_step+1)), exist_ok=True)
                    nodes_save_root = os.path.join('/data1/xzhang2/liver_vessel_segmentation/model/GATSegDiff/outputs/logger', self.name, 'fold_'+str(self.fold_idx), 'pred_nodes_train', 'Iter_'+str(self.step+self.resume_step+1))
                bi_cnt = 0
                for path_bi in path:
                    # if 'Edge' not in path_bi:
                    np.save(os.path.join(nodes_save_root, 'predNodes_'+path_bi.replace('.nii.gz','.npy')), cal[bi_cnt,].unsqueeze(0).detach().cpu().numpy())
                    np.save(os.path.join(nodes_save_root, 'gtNodes_'+path_bi.replace('.nii.gz','.npy')), speed.view(*cal.shape)[bi_cnt,].unsqueeze(0).detach().cpu().numpy())
                    bi_cnt +=1
                    # elif 'Edge' in path_bi:
                    #     bi_cnt +=1
                    #     continue

            if (self.step+self.resume_step+1) % self.log_interval == 0:
                logger.dumpkvs()
                self.writer.add_scalar("train/loss", scalar_value=lossesMeter.avg, global_step=self.step+self.resume_step+1)
                self.writer.add_scalar("train/loss_cond", scalar_value=lossesMeter_cond.avg, global_step=self.step+self.resume_step+1)
                self.writer.add_scalar("train/loss_diff_MSE", scalar_value=lossesMeter_diff.avg, global_step=self.step+self.resume_step+1)
                # self.writer.add_scalar("train/loss_diff_Boundary", scalar_value=lossesMeter_diff_boundary.avg, global_step=self.step+self.resume_step+1)
                self.writer.add_scalar("train_metrics/precision", scalar_value=precisionMeter.avg, global_step=self.step+self.resume_step+1)
                self.writer.add_scalar("train_metrics/recall", scalar_value=recallMeter.avg, global_step=self.step+self.resume_step+1)
                self.writer.add_scalar("train_metrics/f1", scalar_value=fscoreMeter.avg, global_step=self.step+self.resume_step+1)
                self.writer.add_scalar("train_metrics/accracy", scalar_value=accMeter.avg, global_step=self.step+self.resume_step+1)
           
            i += 1
          
            if (self.step+self.resume_step+1) % self.log_interval == 0:
                logger.dumpkvs()
            if (self.step+self.resume_step+1) % self.save_interval == 0:
                # self.save()
                if self.fold_idx is None:
                    self.save_model(self.mp_trainer, self.opt, self.step+self.resume_step+1, os.path.join(self.output_root, 'model', self.name))
                else:
                    self.save_model(self.mp_trainer, self.opt, self.step+self.resume_step+1, os.path.join(self.output_root, 'model', self.name, 'fold_'+str(self.fold_idx)))
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step+self.resume_step - 1) % self.save_interval != 0:
            # self.save()
            if self.fold_idx is None:
                self.save_model(self.mp_trainer, self.opt, self.step+self.resume_step+1, os.path.join(self.output_root, 'model', self.name))
            else:
                self.save_model(self.mp_trainer, self.opt, self.step+self.resume_step+1, os.path.join(self.output_root, 'model', self.name, 'fold_'+str(self.fold_idx)))

            self.writer.close()


    def run_step(self, batch, cond, graph):
        batch=th.cat((batch, cond), dim=1) 

        cond={}
        losses, sample, t, cal = self.forward_backward(batch, cond, graph)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()
        return losses, sample, t, cal

    def forward_backward(self, batch, cond, graph):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            micro_cond.update(graph)

            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses_segmentation,
                self.ddp_model,
                self.classifier,
                micro,
                t,
                self.logger,
                model_kwargs=micro_cond
            )

            if last_batch or not self.use_ddp:
                losses1 = compute_losses()

            else:
                with self.ddp_model.no_sync():
                    losses1 = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )
            losses = losses1[0]
            sample = losses1[1]
            cal = losses1[2]

            'Losses combination'
            loss = (losses["loss"] * weights + losses['loss_cal'] * 1 ).mean() # formal implementation
            # loss = ((losses["loss"]+0.05*losses["loss_boundary"]) * weights + losses['loss_cal'] * 1 ).mean()

            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            'back propagation'
            self.mp_trainer.backward(loss)

            for name, param in self.ddp_model.named_parameters():
                if param.grad is None:
                    print(name)
            return  losses, sample, t, cal

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save_model(self, mp_trainer, opt, step, save_dir):
        print('saving')
        if dist.get_rank() == 0:
            output_dir = save_dir
            os.makedirs(output_dir,exist_ok=True)
            th.save(
                mp_trainer.master_params_to_state_dict(mp_trainer.master_params),
                os.path.join(output_dir, f"model{step:d}.pt"),
            )
            th.save(opt.state_dict(), os.path.join(output_dir, f"opt{step:d}.pt"))

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"savedmodel{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"emasavedmodel_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"optsavedmodel{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
