import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys
import argparse
import random
import numpy as np
# sys.path.append("..")
# sys.path.append(".")
from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from datasets.dataset import Dataset3D_diffusionGAT_train_LiVS
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
import torch as th
from guided_diffusion.train_util import TrainLoop
import torchvision.transforms as transforms
import re
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
seed=10
th.manual_seed(seed)
th.cuda.manual_seed_all(seed)
np.random.seed(seed)

def my_collate_3D(batch):
    ct_25d = th.cat([item[0] for item in batch],axis=0) 
    label = th.cat([item[1] for item in batch],axis=0) 
    nodes_coord = th.cat([item[2] for item in batch],axis=0) 
    edge_index = [item[3] for item in batch]
    speed = th.cat([item[4] for item in batch],axis=0) 
    path = [item[5] for item in batch]
    return ct_25d, label, nodes_coord, edge_index, speed, path

def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist(args)
    args.fold_idx = int(args.fold_idx)
    if args.fold_idx is not None:
        logger.configure(dir = os.path.join(args.out_dir, 'logger', args.name, 'fold_'+str(args.fold_idx)))
    else:
        logger.configure(dir = os.path.join(args.out_dir, 'logger', args.name))

    logger.log("creating data loader...")
    logger.log("this is a training for fold "+str(args.fold_idx))

    case_idx = []
    for root, dirs, files in os.walk(args.training_root_path): 
        if root.split('/')[-1]=='25DSlice_CT_3slices': 
            for name in files:
                case_i = int(re.findall(r"\d+",name)[0])
                if case_i not in case_idx:
                    case_idx.append(case_i)
    case_idx = sorted(case_idx)
    case_idx = np.array(case_idx)

    random.seed(0)
    random.shuffle(case_idx)
    case_idx = list(case_idx)
    case_idx_split = np.split(np.array(case_idx), args.fold)
    if len(args.test_folder)==0:
        test_folder = list(case_idx_split[args.fold_idx])
    else:
        test_folder = args.test_folder
    test_folder = sorted(test_folder)
    logger.log("cases in the test set: "+str(test_folder))

    train_patch_path = {'ct':[], 'vessel':[], 'graph':[]}
    train_folder = []

    for root, dirs, files in os.walk(args.training_root_path): 
        if root.split('/')[-1]=='25DSlice_CT_3slices': 
            for name in files:
                if int(re.findall(r"\d+",name)[0]) not in test_folder and int(re.findall(r"\d+",name)[0]) in case_idx:
                    train_patch_path['ct'].append(os.path.join(root,name))
                    if int(re.findall(r"\d+",name)[0]) not in train_folder:
                        train_folder.append(int(re.findall(r"\d+",name)[0]))

        elif root.split('/')[-1]=='25DSlice_vessel_3slices': 
            for name in files:
                if int(re.findall(r"\d+",name)[0]) not in test_folder and int(re.findall(r"\d+",name)[0]) in case_idx:
                    train_patch_path['vessel'].append(os.path.join(root,name))

        elif root.split('/')[-1]=='25DSlice_graph_3slices_gridXYZ_8_8_1': 
            for name in files:
                if int(re.findall(r"\d+",name)[0]) not in test_folder and int(re.findall(r"\d+",name)[0]) in case_idx:
                    train_patch_path['graph'].append(os.path.join(root,name))

    train_patch_path['ct'] = sorted(train_patch_path['ct'])
    train_patch_path['vessel'] = sorted(train_patch_path['vessel'])
    train_patch_path['graph'] = sorted(train_patch_path['graph'])

    train_folder = sorted(train_folder)
    logger.log("cases in the training set: "+str(train_folder))

    trainset = Dataset3D_diffusionGAT_train_LiVS(train_patch_path, name='train')

    train_loader = DataLoader(trainset,
                            batch_size=args.batch_size,
                            num_workers=8,
                            drop_last=False,
                            collate_fn=my_collate_3D,
                            pin_memory=False,
                            shuffle=True,
                            prefetch_factor=4,
                            persistent_workers=True
                            )

    data = iter(train_loader)

    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    if args.multi_gpu:
        model = th.nn.DataParallel(model,device_ids=[int(id) for id in args.multi_gpu.split(',')])
        model.to(device = th.device('cuda', int(args.gpu_dev)))
    else:
        model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion,  maxt=args.diffusion_steps)


    logger.log("training...")
    if args.fold_idx is not None:
        writer = SummaryWriter(log_dir=os.path.join(args.out_dir, "logs", args.name, 'fold_'+str(args.fold_idx)))
    else:
        writer = SummaryWriter(log_dir=os.path.join(args.out_dir, "logs", args.name))
    TrainLoop(
        name=args.name,
        writer=writer,
        model=model,
        diffusion=diffusion,
        classifier=None,
        data=data,
        dataloader=train_loader,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        output_root=args.out_dir,
        fold_idx=args.fold_idx,
        logger=logger
    ).run_loop()


def create_argparser():
    defaults = dict(
        name='temp',
        data_name = 'LiVS',
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=10,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=200,
        save_interval=10000,
        resume_checkpoint='',#'"./results/pretrainedmodel.pt",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        gpu_dev = "0",
        multi_gpu = None, #"0,1,2"
        training_root_path = '/data1/xzhang2/liver_vessel_segmentation/data/LiVS',
        out_dir='/home/xzhang2/data1/liver_vessel_segmentation/model/GATSegDiff/outputs',
        test_folder=[],
        fold=3, # cross validation
        fold_idx=None

    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
