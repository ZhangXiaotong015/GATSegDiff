import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["TORCH_SHM_SIZE"] = "1024M"

import argparse

from ssl import OP_NO_TLSv1
import nibabel as nib
# from visdom import Visdom
# viz = Visdom(port=8850)
import sys
import re
from dotmap import DotMap
import random
import SimpleITK as sitk
sys.path.append(".")
import numpy as np
import time
import torch as th
from PIL import Image
import torch.distributed as dist
from guided_diffusion import dist_util, logger
from datasets.dataset import Dataset3D_diffusionGAT_sample_LiVS
import torchvision.utils as vutils
from guided_diffusion.utils import staple
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
import torchvision.transforms as transforms
# from torchsummary import summary
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# from guided_diffusion.custom_model import CustomModel
# seed=10
# th.manual_seed(seed)
# th.cuda.manual_seed_all(seed)
# np.random.seed(seed)

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

def my_collate_sample_GAT(batch):
    ct25d = th.cat([item[0] for item in batch],axis=0) 
    nodes_coord = th.cat([item[1] for item in batch],axis=0) 
    edge_index = [item[2] for item in batch]
    path = [item[3] for item in batch]
    origin = [item[4] for item in batch]
    space = [item[5] for item in batch]
    direction = [item[6] for item in batch]
    return ct25d, nodes_coord, edge_index, path, origin, space, direction


def main_GAT(args):
    
    dist_util.setup_dist(args)

    logger.configure(dir = args.out_dir)
    
    args.num_samples = len(os.listdir(args.data_test_CT))

    # logger.log("Cases in the test set: "+str(test_folder))

    seed = args.seed
    logger.log(f"Use seed {seed}!")
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    np.random.seed(seed)


    test_patch_path = {'ct':[]}

    for root, dirs, files in os.walk(args.test_root_path): 
        if len(files)==0:
            continue 
        for name in files:
            test_patch_path['ct'].append(os.path.join(root,name))


    test_patch_path['ct'] = sorted(test_patch_path['ct'])


    testset = Dataset3D_diffusionGAT_sample_LiVS(test_patch_path, name='sample')

    test_loader = DataLoader(testset,
                            batch_size=args.batch_size,
                            num_workers=2,
                            drop_last=False,
                            collate_fn=my_collate_sample_GAT,
                            pin_memory=False,
                            shuffle=True,
                            # prefetch_factor=4,
                            # persistent_workers=True
                            )

    data = iter(test_loader)

    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    all_images = []

    model_path = args.model_path

    state_dict = dist_util.load_state_dict(model_path, map_location="cpu")
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module.' in k:
            new_state_dict[k[7:]] = v
            # load params
        else:
            new_state_dict = state_dict

    model.load_state_dict(new_state_dict)

    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("model loaded from "+ model_path)

    if args.dpm_solver:
        for vessel_ite in range(args.num_samples): # case-level
            try:
                batch, nodes_coord, edge_index, path, origin, space, direction = next(data)
            except StopIteration:
                data = iter(test_loader)
                batch, nodes_coord, edge_index, path, origin, space, direction = next(data)

            text = path[0].split('/')[-1]   
            width = 35                      
            print('#' * width)
            print(f"# {text.center(width - 4)} #")
            print('#' * width)

            liver_mask = th.zeros(batch[:,1][:,None].shape) # (N,1,256,256)
            liver_mask[batch[:,1][:,None]>0] = 1

            ensemble_sample_list = []

            for ensemble_ite in range(args.num_ensemble):
                pred_list = [] # diffusion results
                graph_pred = []
                for bi in range(int(np.ceil(batch.shape[0]/args.batch_size_sample))): # per vessel
                    S = bi*args.batch_size_sample
                    E = (bi+1)*args.batch_size_sample

                    if E>batch.shape[0]:
                        E = batch.shape[0]
                    batch_size_real = E-S
                    'graph used in the inference is the fully connected grid'
                    graph = {'nodes':nodes_coord.repeat(batch_size_real,1,1).cuda(), 'edges':[edge_index[0].cuda() for i in range(batch_size_real)]}

                    if liver_mask[S:E,].sum()>0:

                        liver = liver_mask[S:E,]
                        b = batch[S:E,] * liver

                        c = th.randn_like(b[:, :1, ...])
                        img = th.cat((b, c), dim=1)     #add a noise channel$

                        logger.log('sampling '+path[0].split('/')[-1]+' with '+model_path)

                        start = th.cuda.Event(enable_timing=True)
                        end = th.cuda.Event(enable_timing=True)
                        
                        model_kwargs = {}
                        model_kwargs.update(graph)
                        start.record()
                        sample_fn = (
                            diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
                        )
                        sample, x_noisy, org, cal, cal_out = sample_fn(
                            model,
                            (batch_size_real, 4, args.image_size, args.image_size), 
                            img,
                            step = args.diffusion_steps,
                            clip_denoised=args.clip_denoised,
                            model_kwargs=model_kwargs,
                            progress=True
                        )
                        end.record()
                        th.cuda.synchronize()
                        print('time for 1 sample', start.elapsed_time(end))  #time measurement for the generation of 1 sample

                        sample = th.clamp(sample, 0, 1)
                        sample[sample>=0.5] = 1
                        sample[sample<0.5] = 0
                        cal_out[cal_out>=0.5] = 1
                        cal_out[cal_out<0.5] = 0

                        pred_list.append(sample * liver.cuda())

                    else:
                        pred_list.append(th.zeros(liver_mask[S:E,].shape).cuda())

                ensemble_sample_list.append(th.cat(pred_list, dim=0).squeeze(1)) # 256*256*256

                pred_list = th.cat(pred_list, dim=0).squeeze(1).detach().cpu().numpy()

                save_root = args.out_dir
                'save sampling results per ensemble step'
                img = sitk.GetImageFromArray(pred_list)
                img.SetOrigin(origin[0])
                img.SetDirection(direction[0])
                img.SetSpacing(space[0])
                os.makedirs('{}/{}'.format(save_root, 'ensemble_'+str(ensemble_ite)), exist_ok=True)
                sitk.WriteImage(img, '{}/{}/{}'.format(save_root, 'ensemble_'+str(ensemble_ite), 'pred_sample_'+path[0].split('/')[-1]))

            'save ensembed sampling results'
            ensemble_sample = staple(th.stack(ensemble_sample_list,dim=0)) # sliceNum*256*256

            ensemble_sample = th.clamp(ensemble_sample, 0, 1).squeeze(0).detach().cpu().numpy()

            ensemble_sample[ensemble_sample>=0.5] = 1
            ensemble_sample[ensemble_sample<0.5] = 0

            img = sitk.GetImageFromArray(ensemble_sample)
            img.SetOrigin(origin[0])
            img.SetDirection(direction[0])
            img.SetSpacing(space[0])
            sitk.WriteImage(img, '{}/{}'.format(save_root, 'ensembled_pred_sample_'+path[0].split('/')[-1]))

    else: # Plain DDPM sampling
        for vessel_ite in range(args.num_samples): # case-level
            try:
                batch, nodes_coord, edge_index, path, origin, space, direction = next(data)
            except StopIteration:
                data = iter(test_loader)
                batch, nodes_coord, edge_index, path, origin, space, direction = next(data)

            liver_mask = th.zeros(batch[:,1][:,None].shape) # (N,1,256,256)
            liver_mask[batch[:,1][:,None]>0] = 1

            logger.log("current inference is for case " + path[0].split('/')[-1] +' !')

            ensemble_sample_list = []

            for ensemble_ite in range(args.num_ensemble):

                pred_combine_list = [] # combined predict (integrate condition unet results with diffusion results together)
                pred_list = [] # diffusion results

                for bi in range(int(np.ceil(batch.shape[0]/args.batch_size_sample))): # per vessel
                    S = bi*args.batch_size_sample
                    E = (bi+1)*args.batch_size_sample

                    if E>batch.shape[0]:
                        E = batch.shape[0]
                    batch_size_real = E-S
                    'graph used in the inference is the fully connected grid'
                    graph = {'nodes':nodes_coord.repeat(batch_size_real,1,1).cuda(), 'edges':[edge_index[0].cuda() for i in range(batch_size_real)]}

                    if liver_mask[S:E,].sum()>0:

                        liver = liver_mask[S:E,]
                        b = batch[S:E,] * liver

                        c = th.randn_like(b[:, :1, ...])
                        img = th.cat((b, c), dim=1)     #add a noise channel$

                        logger.log('sampling case '+path[0].split('/')[-1]+' with '+ model_path)

                        start = th.cuda.Event(enable_timing=True)
                        end = th.cuda.Event(enable_timing=True)
                        
                        # for i in range(args.num_ensemble):  #this is for the generation of an ensemble of 5 masks.
                        model_kwargs = {}
                        model_kwargs.update(graph)
                        start.record()
                        sample_fn = (
                            diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
                        )
                        sample, x_noisy, org, cal, cal_out, sample_prog, cal_out_prog = sample_fn(
                            model,
                            (batch_size_real, 4, args.image_size, args.image_size), 
                            img,
                            step = args.diffusion_steps,
                            clip_denoised=args.clip_denoised,
                            model_kwargs=model_kwargs,
                            progress=True
                        )
                        end.record()
                        th.cuda.synchronize()
                        print('time for 1 sample', start.elapsed_time(end))  #time measurement for the generation of 1 sample

                        sample = th.clamp(sample, 0, 1)
                        sample[sample>=0.5] = 1
                        sample[sample<0.5] = 0
                        cal_out[cal_out>=0.5] = 1
                        cal_out[cal_out<0.5] = 0

                        pred_list.append(sample * liver.cuda())

                    else:
                        pred_list.append(th.zeros(liver_mask[S:E,].shape).cuda())

                ensemble_sample_list.append(th.cat(pred_list, dim=0).squeeze(1)) # 256*256*256

                pred_list = th.cat(pred_list, dim=0).squeeze(1).detach().cpu().numpy()

                save_root = args.out_dir
                'save sampling results per ensemble step'
                img = sitk.GetImageFromArray(pred_list)
                img.SetOrigin(origin[0])
                img.SetDirection(direction[0])
                img.SetSpacing(space[0])
                os.makedirs('{}/{}'.format(save_root, 'ensemble_'+str(ensemble_ite)), exist_ok=True)
                sitk.WriteImage(img, '{}/{}/{}'.format(save_root, 'ensemble_'+str(ensemble_ite), 'pred_sample_'+path[0].split('/')[-1]))

            'save ensembed sampling results'
            ensemble_sample = staple(th.stack(ensemble_sample_list,dim=0)) # sliceNum*256*256
            ensemble_sample = th.clamp(ensemble_sample, 0, 1).squeeze(0).detach().cpu().numpy()
            ensemble_sample[ensemble_sample>=0.5] = 1
            ensemble_sample[ensemble_sample<0.5] = 0

            img = sitk.GetImageFromArray(ensemble_sample)
            img.SetOrigin(origin[0])
            img.SetDirection(direction[0])
            img.SetSpacing(space[0])
            sitk.WriteImage(img, '{}/{}'.format(save_root, 'ensembled_pred_sample_'+path[0].split('/')[-1]))


def create_argparser():
    defaults = dict(
        name='temp',
        model_path=os.getenv("MODEL_PATH", "/app/model/LiVS_reannotated_subset_F0_model160000.pt"),
        seed=10,
        clip_denoised=True,
        num_samples=-1, # validation vessel number
        batch_size=1, # load a full CT volme in the dataloader
        batch_size_sample=32, # how many slices be involved in one sampling iteration
        diffusion_steps=1000,
        use_ddim=False,
        dpm_solver=False,
        num_ensemble=1,      #number of samples in the ensemble
        gpu_dev = "0",
        multi_gpu = None, #"0,1,2"
        data_test_CT = '', # path of the original CT images
        test_root_path = '',# path of 2.5D CT input
        out_dir='',# Output dir
        noise_cancel=False
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":

    args = create_argparser().parse_args()

    '2.5D data generation'
    from data_processing import crop25d_testset_LiVS
    print('2.5D data generation...')
    print()
    crop25d_testset_LiVS(args.data_test_CT, args.test_root_path, slices=3)

    'Inference'
    print('Inference...')
    print()
    main_GAT(args)

    'Post-processing'
    from data_processing import prediction_transpose
    from noise_cancel import noise_remove_connected_region_2
    print('Transposition...')
    print()
    prediction_transpose(args.out_dir)

    if args.noise_cancel:
        print('Noice Cancellation...')
        print()
        noise_remove_connected_region_2(args.out_dir)
