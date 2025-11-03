import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
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
from datasets.dataset import Dataset3D_diffusionGAT_sample_LiVS, Dataset3D_diffusionGAT_uncertainty_control_LiVS
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
from guided_diffusion.custom_model import CustomModel
# seed=10
# th.manual_seed(seed)
# th.cuda.manual_seed_all(seed)
# np.random.seed(seed)

def staple_ensemble_tensor(ensemble_sample):
    """
    STAPLE ensemble implementation for a tensor input using SimpleITK.

    Args:
        ensemble_sample (torch.Tensor): Tensor of shape (num_models, slices, height, width).
                                         Binary segmentation maps for each model.

    Returns:
        torch.Tensor: Probabilistic segmentation map (STAPLE result) as a PyTorch tensor.
    """
    # Ensure the input tensor is of shape (num_models, slices, height, width)
    assert len(ensemble_sample.shape) == 4, "Tensor must be of shape (num_models, slices, height, width)"
    
    # Convert tensor to numpy array and then to SimpleITK images
    num_models, slices, height, width = ensemble_sample.shape
    ensemble_images = []
    
    for model_idx in range(num_models):
        # Convert model-specific segmentation map (slices, height, width) to SimpleITK.Image
        np_image = ensemble_sample[model_idx].cpu().numpy().astype(np.uint8)  # Convert to numpy and ensure binary
        sitk_image = sitk.GetImageFromArray(np_image)  # Convert to SimpleITK.Image
        ensemble_images.append(sitk_image)

    # Perform STAPLE ensemble
    staple_filter = sitk.STAPLEImageFilter()
    staple_result = staple_filter.Execute(ensemble_images)  # Probabilistic result

    # Convert back to PyTorch tensor
    result_np = sitk.GetArrayFromImage(staple_result)  # Convert to numpy (slices, height, width)
    
    return result_np

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

def my_collate_uncertainty_control(batch):
    # x_t
    xt = th.cat([item[0] for item in batch],axis=0) 
    # cond
    ct_25d = th.cat([item[1] for item in batch],axis=0) 
    nodes_coord = th.cat([item[2] for item in batch],axis=0) 
    edge_index = [item[3] for item in batch]
    # t
    t = th.cat([item[4] for item in batch],axis=0) 
    # y: target noise
    y = th.cat([item[5] for item in batch],axis=0) 
    return xt, ct_25d, nodes_coord, edge_index, t, y

def my_collate_sample(batch):
    # ct = th.cat([item[0] for item in batch],axis=0) 
    ct25d = th.cat([item[0] for item in batch],axis=0) 
    label = th.cat([item[1] for item in batch],axis=0) 
    liver = th.cat([item[2] for item in batch],axis=0) 
    path = [item[3] for item in batch]
    origin = [item[4] for item in batch]
    space = [item[5] for item in batch]
    direction = [item[6] for item in batch]
    return ct25d, label, liver, path, origin, space, direction

def my_collate_sample_GAT(batch):
    ct25d = th.cat([item[0] for item in batch],axis=0) 
    label = th.cat([item[1] for item in batch],axis=0) 
    nodes_coord = th.cat([item[2] for item in batch],axis=0) 
    edge_index = [item[3] for item in batch]
    # label_slice_idx = [item[4] for item in batch]
    path = [item[4] for item in batch]
    origin = [item[5] for item in batch]
    space = [item[6] for item in batch]
    direction = [item[7] for item in batch]
    # ct_slice_num_total = [item[9] for item in batch]
    return ct25d, label, nodes_coord, edge_index, path, origin, space, direction

def my_collate_sample_liver_example(batch):
    ct25d = th.cat([item[0] for item in batch],axis=0) 
    label = th.cat([item[1] for item in batch],axis=0) 
    nodes_coord = th.cat([item[2] for item in batch],axis=0) 
    edge_index = [item[3] for item in batch]
    liver = th.cat([item[4] for item in batch],axis=0) 
    path = [item[5] for item in batch]
    origin = [item[6] for item in batch]
    space = [item[7] for item in batch]
    direction = [item[8] for item in batch]
    return ct25d, label, nodes_coord, edge_index, liver, path, origin, space, direction

def load_liver_example(test_root_path):
    test_patch_path = {'ct':[], 'liver':[], 'label':[]}

    for root, dirs, files in os.walk(test_root_path): 
        if len(files)==0:
            continue
        for name in files:
            if 'PrePortal' in name:
                if 'orig.nii.gz' in name:
                    test_patch_path['ct'].append(os.path.join(root,name))
                elif 'liver.nii.gz' in name:
                    test_patch_path['liver'].append(os.path.join(root,name))
                elif 'manualgrow' in name:
                    test_patch_path['label'].append(os.path.join(root,name))

    test_patch_path['ct'] = sorted(test_patch_path['ct'])
    test_patch_path['liver'] = sorted(test_patch_path['liver'])
    test_patch_path['label'] = sorted(test_patch_path['label'])
    return test_patch_path


def main_GAT():
    args = create_argparser().parse_args()
    dist_util.setup_dist(args)
    args.fold_idx = int(args.fold_idx)
    args.inFold_sub_idx = int(args.inFold_sub_idx)

    if args.fold_idx is not None:
        logger.configure(dir = os.path.join(args.out_dir, 'logger_validation', args.name, 'log', 'fold_'+str(args.fold_idx)+'_subset_'+str(args.inFold_sub_idx)))
    else:
        logger.configure(dir = os.path.join(args.out_dir, 'logger_validation', args.name))
    logger.log("this is a inference for fold "+str(args.fold_idx))

    case_idx = []
    for root, dirs, files in os.walk(args.test_root_path): 
        if root.split('/')[-1]=='Test25DCT_CTVolume_3slices': 
            for name in files:
                case_i = name.split('.')[0]
                if case_i not in case_idx:
                    case_idx.append(case_i)
    case_idx = sorted(case_idx)
    case_idx = np.array(case_idx)

    random.seed(0)
    random.shuffle(case_idx)
    case_idx_split = np.split(np.array(case_idx), args.fold)
    if len(args.test_folder)==0:
        test_folder_whole = list(case_idx_split[args.fold_idx]) # 101
    else:
        test_folder_whole = args.test_folder
    test_folder_sub = list(np.array_split(np.array(test_folder_whole), args.inFold_subset)[args.inFold_sub_idx])
    test_folder = sorted(test_folder_sub)

    args.num_samples = len(test_folder)

    logger.log("This is an inference for subset "+str(args.inFold_sub_idx))
    logger.log("Cases in the test set: "+str(test_folder))

    seed = args.seed
    logger.log(f"Use seed {seed}!")
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    if args.uncertainty_control:
        training_root_path = args.test_root_path
        train_patch_path = {'ct':[], 'vessel':[], 'graph':[]}
        for root, dirs, files in os.walk(training_root_path): 
            if root.split('/')[-1]=='25DSlice_CT_3slices': 
                for name in files:
                    if name.split('_')[0] not in test_folder and name.split('_')[0] in case_idx:
                        train_patch_path['ct'].append(os.path.join(root,name))

            elif root.split('/')[-1]=='25DSlice_vessel_3slices': 
                for name in files:
                    if name.split('_')[0] not in test_folder and name.split('_')[0] in case_idx:
                        train_patch_path['vessel'].append(os.path.join(root,name))

            elif root.split('/')[-1]=='25DSlice_graph_3slices_gridXYZ_8_8_1': 
                for name in files:
                    if name.split('_')[0].replace('graph','case') not in test_folder and name.split('_')[0].replace('graph','case') in case_idx:
                        train_patch_path['graph'].append(os.path.join(root,name))

        train_patch_path['ct'] = sorted(train_patch_path['ct'])
        train_patch_path['vessel'] = sorted(train_patch_path['vessel'])
        train_patch_path['graph'] = sorted(train_patch_path['graph'])


        trainset = Dataset3D_diffusionGAT_uncertainty_control_LiVS(train_patch_path, 
                                beta_schedule='linear', 
                                beta_start=0.0001,
                                beta_end=0.02,
                                num_diffusion_timesteps=1000,
                                name='train set based uncertainty control')

        train_loader = DataLoader(trainset,
                                batch_size=args.batch_size,
                                num_workers=6,
                                drop_last=False,
                                collate_fn=my_collate_uncertainty_control,
                                pin_memory=False,
                                shuffle=True,
                                prefetch_factor=4,
                                persistent_workers=True
                                )
    else:
        train_loader = None


    # test_patch_path = {'ct':[], 'vessel':[], 'label_slice_idx':[]}
    test_patch_path = {'ct':[], 'vessel':[]}

    for root, dirs, files in os.walk(args.test_root_path): 
        if root.split('/')[-1]=='Test25DCT_CTVolume_3slices': 
            for name in files:
                if name.split('.')[0] in test_folder:
                    test_patch_path['ct'].append(os.path.join(root,name))

        elif root.split('/')[-1]=='Test25DCT_Vessel_3slices': 
            for name in files:
                if name.split('.')[0] in test_folder:
                    test_patch_path['vessel'].append(os.path.join(root,name))

        # elif root.split('/')[-1]=='vessel_mask_w_label_idx_stage3': 
        #     for name in files:
        #         if int(re.findall(r"\d+",name)[0]) in test_folder:
        #             test_patch_path['label_slice_idx'].append(os.path.join(root,name))


    test_patch_path['ct'] = sorted(test_patch_path['ct'])
    test_patch_path['vessel'] = sorted(test_patch_path['vessel'])
    # test_patch_path['label_slice_idx'] = sorted(test_patch_path['label_slice_idx'])

    # test_patch_path = load_liver_example(args.test_root_path)

    testset = Dataset3D_diffusionGAT_sample_LiVS(test_patch_path, name='sample')
    # testset = Dataset3D_diffusion_sample_liver_example(test_patch_path, name='sample')

    test_loader = DataLoader(testset,
                            batch_size=args.batch_size,
                            num_workers=6,
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
    if args.fold_idx is not None:
        writer = SummaryWriter(log_dir=os.path.join(args.out_dir, "logs_validation", args.name, 'fold_'+str(args.fold_idx)))
        model_name = os.path.join(args.model_name, 'fold_'+str(args.fold_idx))
    else:
        writer = SummaryWriter(log_dir=os.path.join(args.out_dir, "logs_validation", args.name))
        model_name = args.model_name

    while True:
        for root, dirs, files in os.walk(os.path.join(args.out_dir, 'model', model_name)): 
            if len(files)==0:
                continue
            files = [s.replace(re.findall("\d+",s)[0]+'.', re.findall("\d+",s)[0].zfill(6)+'.') for s in files]
            files = sorted(files)

            for name in files:
                if 'model' not in name:
                    continue
                steps = int(re.findall("\d+",name)[0])
                if steps % args.log_interval !=0:
                    continue
                if steps < args.start_log_step:
                    continue
                if steps > args.end_log_step:
                    continue

                name = 'model' + str(int(re.findall("\d+",name)[0])) +'.pt'

                state_dict = dist_util.load_state_dict(os.path.join(root,name), map_location="cpu")
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    # name = k[7:] # remove `module.`
                    if 'module.' in k:
                        new_state_dict[k[7:]] = v
                        # load params
                    else:
                        new_state_dict = state_dict

                model.load_state_dict(new_state_dict)

                # model.load_state_dict(dist_util.load_state_dict(os.path.join(root,name), map_location="cpu"))
                model.to(dist_util.dev())
                if args.use_fp16:
                    model.convert_to_fp16()
                model.eval()

                logger.log("model loaded "+model_name+'/'+name)

                if args.uncertainty_control:
                    train_dataloader = train_loader
                    dataset = ''
                    image_size = 256
                    config = DotMap({'data': {'dataset': dataset, 'image_size':image_size,
                                    'logit_transform': False, 'uniform_dequantization':False,
                                    'gaussian_dequantization': False, 'random_flip':True, 'rescaled': True}})

                    custom_model = CustomModel(model, train_dataloader, {}, config, dataset, image_size) 

                if args.dpm_solver:
                    for vessel_ite in range(args.num_samples): # case-level
                        try:
                            # batch, mask, nodes_coord, edge_index, label_slice_idx, path, origin, space, direction, ct_slice_num_total = next(data)  #should return an image from the dataloader "data"
                            batch, mask, nodes_coord, edge_index, path, origin, space, direction = next(data)
                        except StopIteration:
                            data = iter(test_loader)
                            # batch, mask, nodes_coord, edge_index, label_slice_idx, path, origin, space, direction, ct_slice_num_total = next(data)
                            batch, mask, nodes_coord, edge_index, path, origin, space, direction = next(data)

                        liver_mask = th.zeros(batch[:,1][:,None].shape) # (N,1,256,256)
                        liver_mask[batch[:,1][:,None]>0] = 1

                        if args.fold_idx is not None:
                            save_root_ = os.path.join(args.out_dir, 'sample_validation', args.name, 'fold_'+str(args.fold_idx), 'validation_iter'+str(steps))
                        else:
                            save_root_ = os.path.join(args.out_dir, 'sample_validation', args.name, 'validation_iter'+str(steps))
                        if os.path.exists(os.path.join(save_root_, 'ensembled_pred_sample_'+path[0].split('/')[-1])):
                            continue

                        ensemble_sample_list = []
                        ensemble_combine_list = []
                        for ensemble_ite in range(args.num_ensemble):
                            pred_list = [] # diffusion results
                            # pred_list_progress = {}
                            graph_pred = []
                            # graph_pred_progress = {}
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
                                    # m = mask[S:E,] * liver

                                    c = th.randn_like(b[:, :1, ...])
                                    img = th.cat((b, c), dim=1)     #add a noise channel$

                                    logger.log('sampling '+path[0].split('/')[-1]+' with '+name)

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

                                    # for timestep in sample_prog.keys():
                                    #     sample_prog[timestep] = th.clamp(sample_prog[timestep], 0, 1)
                                    #     sample_prog[timestep][sample_prog[timestep]>=0.5] = 1
                                    #     sample_prog[timestep][sample_prog[timestep]<0.5] = 0
                                    #     sample_prog[timestep] = sample_prog[timestep] * liver.cuda()
                                    #     cal_out_prog[timestep][cal_out_prog[timestep]>=0.5] = 1
                                    #     cal_out_prog[timestep][cal_out_prog[timestep]<0.5] = 0
                                    #     if len(pred_list_progress.keys())!=0:
                                    #         pred_list_progress[timestep] = th.cat((pred_list_progress[timestep].squeeze(1), sample_prog[timestep].squeeze(1)), dim=0)
                                    #         graph_pred_progress[timestep] = th.cat((graph_pred_progress[timestep], cal_out_prog[timestep]), dim=0)
                                    # if len(pred_list_progress.keys())==0:
                                    #     pred_list_progress.update(sample_prog)
                                    #     graph_pred_progress.update(cal_out_prog)

                                    pred_list.append(sample * liver.cuda())
                                    graph_pred.append(cal_out)
                                else:
                                    pred_list.append(th.zeros(liver_mask[S:E,].shape).cuda())
                                    graph_pred.append(th.zeros((liver_mask[S:E,].shape[0], 32,32)).cuda())
                                    # for timestep in ['999','899','799','699','599','499','399','299','199','99','89','79','69','59','49','39','29','19','9','0']:
                                    #     if len(pred_list_progress.keys())==20:
                                    #         pred_list_progress[timestep] = th.cat((pred_list_progress[timestep].squeeze(1), th.zeros(liver_mask[S:E,].shape).cuda().squeeze(1)), dim=0)
                                    #         graph_pred_progress[timestep] = th.cat((graph_pred_progress[timestep], th.zeros((liver_mask[S:E,].shape[0], 32,32)).cuda()), dim=0)
                                    #     else:
                                    #         pred_list_progress[timestep] = th.zeros(liver_mask[S:E,].shape).cuda().squeeze(1)
                                    #         graph_pred_progress[timestep] = th.zeros((liver_mask[S:E,].shape[0], 32,32)).cuda()

                            ensemble_sample_list.append(th.cat(pred_list, dim=0).squeeze(1)) # 256*256*256

                            graph_pred = th.cat(graph_pred, dim=0).detach().cpu().numpy() # (slice_nums,32,32) -> (slice_nums,x,y)
                            if args.fold_idx is not None:
                                os.makedirs('{}/{}/{}/{}/{}'.format(r'/data1/xzhang2/liver_vessel_segmentation/model/GATSegDiff/outputs/logger_validation', args.name, 'pred_nodes_validation', 'fold_'+str(args.fold_idx), 'Iter_'+str(int(re.findall("\d+",name)[0]))), exist_ok=True)
                                nodes_save_root = os.path.join(r'/data1/xzhang2/liver_vessel_segmentation/model/GATSegDiff/outputs/logger_validation', args.name, 'pred_nodes_validation', 'fold_'+str(args.fold_idx), 'Iter_'+str(int(re.findall("\d+",name)[0])))
                            else:
                                os.makedirs('{}/{}/{}/{}'.format(r'/data1/xzhang2/liver_vessel_segmentation/model/GATSegDiff/outputs/logger_validation', args.name, 'pred_nodes_validation', 'Iter_'+str(int(re.findall("\d+",name)[0]))), exist_ok=True)
                                nodes_save_root = os.path.join(r'/data1/xzhang2/liver_vessel_segmentation/model/GATSegDiff/outputs/logger_validation', args.name, 'pred_nodes_validation', 'Iter_'+str(int(re.findall("\d+",name)[0])))
                            np.save(os.path.join(nodes_save_root, 'predNodes_'+path[0].split('/')[-1].replace('_DICOM.nii.gz','.npy')), graph_pred)

                            # graph_pred_progress = {key:graph_pred_progress[key].detach().cpu().numpy() for key in graph_pred_progress.keys()}
                            # if args.fold_idx is not None:
                            #     os.makedirs('{}/{}/{}/{}/{}'.format(r'/exports/lkeb-hpc/xzhang/liver_vessel_segmentation/model/GATSegDiff/outputs/logger_validation', args.name, 'pred_nodes_validation_progress', 'fold_'+str(args.fold_idx), 'Iter_'+str(int(re.findall("\d+",name)[0]))), exist_ok=True)
                            #     nodes_save_root = os.path.join(r'/exports/lkeb-hpc/xzhang/liver_vessel_segmentation/model/GATSegDiff/outputs/logger_validation', args.name, 'pred_nodes_validation_progress', 'fold_'+str(args.fold_idx), 'Iter_'+str(int(re.findall("\d+",name)[0])))
                            # else:
                            #     os.makedirs('{}/{}/{}/{}'.format(r'/exports/lkeb-hpc/xzhang/liver_vessel_segmentation/model/GATSegDiff/outputs/logger_validation', args.name, 'pred_nodes_validation_progress', 'Iter_'+str(int(re.findall("\d+",name)[0]))), exist_ok=True)
                            #     nodes_save_root = os.path.join(r'/exports/lkeb-hpc/xzhang/liver_vessel_segmentation/model/GATSegDiff/outputs/logger_validation', args.name, 'pred_nodes_validation_progress', 'Iter_'+str(int(re.findall("\d+",name)[0])))
                            # np.save(os.path.join(nodes_save_root, 'predNodesDict_'+path[0].split('/')[-1].replace('_DICOM.nii.gz','.npy')), graph_pred_progress)

                            pred_list = th.cat(pred_list, dim=0).squeeze(1).detach().cpu().numpy()

                            if args.fold_idx is not None:
                                save_root = os.path.join(args.out_dir, 'sample_validation', args.name, 'fold_'+str(args.fold_idx), 'validation_iter'+str(steps))
                            else:
                                save_root = os.path.join(args.out_dir, 'sample_validation', args.name, 'validation_iter'+str(steps))
                            'save sampling results per ensemble step'
                            img = sitk.GetImageFromArray(pred_list)
                            img.SetOrigin(origin[0])
                            img.SetDirection(direction[0])
                            img.SetSpacing(space[0])
                            os.makedirs('{}/{}'.format(save_root, 'ensemble_'+str(ensemble_ite)), exist_ok=True)
                            sitk.WriteImage(img, '{}/{}/{}'.format(save_root, 'ensemble_'+str(ensemble_ite), 'pred_sample_'+path[0].split('/')[-1]))
                            'save progressive sampling results per ensemble step'
                            # for timestep in pred_list_progress.keys():
                            #     img = sitk.GetImageFromArray(pred_list_progress[timestep].detach().cpu().numpy())
                            #     img.SetOrigin(origin[0])
                            #     img.SetDirection(direction[0])
                            #     img.SetSpacing(space[0])
                            #     os.makedirs('{}/{}'.format(save_root, 'ensemble_'+str(ensemble_ite)), exist_ok=True)
                            #     sitk.WriteImage(img, '{}/{}/{}'.format(save_root, 'ensemble_'+str(ensemble_ite), 'time_'+timestep+'_pred_sample_'+path[0].split('/')[-1]))

                            if ensemble_ite==0:
                                label = sitk.GetImageFromArray(mask.squeeze(1).type(th.float32).detach().cpu().numpy())
                                label.SetOrigin(origin[0])
                                label.SetDirection(direction[0])
                                label.SetSpacing(space[0])
                                sitk.WriteImage(label, '{}/{}'.format(save_root, 'label_'+path[0].split('/')[-1]))

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

                        raise ValueError(
                                os.path.join(save_root_, 'ensembled_pred_sample_'+path[0].split('/')[-1]) +
                                ' has finished!'
                            )

                elif args.use_ddim:
                    for vessel_ite in range(args.num_samples): # case-level
                        try:
                            # batch, mask, nodes_coord, edge_index, label_slice_idx, path, origin, space, direction, ct_slice_num_total = next(data)  #should return an image from the dataloader "data"
                            batch, mask, nodes_coord, edge_index, path, origin, space, direction = next(data)
                        except StopIteration:
                            data = iter(test_loader)
                            # batch, mask, nodes_coord, edge_index, label_slice_idx, path, origin, space, direction, ct_slice_num_total = next(data)
                            batch, mask, nodes_coord, edge_index, path, origin, space, direction = next(data)

                        liver_mask = th.zeros(batch[:,1][:,None].shape) # (N,1,256,256)
                        liver_mask[batch[:,1][:,None]>0] = 1

                        if args.fold_idx is not None:
                            save_root_ = os.path.join(args.out_dir, 'sample_validation', args.name, 'fold_'+str(args.fold_idx), 'validation_iter'+str(steps))
                        else:
                            save_root_ = os.path.join(args.out_dir, 'sample_validation', args.name, 'validation_iter'+str(steps))
                        if os.path.exists(os.path.join(save_root_, 'ensembled_pred_sample_'+path[0].split('/')[-1])):
                            continue

                        ensemble_sample_list = []
                        ensemble_combine_list = []
                        for ensemble_ite in range(args.num_ensemble):

                            pred_combine_list = [] # combined predict (integrate condition unet results with diffusion results together)
                            pred_list = [] # diffusion results
                            pred_list_progress = {}
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
                                    # m = mask[S:E,] * liver

                                    c = th.randn_like(b[:, :1, ...])
                                    img = th.cat((b, c), dim=1)     #add a noise channel$

                                    logger.log('sampling '+path[0].split('/')[-1]+' with '+os.path.join(model_name,name))

                                    start = th.cuda.Event(enable_timing=True)
                                    end = th.cuda.Event(enable_timing=True)
                                    
                                    # for i in range(args.num_ensemble):  #this is for the generation of an ensemble of 5 masks.
                                    model_kwargs = {}
                                    model_kwargs.update(graph)
                                    start.record()
                                    sample_fn = (
                                        diffusion.ddim_sample_loop_known
                                    )
                                    sample, x_noisy, org, sample_prog = sample_fn(
                                        model,
                                        (batch_size_real, 4, args.image_size, args.image_size), 
                                        img,
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

                                    # if len(label_slice_idx)<ct_slice_num_total[0]: # only perform inference for slices with real ground truth
                                    #     pass
                                    # else:# perform inference for all of the slices
                                    for timestep in sample_prog.keys():
                                        sample_prog[timestep] = th.clamp(sample_prog[timestep], 0, 1)
                                        sample_prog[timestep][sample_prog[timestep]>=0.5] = 1
                                        sample_prog[timestep][sample_prog[timestep]<0.5] = 0
                                        sample_prog[timestep] = sample_prog[timestep] * liver.cuda()
                                        if len(pred_list_progress.keys())!=0:
                                            pred_list_progress[timestep] = th.cat((pred_list_progress[timestep].squeeze(1), sample_prog[timestep].squeeze(1)), dim=0)
                                    if len(pred_list_progress.keys())==0:
                                        pred_list_progress.update(sample_prog)

                                    pred_list.append(sample * liver.cuda())
                                else:
                                    pred_list.append(th.zeros(liver_mask[S:E,].shape).cuda())

                                    for timestep in ['399','299','199','99','89','79','69','59','49','39','29','19','9','0']:
                                        if len(pred_list_progress.keys())==14:
                                            pred_list_progress[timestep] = th.cat((pred_list_progress[timestep].squeeze(1), th.zeros(liver_mask[S:E,].shape).cuda().squeeze(1)), dim=0)
                                        else:
                                            pred_list_progress[timestep] = th.zeros(liver_mask[S:E,].shape).cuda().squeeze(1)

                            # if len(label_slice_idx)<ct_slice_num_total[0]: # only perform inference for slices with real ground truth
                            #     pred_list_cat = th.zeros(ct_slice_num_total[0],1,256,256).cuda()
                            #     pred_list_cat[label_slice_idx,] = th.cat(pred_list, dim=0)
                            #     ensemble_sample_list.append(pred_list_cat.squeeze(1))
                            # else: # perform inference for all of the slices
                            ensemble_sample_list.append(th.cat(pred_list, dim=0).squeeze(1)) # 256*256*256

                            # if len(label_slice_idx)<ct_slice_num_total[0]: # only perform inference for slices with real ground truth
                            #     pred_list = pred_list_cat.squeeze(1).detach().cpu().numpy()
                            #     pass
                            # else:# perform inference for all of the slices

                            pred_list = th.cat(pred_list, dim=0).squeeze(1).detach().cpu().numpy()

                            if args.fold_idx is not None:
                                save_root = os.path.join(args.out_dir, 'sample_validation', args.name, 'fold_'+str(args.fold_idx), 'validation_iter'+str(steps))
                            else:
                                save_root = os.path.join(args.out_dir, 'sample_validation', args.name, 'validation_iter'+str(steps))
                            'save sampling results per ensemble step'
                            img = sitk.GetImageFromArray(pred_list)
                            img.SetOrigin(origin[0])
                            img.SetDirection(direction[0])
                            img.SetSpacing(space[0])
                            os.makedirs('{}/{}'.format(save_root, 'ensemble_'+str(ensemble_ite)), exist_ok=True)
                            sitk.WriteImage(img, '{}/{}/{}'.format(save_root, 'ensemble_'+str(ensemble_ite), 'pred_sample_'+path[0].split('/')[-1]))
                            'save progressive sampling results per ensemble step'
                            # if len(label_slice_idx)<ct_slice_num_total[0]: # only perform inference for slices with real ground truth
                            #     pass
                            # else:# perform inference for all of the slices
                            for timestep in pred_list_progress.keys():
                                img = sitk.GetImageFromArray(pred_list_progress[timestep].detach().cpu().numpy())
                                img.SetOrigin(origin[0])
                                img.SetDirection(direction[0])
                                img.SetSpacing(space[0])
                                os.makedirs('{}/{}'.format(save_root, 'ensemble_'+str(ensemble_ite)), exist_ok=True)
                                sitk.WriteImage(img, '{}/{}/{}'.format(save_root, 'ensemble_'+str(ensemble_ite), 'time_'+timestep+'_pred_sample_'+path[0].split('/')[-1]))

                            if ensemble_ite==0:
                                # mask_full = th.zeros(ct_slice_num_total[0],1,256,256).type(th.LongTensor)
                                # mask_full[label_slice_idx,] = mask
                                # label = sitk.GetImageFromArray(mask_full.squeeze(1).type(th.float32).detach().cpu().numpy())
                                label = sitk.GetImageFromArray(mask.squeeze(1).type(th.float32).detach().cpu().numpy())
                                label.SetOrigin(origin[0])
                                label.SetDirection(direction[0])
                                label.SetSpacing(space[0])
                                sitk.WriteImage(label, '{}/{}'.format(save_root, 'label_'+path[0].split('/')[-1]))

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

                        raise ValueError(
                                os.path.join(save_root_, 'ensembled_pred_sample_' + path[0].split('/')[-1]) +
                                ' has finished!'
                            )

                else: # Plain DDPM sampling
                    if not args.uncertainty_control:
                        for vessel_ite in range(args.num_samples): # case-level
                            try:
                                # batch, mask, nodes_coord, edge_index, label_slice_idx, path, origin, space, direction, ct_slice_num_total = next(data)  #should return an image from the dataloader "data"
                                batch, mask, nodes_coord, edge_index, path, origin, space, direction = next(data)
                            except StopIteration:
                                data = iter(test_loader)
                                # batch, mask, nodes_coord, edge_index, label_slice_idx, path, origin, space, direction, ct_slice_num_total = next(data)
                                batch, mask, nodes_coord, edge_index, path, origin, space, direction = next(data)

                            liver_mask = th.zeros(batch[:,1][:,None].shape) # (N,1,256,256)
                            liver_mask[batch[:,1][:,None]>0] = 1

                            if args.fold_idx is not None:
                                save_root_ = os.path.join(args.out_dir, 'sample_validation', args.name, 'fold_'+str(args.fold_idx), 'validation_iter'+str(steps))
                            else:
                                save_root_ = os.path.join(args.out_dir, 'sample_validation', args.name, 'validation_iter'+str(steps))
                            if os.path.exists(os.path.join(save_root_, 'ensembled_pred_sample_'+path[0].split('/')[-1])):
                                continue
                            logger.log("current inference is for " + path[0].split('/')[-1] +' !')

                            ensemble_sample_list = []
                            ensemble_combine_list = []
                            for ensemble_ite in range(args.num_ensemble):

                                pred_combine_list = [] # combined predict (integrate condition unet results with diffusion results together)
                                pred_list = [] # diffusion results
                                pred_list_progress = {}
                                graph_pred = []
                                graph_pred_progress = {}
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
                                        # m = mask[S:E,] * liver

                                        c = th.randn_like(b[:, :1, ...])
                                        img = th.cat((b, c), dim=1)     #add a noise channel$

                                        logger.log('sampling '+path[0].split('/')[-1]+' with '+os.path.join(model_name,name))

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

                                        # if len(label_slice_idx)<ct_slice_num_total[0]: # only perform inference for slices with real ground truth
                                        #     pass
                                        # else:# perform inference for all of the slices
                                        for timestep in sample_prog.keys():
                                            'debug'
                                            # sample_prog[timestep] = th.clamp(sample_prog[timestep], 0, 1)
                                            # sample_prog[timestep][sample_prog[timestep]>=0.5] = 1
                                            # sample_prog[timestep][sample_prog[timestep]<0.5] = 0
                                            sample_prog[timestep] = sample_prog[timestep] * liver.cuda()
                                            cal_out_prog[timestep][cal_out_prog[timestep]>=0.5] = 1
                                            cal_out_prog[timestep][cal_out_prog[timestep]<0.5] = 0
                                            if len(pred_list_progress.keys())!=0:
                                                pred_list_progress[timestep] = th.cat((pred_list_progress[timestep].squeeze(1), sample_prog[timestep].squeeze(1)), dim=0)
                                                graph_pred_progress[timestep] = th.cat((graph_pred_progress[timestep], cal_out_prog[timestep]), dim=0)
                                        if len(pred_list_progress.keys())==0:
                                            pred_list_progress.update(sample_prog)
                                            graph_pred_progress.update(cal_out_prog)

                                        pred_list.append(sample * liver.cuda())
                                        graph_pred.append(cal_out)
                                    else:
                                        pred_list.append(th.zeros(liver_mask[S:E,].shape).cuda())
                                        graph_pred.append(th.zeros((liver_mask[S:E,].shape[0], 32,32)).cuda())
                                        # if len(label_slice_idx)<ct_slice_num_total[0]: # only perform inference for slices with real ground truth
                                        #     pass
                                        # else:# perform inference for all of the slices
                                        for timestep in ['999','899','799','699','599','499','399','299','199','99','89','79','69','59','49','39','29','19','9','0']:
                                            if len(pred_list_progress.keys())==20:
                                                pred_list_progress[timestep] = th.cat((pred_list_progress[timestep].squeeze(1), th.zeros(liver_mask[S:E,].shape).cuda().squeeze(1)), dim=0)
                                                graph_pred_progress[timestep] = th.cat((graph_pred_progress[timestep], th.zeros((liver_mask[S:E,].shape[0], 32,32)).cuda()), dim=0)
                                            else:
                                                pred_list_progress[timestep] = th.zeros(liver_mask[S:E,].shape).cuda().squeeze(1)
                                                graph_pred_progress[timestep] = th.zeros((liver_mask[S:E,].shape[0], 32,32)).cuda()

                                ensemble_sample_list.append(th.cat(pred_list, dim=0).squeeze(1)) # 256*256*256

                                # graph_pred = th.cat(graph_pred, dim=0).detach().cpu().numpy() # (slice_nums,32,32) -> (slice_nums,x,y)
                                # if args.fold_idx is not None:
                                #     os.makedirs('{}/{}/{}/{}/{}'.format(r'/data1/xzhang2/liver_vessel_segmentation/model/GATSegDiff/outputs/logger_validation', args.name, 'pred_nodes_validation', 'fold_'+str(args.fold_idx), 'Iter_'+str(int(re.findall("\d+",name)[0]))), exist_ok=True)
                                #     nodes_save_root = os.path.join(r'/data1/xzhang2/liver_vessel_segmentation/model/GATSegDiff/outputs/logger_validation', args.name, 'pred_nodes_validation', 'fold_'+str(args.fold_idx), 'Iter_'+str(int(re.findall("\d+",name)[0])))
                                # else:
                                #     os.makedirs('{}/{}/{}/{}'.format(r'/data1/xzhang2/liver_vessel_segmentation/model/GATSegDiff/outputs/logger_validation', args.name, 'pred_nodes_validation', 'Iter_'+str(int(re.findall("\d+",name)[0]))), exist_ok=True)
                                #     nodes_save_root = os.path.join(r'/data1/xzhang2/liver_vessel_segmentation/model/GATSegDiff/outputs/logger_validation', args.name, 'pred_nodes_validation', 'Iter_'+str(int(re.findall("\d+",name)[0])))
                                # np.save(os.path.join(nodes_save_root, 'predNodes_'+path[0].split('/')[-1].replace('_DICOM.nii.gz','.npy')), graph_pred)

                                # graph_pred_progress = {key:graph_pred_progress[key].detach().cpu().numpy() for key in graph_pred_progress.keys()}
                                # if args.fold_idx is not None:
                                #     os.makedirs('{}/{}/{}/{}/{}'.format(r'/data1/xzhang2/liver_vessel_segmentation/model/GATSegDiff/outputs/logger_validation', args.name, 'pred_nodes_validation_progress', 'fold_'+str(args.fold_idx), 'Iter_'+str(int(re.findall("\d+",name)[0]))), exist_ok=True)
                                #     nodes_save_root = os.path.join(r'/data1/xzhang2/liver_vessel_segmentation/model/GATSegDiff/outputs/logger_validation', args.name, 'pred_nodes_validation_progress', 'fold_'+str(args.fold_idx), 'Iter_'+str(int(re.findall("\d+",name)[0])))
                                # else:
                                #     os.makedirs('{}/{}/{}/{}'.format(r'/data1/xzhang2/liver_vessel_segmentation/model/GATSegDiff/outputs/logger_validation', args.name, 'pred_nodes_validation_progress', 'Iter_'+str(int(re.findall("\d+",name)[0]))), exist_ok=True)
                                #     nodes_save_root = os.path.join(r'/data1/xzhang2/liver_vessel_segmentation/model/GATSegDiff/outputs/logger_validation', args.name, 'pred_nodes_validation_progress', 'Iter_'+str(int(re.findall("\d+",name)[0])))
                                # np.save(os.path.join(nodes_save_root, 'predNodesDict_'+path[0].split('/')[-1].replace('_DICOM.nii.gz','.npy')), graph_pred_progress)

                                pred_list = th.cat(pred_list, dim=0).squeeze(1).detach().cpu().numpy()

                                if args.fold_idx is not None:
                                    save_root = os.path.join(args.out_dir, 'sample_validation', args.name, 'fold_'+str(args.fold_idx), 'validation_iter'+str(steps))
                                else:
                                    save_root = os.path.join(args.out_dir, 'sample_validation', args.name, 'validation_iter'+str(steps))
                                'save sampling results per ensemble step'
                                img = sitk.GetImageFromArray(pred_list)
                                img.SetOrigin(origin[0])
                                img.SetDirection(direction[0])
                                img.SetSpacing(space[0])
                                os.makedirs('{}/{}'.format(save_root, 'ensemble_'+str(ensemble_ite)), exist_ok=True)
                                sitk.WriteImage(img, '{}/{}/{}'.format(save_root, 'ensemble_'+str(ensemble_ite), 'pred_sample_'+path[0].split('/')[-1]))
                                'save progressive sampling results per ensemble step'
                                # if len(label_slice_idx)<ct_slice_num_total[0]: # only perform inference for slices with real ground truth
                                #     pass
                                # else:# perform inference for all of the slices
                                for timestep in pred_list_progress.keys():
                                    img = sitk.GetImageFromArray(pred_list_progress[timestep].detach().cpu().numpy())
                                    img.SetOrigin(origin[0])
                                    img.SetDirection(direction[0])
                                    img.SetSpacing(space[0])
                                    os.makedirs('{}/{}'.format(save_root, 'ensemble_'+str(ensemble_ite)), exist_ok=True)
                                    sitk.WriteImage(img, '{}/{}/{}'.format(save_root, 'ensemble_'+str(ensemble_ite), 'time_'+timestep+'_pred_sample_'+path[0].split('/')[-1]))

                                if ensemble_ite==0:
                                    # mask_full = th.zeros(ct_slice_num_total[0],1,256,256).type(th.LongTensor)
                                    # mask_full[label_slice_idx,] = mask
                                    # label = sitk.GetImageFromArray(mask_full.squeeze(1).type(th.float32).detach().cpu().numpy())
                                    label = sitk.GetImageFromArray(mask.squeeze(1).type(th.float32).detach().cpu().numpy())
                                    label.SetOrigin(origin[0])
                                    label.SetDirection(direction[0])
                                    label.SetSpacing(space[0])
                                    sitk.WriteImage(label, '{}/{}'.format(save_root, 'label_'+path[0].split('/')[-1]))

                            'save ensembed sampling results'
                            ensemble_sample = staple(th.stack(ensemble_sample_list,dim=0)) # sliceNum*256*256
                            ensemble_sample = th.clamp(ensemble_sample, 0, 1).squeeze(0).detach().cpu().numpy()
                            ensemble_sample[ensemble_sample>=0.5] = 1
                            ensemble_sample[ensemble_sample<0.5] = 0

                            # Perform STAPLE ensemble
                            # ensemble_sample = staple_ensemble_tensor(th.stack(ensemble_sample_list,dim=0))
                            # ensemble_sample = (ensemble_sample > 0.5).astype(np.uint8)


                            img = sitk.GetImageFromArray(ensemble_sample)
                            img.SetOrigin(origin[0])
                            img.SetDirection(direction[0])
                            img.SetSpacing(space[0])
                            sitk.WriteImage(img, '{}/{}'.format(save_root, 'ensembled_pred_sample_'+path[0].split('/')[-1]))

                            raise ValueError(
                                    os.path.join(save_root_, 'ensembled_pred_sample_' + path[0].split('/')[-1]) +
                                    ' has finished!'
                                )
                    elif args.uncertainty_control:
                        for vessel_ite in range(args.num_samples): # case-level
                            try:
                                batch, mask, nodes_coord, edge_index, path, origin, space, direction = next(data)
                            except StopIteration:
                                data = iter(test_loader)
                                batch, mask, nodes_coord, edge_index, path, origin, space, direction = next(data)

                            liver_mask = th.zeros(batch[:,1][:,None].shape) # (N,1,256,256)
                            liver_mask[batch[:,1][:,None]>0] = 1

                            if args.fold_idx is not None:
                                save_root_ = os.path.join(args.out_dir, 'sample_validation', args.name, 'fold_'+str(args.fold_idx), 'validation_iter'+str(steps))
                            else:
                                save_root_ = os.path.join(args.out_dir, 'sample_validation', args.name, 'validation_iter'+str(steps))
                            # if os.path.exists(os.path.join(save_root_, 'ensembled_pred_sample_PATIENT_'+re.findall(r"\d+",path[0].split('/')[-1])[0]+'.nii.gz')):
                            #     continue

                            ensemble_sample_list = []
                            pred_list = [] # diffusion results
                            pred_list_progress = {}
                            for bi in range(int(np.ceil(batch.shape[0]/args.batch_size_sample))): # per vessel
                                S = bi*args.batch_size_sample
                                E = (bi+1)*args.batch_size_sample

                                if E>batch.shape[0]:
                                    E = batch.shape[0]
                                batch_size_real = E-S
                                'graph used in the inference is the fully connected grid'
                                graph = {'nodes':nodes_coord.repeat(batch_size_real,1,1).cuda(), 'edges':[edge_index[0].cuda() for i in range(batch_size_real)]}

                                if liver_mask[S:E,].sum()>0:
                                # if liver_mask[S:E,].sum()>30000:

                                    liver = liver_mask[S:E,]
                                    b = batch[S:E,] * liver
                                    # m = mask[S:E,] * liver

                                    c = th.randn_like(b[:, :1, ...])
                                    img = th.cat((b, c), dim=1)     #add a noise channel$

                                    # logger.log("sampling...")
                                    logger.log('sampling '+path[0].split('/')[-1]+' with '+name)

                                    start = th.cuda.Event(enable_timing=True)
                                    end = th.cuda.Event(enable_timing=True)
                                    
                                    model_kwargs = {}
                                    model_kwargs.update(graph)
                                    start.record()
                                    sample_fn = (
                                        diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
                                    )

                                    sample = sample_fn(
                                        model,
                                        (batch_size_real, 4, args.image_size, args.image_size), 
                                        img,
                                        step = args.diffusion_steps,
                                        clip_denoised=args.clip_denoised,
                                        model_kwargs=model_kwargs,
                                        progress=True,
                                        custom_model=custom_model,
                                        config=config,
                                        fixed_xT=c,
                                        diffusion=diffusion,
                                        num_rounds=args.num_ensemble,
                                        step_control=args.step_control,
                                    )
                                    end.record()
                                    th.cuda.synchronize()
                                    print('time for 1 sample', start.elapsed_time(end))  #time measurement for the generation of 1 sample

                                    sample = th.clamp(sample, 0, 1)
                                    sample[sample>=0.5] = 1
                                    sample[sample<0.5] = 0
                                    pred_list.append(sample * liver.cuda())
                                else:
                                    pred_list.append(th.zeros((E-S, args.num_ensemble, config.data.image_size, config.data.image_size)).cuda())

                            ensemble_sample = th.cat(pred_list, dim=0) # (slices,num_ensemble,256,256)

                            for ensemble_ite in range(ensemble_sample.shape[1]):

                                pred_per_ensem = ensemble_sample[:,ensemble_ite,].detach().cpu().numpy() # (slices,256,256)
                                if args.fold_idx is not None:
                                    save_root = os.path.join(args.out_dir, 'sample_validation', args.name, 'fold_'+str(args.fold_idx), 'validation_iter'+str(steps))
                                else:
                                    save_root = os.path.join(args.out_dir, 'sample_validation', args.name, 'validation_iter'+str(steps))

                                'save sampling results per ensemble step'
                                img = sitk.GetImageFromArray(pred_per_ensem)
                                img.SetOrigin(origin[0])
                                img.SetDirection(direction[0])
                                img.SetSpacing(space[0])
                                os.makedirs('{}/{}'.format(save_root, 'ensemble_'+str(ensemble_ite)), exist_ok=True)
                                sitk.WriteImage(img, '{}/{}/{}'.format(save_root, 'ensemble_'+str(ensemble_ite), 'pred_sample_'+path[0].split('/')[-1]))

                                if ensemble_ite==0:
                                    label = sitk.GetImageFromArray(mask.squeeze(1).type(th.float32).detach().cpu().numpy())
                                    label.SetOrigin(origin[0])
                                    label.SetDirection(direction[0])
                                    label.SetSpacing(space[0])
                                    sitk.WriteImage(label, '{}/{}'.format(save_root, 'label_'+path[0].split('/')[-1]))

                            'save sampling results with smallest uncertainty'
                            uncertainty_control_sample = ensemble_sample[:,0,].detach().cpu().numpy() # (num_slices,256,256)

                            img = sitk.GetImageFromArray(uncertainty_control_sample)
                            img.SetOrigin(origin[0])
                            img.SetDirection(direction[0])
                            img.SetSpacing(space[0])
                            sitk.WriteImage(img, '{}/{}'.format(save_root, 'ensembled_pred_sample_'+path[0].split('/')[-1]))

                            raise ValueError(
                                    os.path.join(save_root_, 'ensembled_pred_sample_' + path[0].split('/')[-1]) +
                                    ' has finished!'
                                )


def create_argparser():
    defaults = dict(
        name='temp',
        model_name='',
        data_name = 'LiVS',
        seed=10,
        clip_denoised=True,
        num_samples=-1, # validation vessel number
        batch_size=1, # load a full CT volme in the dataloader
        batch_size_sample=32, # how many slices be involved in one sampling iteration
        log_interval=20000,
        start_log_step=80000,
        end_log_step=80000,
        diffusion_steps=1000,
        use_ddim=False,
        dpm_solver=False,
        uncertainty_control=False,
        num_ensemble=1,      #number of samples in the ensemble
        step_control=-1,
        gpu_dev = "0",
        multi_gpu = None, #"0,1,2"
        test_root_path = '/data1/xzhang2/liver_vessel_segmentation/data/LiVS',
        out_dir='/data1/xzhang2/liver_vessel_segmentation/model/GATSegDiff/outputs',
        debug = False,
        test_folder=[],
        fold=-1, # how many folds in cross validation
        fold_idx=None,
        inFold_subset=1, # how many subset splited for a single fold 
        inFold_sub_idx=None
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main_GAT()
