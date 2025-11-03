import os
import numpy as np
import SimpleITK as sitk
import argparse
import random
import re
from guided_diffusion.script_util import add_dict_to_argparser
from scipy.ndimage import gaussian_filter
import cv2
from skimage.restoration import denoise_tv_chambolle


def create_argparser():
    defaults = dict(
        infer_name='',
        fold=-1, # how many folds in cross validation
        fold_idx=None,
        inFold_subset=1, # how many subset splited for a single fold 
        inFold_sub_idx=None,
        iters=-1
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def noise_remove_connected_region_2(): # skimage.measure.label & scipy.ndimage.label
    from skimage import morphology, measure
    from scipy.ndimage import label, generate_binary_structure
    def find_outliers_iqr(data):
        # Calculate the interquartile range
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr_value = q3 - q1
        # Define the lower and upper bounds for outliers
        lower_bound = q1 - 1.5 * iqr_value
        upper_bound = q3 + 1.5 * iqr_value
        # Identify outliers
        outliers = [x for x in data if x < lower_bound or x > upper_bound]
        return outliers
    
    args = create_argparser().parse_args()

    infer_name = args.infer_name
    fold_idx = int(args.fold_idx)
    inFold_subset = int(args.inFold_subset)
    inFold_sub_idx = int(args.inFold_sub_idx)
    fold = int(args.fold)
    iters = int(args.iters)

    # Read the segmented 3D binary volume
    root_path = os.path.join('/home/xzhang2/data1/liver_vessel_segmentation/model/GATSegDiff/outputs/sample_validation',infer_name,'fold_'+str(fold_idx),'validation_iter'+str(iters))

    case_idx = []
    for root, dirs, files in os.walk('/data1/xzhang2/liver_vessel_segmentation/data/LiVS'): 
        if root.split('/')[-1]=='Test25DCT_CTVolume_3slices': 
            for name in files:                                                    
                case_i = name.split('.')[0]
                if case_i not in case_idx:
                    case_idx.append(case_i)
    case_idx = sorted(case_idx)
    case_idx = np.array(case_idx)

    random.seed(0)
    random.shuffle(case_idx)
    case_idx_split = np.split(np.array(case_idx), fold)
    test_folder_whole = list(case_idx_split[fold_idx]) # 101

    test_folder_sub = list(np.array_split(np.array(test_folder_whole), inFold_subset)[inFold_sub_idx])
    test_folder = sorted(test_folder_sub)
    print(test_folder)

    for root, dirs, files in os.walk(root_path): 
        # if 'validation_iter' not in root.split('/')[-1]:
        #     continue
        if 'ensembled_pred_staple' != root.split('/')[-1]:
            continue
        # if 'ensembled_pred_majorityVoting' != root.split('/')[-1]:
        #     continue
        steps = int(re.findall("\d+",root.split('/')[-2])[0])
        if steps != iters:
            continue
        for name in files:
            if os.path.exists(os.path.join(root, name.replace('ensembled','noiseCancelConnect'))):
                continue
            if 'case'+str(re.findall("\d+",name)[0]).zfill(4) not in test_folder:
                continue
            # if 'resolution_recovery_validation' not in root.split('\\')[-1]:
            #     continue
            if 'ensembled_pred_sample' not in name:
                continue

            segmented_volume = sitk.ReadImage(os.path.join(root, name))
            origin = segmented_volume.GetOrigin()
            space = segmented_volume.GetSpacing()
            direction = segmented_volume.GetDirection()
            volume_unit = np.round(space[0]*space[1]*space[2], 4) # mm3

            segmented_volume = sitk.GetArrayFromImage(segmented_volume)
            structure = generate_binary_structure(3, 1)
            opened_volume, num_features = label(segmented_volume, structure=structure)

            component_size_list = []
            accumulated_img = np.zeros_like(segmented_volume)
            for label_item in range(1, np.max(opened_volume) + 1):
                component_mask = np.uint8(opened_volume == label_item)
                component_size = np.sum(component_mask)
                accumulated_img[opened_volume == label_item] = component_size
                component_size_list.append(component_size)
            

            accumulated_img = (accumulated_img-accumulated_img.min()) / (accumulated_img.max()-accumulated_img.min())
            accumulated_img[accumulated_img>0.01] = 1
            accumulated_img[accumulated_img<=0.01] = 0

            img = sitk.GetImageFromArray(accumulated_img)
            img.SetOrigin(origin)
            img.SetDirection(direction)
            img.SetSpacing(space)
            save_path = os.path.join(root, name.replace('ensembled','noiseCancelConnect'))
            sitk.WriteImage(img, '{}'.format(save_path))

            raise ValueError(
                    os.path.join(root, name.replace('ensembled','noiseCancelConnect')) +
                    ' has finished!'
                )

def IRCADB_noise_remove_connected_region_2(): # skimage.measure.label & scipy.ndimage.label
    from skimage import morphology, measure
    from scipy.ndimage import label, generate_binary_structure
    def find_outliers_iqr(data):
        # Calculate the interquartile range
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr_value = q3 - q1
        # Define the lower and upper bounds for outliers
        lower_bound = q1 - 1.5 * iqr_value
        upper_bound = q3 + 1.5 * iqr_value
        # Identify outliers
        outliers = [x for x in data if x < lower_bound or x > upper_bound]
        return outliers
    
    args = create_argparser().parse_args()

    infer_name = args.infer_name
    fold_idx = int(args.fold_idx)
    inFold_subset = int(args.inFold_subset)
    inFold_sub_idx = int(args.inFold_sub_idx)
    fold = int(args.fold)
    iters = int(args.iters)

    fixed_cases_in_trainset = ['PATIENT01','PATIENT02','PATIENT03','PATIENT05','PATIENT07','PATIENT09','PATIENT10','PATIENT12','PATIENT13',
                               'PATIENT14','PATIENT15','PATIENT17','PATIENT18','PATIENT19','PATIENT20']

    # Read the segmented 3D binary volume
    root_path = os.path.join('/home/xzhang2/data1/liver_vessel_segmentation/model/GATSegDiff/outputs/sample_validation',infer_name,'fold_'+str(fold_idx)+'_check','validation_iter'+str(iters))

    case_idx = []
    for root, dirs, files in os.walk('/data1/xzhang2/liver_vessel_segmentation/data/3DircadbNifti_newLiverMask'): 
        if root.split('/')[-1]=='Resize25DCT_patientCTVolume_3slices': 
            for name in files:                                                    
                case_i = name.split('_')[0]
                if case_i not in case_idx and case_i not in fixed_cases_in_trainset:
                    case_idx.append(case_i)
    case_idx = sorted(case_idx)
    case_idx = np.array(case_idx)

    random.seed(0)
    random.shuffle(case_idx)
    case_idx_split = np.split(np.array(case_idx), fold)
    test_folder_whole = list(case_idx_split[fold_idx]) # 101

    test_folder_sub = list(np.array_split(np.array(test_folder_whole), inFold_subset)[inFold_sub_idx])
    test_folder = sorted(test_folder_sub)
    print(test_folder)

    for root, dirs, files in os.walk(root_path): 
        # if 'validation_iter' not in root.split('/')[-1]:
        #     continue
        if 'ensembled_pred_staple' != root.split('/')[-1]:
            continue
        # if 'ensembled_pred_majorityVoting' != root.split('/')[-1]:
        #     continue
        steps = int(re.findall("\d+",root.split('/')[-2])[0])
        if steps != iters:
            continue
        for name in files:
            if os.path.exists(os.path.join(root, name.replace('ensembled','noiseCancelConnect'))):
                continue
            if 'PATIENT'+str(re.findall("\d+",name)[0]).zfill(2) not in test_folder:
                continue
            # if 'resolution_recovery_validation' not in root.split('\\')[-1]:
            #     continue
            if 'ensembled_pred_sample' not in name:
                continue

            segmented_volume = sitk.ReadImage(os.path.join(root, name))
            origin = segmented_volume.GetOrigin()
            space = segmented_volume.GetSpacing()
            direction = segmented_volume.GetDirection()
            volume_unit = np.round(space[0]*space[1]*space[2], 4) # mm3

            segmented_volume = sitk.GetArrayFromImage(segmented_volume)
            structure = generate_binary_structure(3, 1)
            opened_volume, num_features = label(segmented_volume, structure=structure)

            component_size_list = []
            accumulated_img = np.zeros_like(segmented_volume)
            for label_item in range(1, np.max(opened_volume) + 1):
                component_mask = np.uint8(opened_volume == label_item)
                component_size = np.sum(component_mask)
                accumulated_img[opened_volume == label_item] = component_size
                component_size_list.append(component_size)
            

            accumulated_img = (accumulated_img-accumulated_img.min()) / (accumulated_img.max()-accumulated_img.min())
            accumulated_img[accumulated_img>0.01] = 1
            accumulated_img[accumulated_img<=0.01] = 0

            img = sitk.GetImageFromArray(accumulated_img)
            img.SetOrigin(origin)
            img.SetDirection(direction)
            img.SetSpacing(space)
            save_path = os.path.join(root, name.replace('ensembled','noiseCancelConnect'))
            sitk.WriteImage(img, '{}'.format(save_path))

            raise ValueError(
                    os.path.join(root, name.replace('ensembled','noiseCancelConnect')) +
                    ' has finished!'
                )


if __name__ == "__main__":
    ## 'LiVS noise cancelling'
    noise_remove_connected_region_2()

    ## 'IRCADB noise cancelling'
    # IRCADB_noise_remove_connected_region_2()