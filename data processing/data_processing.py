import numpy as np
import os
import nibabel
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('Qt5Agg')
import cv2
import SimpleITK as sitk
import monai
import torch
from torchmetrics import Dice, Recall, Precision, Specificity
import skfmm
import numpy as np
from skimage.morphology import skeletonize, skeletonize_3d
from scipy.ndimage import distance_transform_edt, binary_erosion
import tkinter as tk

SMOOTH = 1e-6  # Small smoothing factor to avoid division by zero

def calculate_rates_from_arrays(predictions, ground_truth):
    """
    Calculate TPR, FPR, TNR, and FNR from prediction and ground truth arrays.

    Parameters:
        predictions (numpy.ndarray): Binary array of model predictions (0 or 1).
        ground_truth (numpy.ndarray): Binary array of ground truth labels (0 or 1).

    Returns:
        dict: A dictionary containing TPR, FPR, TNR, FNR.
    """
    # Ensure binary arrays
    predictions = np.asarray(predictions).astype(int)
    ground_truth = np.asarray(ground_truth).astype(int)

    # Calculate TP, FP, TN, FN
    tp = np.sum((predictions == 1) & (ground_truth == 1))
    fp = np.sum((predictions == 1) & (ground_truth == 0))
    tn = np.sum((predictions == 0) & (ground_truth == 0))
    fn = np.sum((predictions == 0) & (ground_truth == 1))

    # Avoid division by zero
    tpr = tp / (tp + fn) if (tp + fn) != 0 else 0  # True Positive Rate
    fpr = fp / (fp + tn) if (fp + tn) != 0 else 0  # False Positive Rate
    tnr = tn / (tn + fp) if (tn + fp) != 0 else 0  # True Negative Rate
    fnr = fn / (tp + fn) if (tp + fn) != 0 else 0  # False Negative Rate

    return {
        "TPR": tpr,
        "FPR": fpr,
        "TNR": tnr,
        "FNR": fnr,
    }

def iou_numpy_3d(outputs: np.array, labels: np.array):
    """
    Calculate IoU for 3D binary segmentation.
    
    Parameters:
    - outputs: Binary predicted segmentation (B, D, H, W).
    - labels: Binary ground truth segmentation (B, D, H, W).
    
    Returns:
    - IoU for each sample in the batch.
    """
    # Ensure both outputs and labels have the same shape
    if outputs.shape != labels.shape:
        raise ValueError("Shape mismatch between outputs and labels")
    
    # Calculate intersection and union for 3D volumes across D, H, W dimensions
    intersection = np.logical_and(outputs, labels).sum((1, 2, 3))  # Sum over D, H, W
    union = np.logical_or(outputs, labels).sum((1, 2, 3))  # Sum over D, H, W
    
    # Compute IoU for each batch element
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    
    return iou

def cl_score(v, s):
    """[this function computes the skeleton volume overlap]

    Args:
        v ([bool]): [image]
        s ([bool]): [skeleton]

    Returns:
        [float]: [computed skeleton volume intersection]
    """
    return np.sum(v*s)/np.sum(s)

def clDice(v_p, v_l):
    """[this function computes the cldice metric]

    Args:
        v_p ([bool]): [predicted image]
        v_l ([bool]): [ground truth image]

    Returns:
        [float]: [cldice metric]
    """
    if len(v_p.shape)==2:
        tprec = cl_score(v_p,skeletonize(v_l))
        tsens = cl_score(v_l,skeletonize(v_p))
    elif len(v_p.shape)==3:
        tprec = cl_score(v_p,skeletonize_3d(v_l))
        tsens = cl_score(v_l,skeletonize_3d(v_p))
    return 2*tprec*tsens/(tprec+tsens)

def calculate_surface_dice_3d(pred_mask, gt_mask, tolerance=2):
    """
    Calculate the Surface Dice between a predicted and ground truth binary mask.

    Parameters:
    - pred_mask (ndarray): Predicted binary mask (3D array).
    - gt_mask (ndarray): Ground truth binary mask (3D array).
    - tolerance (int): Maximum allowed distance for a match.

    Returns:
    - surface_dice (float): Surface Dice score.
    """
    # Extract surface points using binary erosion
    
    # Ensure input masks are binary (0 and 1) and boolean
    pred_mask = (pred_mask > 0).astype(bool)
    gt_mask = (gt_mask > 0).astype(bool)

    pred_surface = pred_mask ^ binary_erosion(pred_mask)
    gt_surface = gt_mask ^ binary_erosion(gt_mask)
    
    # Distance transform for ground truth and predicted surfaces
    gt_distance = distance_transform_edt(~gt_surface)
    pred_distance = distance_transform_edt(~pred_surface)
    
    # Calculate the number of surface points within tolerance
    pred_within_tol = (pred_surface & (gt_distance <= tolerance)).sum()
    gt_within_tol = (gt_surface & (pred_distance <= tolerance)).sum()
    
    # Total number of surface points
    total_pred_surface = pred_surface.sum()
    total_gt_surface = gt_surface.sum()
    
    # Surface Dice calculation
    if total_pred_surface + total_gt_surface == 0:
        return 1.0  # Perfect match if no surface points
    
    surface_dice = 2 * (pred_within_tol + gt_within_tol) / (total_pred_surface + total_gt_surface)
    return surface_dice

def calculate_surface_dice_2d(pred_mask, gt_mask, tolerance=2):
    """
    Calculate the Surface Dice between a predicted and ground truth binary mask in 2D.

    Parameters:
    - pred_mask (ndarray): Predicted binary mask (2D array).
    - gt_mask (ndarray): Ground truth binary mask (2D array).
    - tolerance (int): Maximum allowed distance for a match.

    Returns:
    - surface_dice (float): Surface Dice score.
    """
    # Ensure binary masks are boolean
    pred_mask = (pred_mask > 0).astype(bool)
    gt_mask = (gt_mask > 0).astype(bool)

    # Check for matching dimensions
    if pred_mask.shape != gt_mask.shape:
        raise ValueError("Predicted and ground truth masks must have the same shape!")

    # Extract surface points
    pred_surface = pred_mask & ~binary_erosion(pred_mask)
    gt_surface = gt_mask & ~binary_erosion(gt_mask)

    # Distance transform for ground truth and predicted surfaces
    gt_distance = distance_transform_edt(~gt_surface)
    pred_distance = distance_transform_edt(~pred_surface)

    # Surface points within tolerance
    pred_within_tol = (pred_surface & (gt_distance <= tolerance)).sum()
    gt_within_tol = (gt_surface & (pred_distance <= tolerance)).sum()

    # Total surface points
    total_pred_surface = pred_surface.sum()
    total_gt_surface = gt_surface.sum()

    # Ensure denominator is not zero
    if total_pred_surface + total_gt_surface == 0:
        return 1.0  # Perfect match if no surface points exist
    
    # Surface Dice calculation
    surface_dice = 1 * (pred_within_tol + gt_within_tol) / (total_pred_surface + total_gt_surface)
    return surface_dice

def compute_boundary_iou(pred_mask, gt_mask):
    # Ensure the masks are binary (0 or 255)
    pred_mask = (pred_mask > 0).astype(np.uint8)  # Convert to binary
    gt_mask = (gt_mask > 0).astype(np.uint8)  # Convert to binary

    # Use Canny edge detection to get the boundaries
    pred_edges = cv2.Canny(pred_mask, threshold1=100, threshold2=200)
    gt_edges = cv2.Canny(gt_mask, threshold1=100, threshold2=200)

    # Compute intersection and union of the boundaries
    intersection = np.sum(np.logical_and(pred_edges, gt_edges))  # True positives: where both predicted and ground truth boundaries overlap
    union = np.sum(np.logical_or(pred_edges, gt_edges))  # Total boundary pixels: where either predicted or ground truth has a boundary pixel

    # Compute Boundary IoU
    if union == 0:
        return 0.0  # Avoid division by zero
    boundary_iou = intersection / union

    return boundary_iou

def dice(preds,target):
    """2TP / (2TP + FP + FN)"""
    dice_m = Dice(average='micro',ignore_index=0).cuda() # exclude the background
    return dice_m(preds,target)

def recall(preds,target):
    recall_m = Recall(task='binary', average='micro',ignore_index=0).cuda() # exclude the background
    return recall_m(preds,target)

def precision(preds,target):
    precision_m = Precision(task='binary', average='micro',ignore_index=0).cuda() # exclude the background
    return precision_m(preds,target)

def specificity(preds,target):
    spe_m = Specificity(task='binary',average='micro').cuda()
    return spe_m(preds,target)

def connecivity(preds,target,thre_voxel):
    from skimage import measure
    component_pred, num_pred = measure.label(preds, connectivity=2, return_num=True)
    component_target, num_target = measure.label(target, connectivity=2, return_num=True)
    num_pred_above_thre = 0
    for idx in range(num_pred):
        flag = component_pred==idx
        flag = flag +0
        if flag.sum()<thre_voxel:
            continue
        else:
            num_pred_above_thre +=1

    num_target_above_thre = 0
    for idx in range(num_target):
        flag = component_target==idx
        flag = flag +0
        if flag.sum()<thre_voxel:
            continue
        else:
            num_target_above_thre +=1
    # con = 1 - min(1, abs(num_target-num_pred)/target.sum())
    con = num_pred_above_thre / num_target_above_thre
    return num_pred_above_thre,num_target_above_thre,con

def dice_surface(preds,target,spacing):
    preds = monai.transforms.AsDiscrete(to_onehot=2)(preds)
    target = monai.transforms.AsDiscrete(to_onehot=2)(target)
    dice_surface_m = monai.metrics.compute_surface_dice(preds[None], target[None], [0.5], include_background=False, distance_metric='euclidean', spacing=[spacing[0],spacing[1],spacing[2]])
    return dice_surface_m

def HD95(preds,target,spacing):
    preds = monai.transforms.AsDiscrete(to_onehot=2)(preds)
    target = monai.transforms.AsDiscrete(to_onehot=2)(target)
    hd95_m = monai.metrics.compute_hausdorff_distance(preds[None], target[None], include_background=False, distance_metric='euclidean', percentile=95, spacing=[spacing[0],spacing[1],spacing[2]])
    return hd95_m

def save_nifti(img, img_path):
    pair_img = nibabel.Nifti1Pair(img,np.eye(4))
    nibabel.save(pair_img,img_path)




def sub_volume_crop_LiVS(): # To generate 25D training set with size of 256*256*3
    root_path = r'Z:\liver_vessel_segmentation\data\LiVS'
    slices_25d = 3
    for root, dirs, files in os.walk(root_path): 
        if root.split('\\')[-1]!='Exp_interp_vessel_mask_nii':
            continue
        for name in files:
            vessel_tree = sitk.ReadImage(os.path.join(root,name))
            origin = vessel_tree.GetOrigin()
            space = vessel_tree.GetSpacing()
            direction = vessel_tree.GetDirection()

            ct_vol = sitk.ReadImage(os.path.join(root.replace('Exp_interp_vessel_mask_nii','Exp_image_nii'),name))
        
            vessel_tree = sitk.GetArrayFromImage(vessel_tree).transpose(1,2,0) 
            ct_vol = sitk.GetArrayFromImage(ct_vol).transpose(1,2,0) 

            label = vessel_tree.copy()
            label = np.nan_to_num(label, nan=0.0)

            edge = np.argwhere(label==1)
            depth_min = edge[:,2].min()
            depth_max = edge[:,2].max()                

            '# Crop orderly based on the vessel label'
            cropper = monai.transforms.RandCropByLabelClassesd(keys=['image','label'], label_key='label', image_key='image',
                                                                spatial_size=[256,256,slices_25d], ratios=[0,1],
                                                                num_classes=2, num_samples=1)
            CT_subVol = []
            vessel_subVol = []
            center_slice_idx_list = []
            for center_slice_idx in range(depth_min,depth_max):
                tmp_ct = ct_vol[:,:,center_slice_idx-1:center_slice_idx+2]
                tmp_vessel = label[:,:,center_slice_idx-1:center_slice_idx+2]
                if tmp_vessel.sum()==0:
                    continue
                if label[:,:,center_slice_idx-1].sum()==0 or label[:,:,center_slice_idx].sum()==0 or label[:,:,center_slice_idx+1].sum()==0:
                    continue

                ct_label_sub = cropper({'image':np.concatenate((tmp_ct[None,], tmp_vessel[None,]),axis=0),
                                        'label':tmp_vessel[None,]}) # channel first
                
                CT_subVol.append(ct_label_sub[0]['image'][0,].numpy())
                vessel_subVol.append(ct_label_sub[0]['image'][1,].numpy())
                center_slice_idx_list.append(center_slice_idx)

            cnt = 0
            for s1,s2,idx in zip(CT_subVol,vessel_subVol,center_slice_idx_list):
                save_path_1 = os.path.join(root_path, '25DSlice_CT_3slices', name.split('.')[0]+'_crop_'+str(idx)+'.nii.gz')
                save_path_2 = os.path.join(root_path, '25DSlice_vessel_3slices', name.split('.')[0]+'_crop_'+str(idx)+'.nii.gz')

                os.makedirs(os.path.join(root_path, '25DSlice_CT_3slices'), exist_ok=True)
                os.makedirs(os.path.join(root_path, '25DSlice_vessel_3slices'), exist_ok=True)

                img = sitk.GetImageFromArray(CT_subVol[cnt].transpose(2,0,1))
                img.SetOrigin(origin)
                img.SetDirection(direction)
                img.SetSpacing(space)
                sitk.WriteImage(img, '{}'.format(save_path_1))

                img = sitk.GetImageFromArray(vessel_subVol[cnt].transpose(2,0,1))
                img.SetOrigin(origin)
                img.SetDirection(direction)
                img.SetSpacing(space)
                sitk.WriteImage(img, '{}'.format(save_path_2))

                cnt +=1


def crop25d_testset_LiVS(): #   
    root_path = r'Z:\liver_vessel_segmentation\data\LiVS'
    level25d = 3# slices

    for root, dirs, files in os.walk(root_path): 
        if root.split('\\')[-1]!='Exp_vessel_mask_nii':
            continue
        for name in files:
            ct_vol = sitk.ReadImage(os.path.join(root.replace('Exp_vessel_mask_nii','Exp_image_nii'),name))
            origin = ct_vol.GetOrigin()
            space = ct_vol.GetSpacing()
            direction = ct_vol.GetDirection()
            ct_vol = sitk.GetArrayFromImage(ct_vol).transpose(1,2,0)

            vessel_mask = sitk.ReadImage(os.path.join(root,name))
            vessel_mask = sitk.GetArrayFromImage(vessel_mask).transpose(1,2,0)

            ct = []
            vessel = []
            for slice_idx in range(ct_vol.shape[-1]):
                block25d_ct = []
                block25d_vessel = []
                for idx in range(level25d):
                    if slice_idx-int(level25d/2)+idx<0 or slice_idx-int(level25d/2)+idx>=ct_vol.shape[-1]:
                        block25d_ct.append(np.zeros((256,256,1)))
                        block25d_vessel.append(np.zeros((256,256,1)))
                    else:
                        block25d_ct.append(ct_vol[..., slice_idx-int(level25d/2)+idx][...,None]) 
                        block25d_vessel.append(vessel_mask[..., slice_idx-int(level25d/2)+idx][...,None])
                block25d_ct = np.concatenate(block25d_ct, axis=-1)[None] # (1,256,256,7)
                block25d_vessel = np.concatenate(block25d_vessel, axis=-1)[None]

                ct.append(block25d_ct)  
                vessel.append(block25d_vessel)

            ct = np.concatenate(ct, axis=0).transpose(0,3,1,2) # (sliceNum,3,256,256)
            vessel = np.concatenate(vessel, axis=0).transpose(0,3,1,2) # (sliceNum,3,256,256)

            save_path_1 = os.path.join(root_path, 'Test25DCT_CTVolume_3slices', name.split('_')[0]+'.nii.gz')
            save_path_2 = os.path.join(root_path, 'Test25DCT_Vessel_3slices', name.split('_')[0]+'.nii.gz')

            os.makedirs(os.path.join(root_path, 'Test25DCT_CTVolume_3slices'), exist_ok=True)
            os.makedirs(os.path.join(root_path, 'Test25DCT_Vessel_3slices'), exist_ok=True)

            img = sitk.GetImageFromArray(ct)
            img.SetOrigin(origin)
            img.SetDirection(direction)
            img.SetSpacing(space)
            sitk.WriteImage(img, '{}'.format(save_path_1))

            img = sitk.GetImageFromArray(vessel)
            img.SetOrigin(origin)
            img.SetDirection(direction)
            img.SetSpacing(space)
            sitk.WriteImage(img, '{}'.format(save_path_2))


def prediction_transpose(root_path=None):

    for root, dirs, files in os.walk(root_path): 
        for name in files:
            try:
                pred_vol = sitk.ReadImage(os.path.join(root,name))
            except:
                os.remove(os.path.join(root,name))
                continue
            origin = pred_vol.GetOrigin()
            space = pred_vol.GetSpacing()
            direction = pred_vol.GetDirection()
            try:
                pred_vol = sitk.GetArrayFromImage(pred_vol).transpose(2,1,0)# # (256,256,D)
            except:
                os.remove(os.path.join(root,name))
                continue
        
            save_path = os.path.join(root,name)
            img = sitk.GetImageFromArray(pred_vol.transpose(2,0,1))
            img.SetOrigin(origin)
            img.SetDirection(direction)
            img.SetSpacing(space)
            sitk.WriteImage(img, '{}'.format(save_path))


def connectivity_graph_3D(hepatic_mask_path=None, portal_mask_path=None, save_path=None, save_path_png=None, sw=8, sh=8, sd=1, tth=1.2, dataset='LiVS', graph_2Dcheck=False):
    matplotlib.use('Agg')
    from PIL import Image

    def find_middle_point_3D(x1, y1, z1, x2, y2, z2):
        middle_x = round((x1 + x2) / 2)
        middle_y = round((y1 + y2) / 2)
        middle_z = round((z1 + z2) / 2)
        return middle_x, middle_y, middle_z

    def generate_nodes_from_mask(mask, sd, sh, sw):
        mask = mask.transpose(2,0,1)
        D, H, W = mask.shape
        GT = np.zeros((int(W/sw),int(H/sh),int(D/sd))) 
        nodes = []
        nodes_pos = []
        for z in range(0, D, sd):
            for y in range(0, H, sh):
                for x in range(0, W, sw):
                    sub_mask = mask[z:z+sd, y:y+sh, x:x+sw]
                    if np.any(sub_mask):  # Check if sub-mask contains vessel pixels
                        # Calculate the average position of vessel pixels
                        indices = np.where(sub_mask)
                        avg_x = np.mean(indices[2]) + x
                        avg_y = np.mean(indices[1]) + y
                        avg_z = np.mean(indices[0]) + z
                        nodes.append((avg_x, avg_y, avg_z))
                        GT[int(x/sw),int(y/sh),int(z/sd)] = 1
                        nodes_pos.append((avg_x, avg_y, avg_z))
                    else:
                        # Choose the center pixel as a node
                        center_x = x + (sw // 2)
                        center_y = y + (sh // 2)
                        center_z = z + (sd // 2)
                        nodes.append((center_x, center_y, center_z))
        GT = np.array(GT)
        return nodes,GT, nodes_pos
    
    # hepatic_mask_path = r'\\vf-lkeb\lkeb$\Scratch\Xiaotong\liver_vessel_segmentation\data\3DircadbNifti_newLiverMask\25DSlice_hepatic_7slices\hepatic01_Rot_0_crop_26.nii.gz'
    # portal_mask_path = r'\\vf-lkeb\lkeb$\Scratch\Xiaotong\liver_vessel_segmentation\data\3DircadbNifti_newLiverMask\25DSlice_portal_7slices\portal01_Rot_0_crop_26.nii.gz'
    hepatic_mask = nibabel.load(hepatic_mask_path).get_fdata()
    portal_mask = nibabel.load(portal_mask_path).get_fdata()
    mask = hepatic_mask + portal_mask
    mask[mask>0] = 1
    # ct_vol = nibabel.load(r'\\vf-lkeb\lkeb$\Scratch\Xiaotong\liver_vessel_segmentation\data\3DircadbNifti_newLiverMask\25DSlice_CT_7slices\PATIENT01_Rot_0_crop_26.nii.gz').get_fdata()
    # mask = nibabel.load(r'\\vf-lkeb\lkeb$\Scratch\Xiaotong\liver_vessel_segmentation\data\3DircadbNifti_newLiverMask\Resize2DCT_patientCTVolume\PATIENT01_DICOM.nii.gz').get_fdata()
    img_size = [mask.shape[0],mask.shape[1],mask.shape[2]]
    # sw = 8
    # sh = 8
    # sd = 1
    nw = int(img_size[0]/sw)
    nh = int(img_size[1]/sh)
    nd = int(img_size[2]/sd)
    # tth = 1.5  # Example threshold for demonstration
    nodes,GT, nodes_pos = generate_nodes_from_mask(mask, sd, sh, sw)
    num_nodes = len(nodes)
    # travel_time = np.zeros((num_nodes, num_nodes))

    edges = []
    for i in range(num_nodes):
        phi = np.ones((nw, nh, nd))
        phi_z_idx = int((i) / (nw*nh))
        phi_y_idx = int((i-phi_z_idx*nw*nh) / nw)
        phi_x_idx = int((i-phi_z_idx*nw*nh) % nw)
        phi[phi_x_idx,phi_y_idx,phi_z_idx] = 0
        if GT[phi_x_idx,phi_y_idx,phi_z_idx]==0:
            continue
        speed = GT.copy()

        t = skfmm.travel_time(phi, speed, dx=1).data

        for j in range(i + 1, num_nodes):
            t_z_idx = int((j) / (nw*nh))
            t_y_idx = int((j-t_z_idx*nw*nh) / nw)
            t_x_idx = int((j-t_z_idx*nw*nh) % nw)
            t_ij = t[t_x_idx,t_y_idx,t_z_idx]
            # t_ij = t[int(nodes[j][0]),int(nodes[j][1]),int(nodes[j][2])]

            if t_ij < tth and t_ij > 0:
                edges.append([i,j])

    '# prune edges based on the ground truth'
    pruned_edges = []
    for edge in edges:
        # line_coords = find_points_between_3d(int(nodes[edge[0]][0]), int(nodes[edge[0]][1]), int(nodes[edge[0]][2]), int(nodes[edge[1]][0]), int(nodes[edge[1]][1]), int(nodes[edge[1]][2]))
        # line_val = mask[line_coords[:,1], line_coords[:,0], line_coords[:,2]]        
        # if np.all(line_val):
        # # if line_val.mean() >=0.9:
        #     pruned_edges.append(edge)
        middle_point_coords = find_middle_point_3D(float(nodes[edge[0]][0]), float(nodes[edge[0]][1]), float(nodes[edge[0]][2]), float(nodes[edge[1]][0]), float(nodes[edge[1]][1]), float(nodes[edge[1]][2]))
        middle_point_val = mask[middle_point_coords[1], middle_point_coords[0], middle_point_coords[2]]
        if middle_point_val ==1:
            pruned_edges.append(edge)
    edges = pruned_edges

    '# save nodes and edges'
    data = {'nodes':nodes, 'edges':edges, 'speed':speed}
    if dataset=='IRCADB':
        np.save(os.path.join(save_path, hepatic_mask_path.split('\\')[-1].replace('hepatic','graph').replace('.nii.gz','.npy')), data)
    elif dataset=='LiVS':
        np.save(os.path.join(save_path, hepatic_mask_path.split('\\')[-1].replace('case','graph').replace('.nii.gz','.npy')), data)

    nodes_pos = np.array(nodes_pos)
    nodes = np.array(nodes)
    edges = np.array(edges)
    '# Create a 3D plot'
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_facecolor("none") 
    # # Plot the points
    # ax.scatter(nodes_pos[:, 0], nodes_pos[:, 1], nodes_pos[:, 2], color='red', s=10, alpha=1.0)
    # # Plot the edges
    # for edge in edges:
    #     ax.plot(nodes[edge, 0], nodes[edge, 1], nodes[edge, 2], color='blue', alpha=1.0)
    # # Set plot properties
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # # ax.set_title('Edges between 3D Points')
    'Create a gif for the 3D graph connectivity'
    # # Set the specific view using view_init
    # for azimuth_angle in range(0,360,1):
    #     # azimuth_angle = 60  # Azimuthal angle (rotation around the z-axis)
    #     elevation_angle = 45  # Elevation angle (rotation around the y-axis)
    #     ax.view_init(elev=elevation_angle, azim=azimuth_angle)
        
    #     fig.savefig(os.path.join(r'Z:\weekly_ppt\gif\3dGraphFrames', 'view_'+str(azimuth_angle).zfill(3)+'.png'), transparent=True, bbox_inches='tight', pad_inches=0.1)
    #     # Show the plot
    #     # plt.show()
    # frame_files = []
    # for root, dirs, files in os.walk(r'Z:\weekly_ppt\gif\3dGraphFrames'): 
    #     for name in files:
    #         frame_files.append(os.path.join(root,name))
    # # Create a list of Image objects
    # frames = [Image.open(frame) for frame in frame_files[1:]]
    # # Save as a GIF
    # frames[0].save(r"Z:\weekly_ppt\gif\3DGraph.gif", save_all=True, append_images=frames[1:], duration=50, loop=0)

    '# Create 2D plot for each slice in the 25D data block'
    if graph_2Dcheck:
        for slice_idx in range(mask.shape[-1]):
            slice_nodes_idx = np.argwhere(nodes_pos[:,-1]==slice_idx)
            slice_nodes = np.squeeze(nodes_pos[slice_nodes_idx], axis=1)
            slice_nodes_list = [list(slice_nodes[i]) for i in range(slice_nodes.shape[0])]
            slice_edges = []
            for edge in edges:
                if list(nodes[edge[0]]) in slice_nodes_list and list(nodes[edge[1]]) in slice_nodes_list:
                    slice_edges.append(edge)

            # Create a 2D plot
            fig = plt.figure()
            ax = fig.add_subplot(121)
            plt.imshow(mask[:,:,slice_idx], cmap='gray')
            # Plot the points
            ax.scatter(slice_nodes[:, 0], slice_nodes[:, 1], color='red', s=10)
            # Plot the edges
            slice_edges = np.array(slice_edges)
            for edge in slice_edges:
                ax.plot(nodes[edge, 0], nodes[edge, 1], color='blue')
            # Plot the sub-areas using dotted lines
            for y in range(0, mask.shape[0], sh):
                plt.plot([0, mask.shape[1]], [y, y], 'r--', alpha=0.3)
            for x in range(0, mask.shape[1], sw):
                plt.plot([x, x], [0, mask.shape[0]], 'r--', alpha=0.3)
            # Set plot properties
            ax.set_xlim((0,mask.shape[0]))
            ax.set_ylim((0,mask.shape[1]))
            ax.set_xlabel('X', fontsize=22)
            ax.set_ylabel('Y', fontsize=22)
            if dataset=='IRCADB':
                ax.set_title('Edges between Points in 2D cross-sectional view \n'+hepatic_mask_path.split('\\')[-1].replace('hepatic','graph')+'(Slice '+str(slice_idx)+')', fontsize=24)
            elif dataset=='LiVS':
                ax.set_title('Edges between Points in 2D cross-sectional view \n'+hepatic_mask_path.split('\\')[-1].replace('case','graph')+'(Slice '+str(slice_idx)+')', fontsize=24)
            ax.tick_params(axis='both', which='major', labelsize=20)

            ax2 = fig.add_subplot(122)
            plt.imshow(speed[:,:,slice_idx].T,'gray')
            ax2.set_xlim((0,int(mask.shape[0]/sw)-1))
            ax2.set_ylim((0,int(mask.shape[1]/sh)-1))
            ax2.set_xlabel('X', fontsize=22)
            ax2.set_ylabel('Y', fontsize=22)
            ax2.set_title('Speed function in 2D cross-sectional view', fontsize=24)
            ax2.tick_params(axis='both', which='major', labelsize=20)

            # mng = plt.get_current_fig_manager()
            # mng.window.showMaximized() 
            # Get the screen size
            root = tk.Tk()
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            root.destroy()
            # Set the figure size to match the screen size
            fig.set_size_inches(screen_width/100, screen_height/100) 
            if dataset=='IRCADB':
                fig.savefig(os.path.join(save_path_png, hepatic_mask_path.split('\\')[-1].replace('hepatic','graph').replace('.nii.gz','_Slice_'+str(slice_idx)+'.png')),bbox_inches='tight', dpi=300)
            elif dataset=='LiVS':
                fig.savefig(os.path.join(save_path_png, hepatic_mask_path.split('\\')[-1].replace('case','graph').replace('.nii.gz','_Slice_'+str(slice_idx)+'.png')),bbox_inches='tight', dpi=300)
            # Show the plot
            # plt.show()


def save_graph3D_25DTrainSet_LiVS():
    sw = 16 #8
    sh = 16 #8
    sd = 1
    hepatic_root = r'Z:\liver_vessel_segmentation\data\LiVS\25DSlice_vessel_3slices'
    portal_root = r'Z:\liver_vessel_segmentation\data\LiVS\25DSlice_vessel_3slices'
    save_root_png = fr'Z:\liver_vessel_segmentation\data\LiVS\25DSlice_graph_3slices_2DPNGCheck_gridXYZ_{sw}_{sh}_{sd}'
    save_root = fr'Z:\liver_vessel_segmentation\data\LiVS\25DSlice_graph_3slices_gridXYZ_{sw}_{sh}_{sd}'

    if not os.path.exists(save_root_png):
        os.makedirs(save_root_png)
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    for root, dirs, files in os.walk(hepatic_root): 
        for name in files:
            hepatic_mask_path = os.path.join(root,name)
            portal_mask_path = os.path.join(root,name)
            if os.path.exists(os.path.join(save_root, hepatic_mask_path.split('\\')[-1].replace('case','graph').replace('.nii.gz','.npy'))):
                continue
            connectivity_graph_3D(hepatic_mask_path=hepatic_mask_path, portal_mask_path=portal_mask_path, save_path=save_root, save_path_png=save_root_png, sw=sw, sh=sh, sd=sd, tth=1.2, dataset='LiVS')


def ensemble_inference_LiVS():
    def mv(a):
        # res = Image.fromarray(np.uint8(img_list[0] / 2 + img_list[1] / 2 ))
        # res.show()
        b = a.size(0)
        return torch.sum(a, 0, keepdim=True) / b
    def staple(a):
        # a: n,c,h,w detach tensor
        mvres = mv(a)
        gap = 0.4
        if gap > 0.02:
            for i, s in enumerate(a):
                r = s * mvres
                res = r if i == 0 else torch.cat((res,r),0)
            nres = mv(res)
            gap = torch.mean(torch.abs(mvres - nres))
            mvres = nres
            a = res
        return mvres

    root_path = r'Z:\liver_vessel_segmentation\model\AblationStudy_DynamicConditioning\outputs\sample_validation\LiVS_DynamicCond_seed10_20_30_40_50\fold_2\validation_iter100000'
    for root, dirs, files in os.walk(root_path): 
        if root.split('\\')[-1]!='ensemble_0':
            continue
        for name in files:
            if 'pred_sample' not in name:
                continue
            segmented_volume_0 = sitk.ReadImage(os.path.join(root, name))
            origin = segmented_volume_0.GetOrigin()
            space = segmented_volume_0.GetSpacing()
            direction = segmented_volume_0.GetDirection()
            segmented_volume_0 = sitk.GetArrayFromImage(segmented_volume_0)

            segmented_volume_1 = sitk.ReadImage(os.path.join(root.replace('ensemble_0','ensemble_1'), name))
            segmented_volume_1 = sitk.GetArrayFromImage(segmented_volume_1)

            segmented_volume_2 = sitk.ReadImage(os.path.join(root.replace('ensemble_0','ensemble_2'), name))
            segmented_volume_2 = sitk.GetArrayFromImage(segmented_volume_2)

            segmented_volume_3 = sitk.ReadImage(os.path.join(root.replace('ensemble_0','ensemble_3'), name))
            segmented_volume_3 = sitk.GetArrayFromImage(segmented_volume_3)

            segmented_volume_4 = sitk.ReadImage(os.path.join(root.replace('ensemble_0','ensemble_4'), name))
            segmented_volume_4 = sitk.GetArrayFromImage(segmented_volume_4)

            ensemble_sample_list = [torch.from_numpy(segmented_volume_0), 
                                    torch.from_numpy(segmented_volume_1), 
                                    torch.from_numpy(segmented_volume_2),
                                    torch.from_numpy(segmented_volume_3),
                                    torch.from_numpy(segmented_volume_4),
                                    ]

            ensemble_sample = staple(torch.stack(ensemble_sample_list,dim=0))
            ensemble_sample = torch.clamp(ensemble_sample, 0, 1).squeeze(0).detach().cpu().numpy()
            ensemble_sample[ensemble_sample>=0.5] = 1
            ensemble_sample[ensemble_sample<0.5] = 0


            img = sitk.GetImageFromArray(ensemble_sample)
            img.SetOrigin(origin)
            img.SetDirection(direction)
            img.SetSpacing(space)
            os.makedirs(os.path.join(root_path, 'ensembled_pred_staple'), exist_ok=True)
            save_path = os.path.join(root_path, 'ensembled_pred_staple', name.replace('pred_sample','ensembled_pred_sample'))
            sitk.WriteImage(img, '{}'.format(save_path))


if __name__ == "__main__":

    ## Sample 2.5D CT blocks from CT volumes for training
    sub_volume_crop_LiVS()

    ## Sample 2.5D CT blocks from CT volumes for testing
    crop25d_testset_LiVS()

    ## Create graph of each 2.5D CT blocks for training
    save_graph3D_25DTrainSet_LiVS()

    ## Transpose segmentation masks
    prediction_transpose()

    ## Sample ensembling for segmentation using five different seeds.
    ensemble_inference_LiVS()

