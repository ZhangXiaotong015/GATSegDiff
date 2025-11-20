# GATSegDiff
The official implementation of 'Continuous and complete liver vessel segmentation with graph-attention guided diffusion'.

## Data preparation
```
from data_processing import sub_volume_crop_LiVS, save_graph3D_25DTrainSet_LiVS

# Crop and resize 2D CT images to (256,256)
sub_volume_crop_LiVS()

# Create a local graph for each 2.5D CT block with three consecutive CT slices
save_graph3D_25DTrainSet_LiVS()
```
The LiVS dataset can be accessed at: ```https://ieee-dataport.org/documents/liver-vessel-segmentation```.

The statistic for the number of annotated slices in LiVS dataset can be found in ```data processing/LiVS_overview.xlsx```.
In our study, we only use cases with at least **30** annotated slices.

The liver vessel masks of the LiVS dataset **interpolated** using ITKSNAP are provided in ```data processing/Exp_interp_vessel_mask_nii```.
The interpolated masks are required when creating the local graph with contextual information.

**NOTICE!** The interpolated liver vessel masks in the LiVS dataset are used **only** for local graph creation.
Only the 2.5D CT blocks whose central slices contain **real annotation** are included in the training.

## Training
![[Training phase]([training phase.png](https://github.com/ZhangXiaotong015/GATSegDiff/blob/main/training%20phase.png))](https://github.com/ZhangXiaotong015/GATSegDiff/blob/main/training%20phase.png)
**Submit a training task to Slurm:**

```
sbatch run_Foldx_LiVS.sh
```

## Inference
![[Inference phase]([inference phase.png](https://github.com/ZhangXiaotong015/GATSegDiff/blob/main/inference%20phase.png))](https://github.com/ZhangXiaotong015/GATSegDiff/blob/main/inference%20phase.png)
**Submit inference tasks to Slurm:**

```
bash run_Infer_Foldx_LiVS.sh
```
**--inFold_subset**: Splits the test cases evenly into num_test_cases / inFold_subset parts.
Each part is processed on an individual GPU.

**--seed**: Seed used for inference. 
(For the LiVS dataset with discontinuous annotations, we recommend ensembling inferences with five different seeds, as described in our paper.)

## Post-processing
