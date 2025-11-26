# GATSegDiff
The official implementation of [Continuous and complete liver vessel segmentation with graph-attention guided diffusion](https://www.sciencedirect.com/science/article/pii/S0950705125017253) published in *Knowledge-Based Systems*.

## Data preparation
```
from data_processing import sub_volume_crop_LiVS, crop25d_testset_LiVS, save_graph3D_25DTrainSet_LiVS

# Sample 2.5D CT blocks from CT volumes for training
sub_volume_crop_LiVS()

# Sample 2.5D CT blocks from CT volumes for testing
crop25d_testset_LiVS()

# Create a local graph for each 2.5D CT block with three consecutive CT slices
save_graph3D_25DTrainSet_LiVS()
```
The LiVS dataset can be accessed at: [LiVS](https://ieee-dataport.org/documents/liver-vessel-segmentation)
.

The statistic for the number of annotated slices in LiVS dataset can be found in [LiVS_overview.xlsx](data%20processing/LiVS_overview.xlsx).
In our study, we only use cases with at least **30** annotated slices.

The liver vessel masks of the LiVS dataset **interpolated** using ITKSNAP are provided in [Exp_interp_vessel_mask_nii](data%20processing/Exp_interp_vessel_mask_nii).
The interpolated masks are required when creating the local graph with contextual information.

**NOTICE!** The interpolated liver vessel masks in the LiVS dataset are used **only** for local graph creation.
Only the 2.5D CT blocks whose central slices contain **real annotation** are included in the training.

**To contribute to future development in the field of liver vessel segmentation, we also reannotated 30 cases in the LiVS dataset to produce continuous and complete vessel trees.
The reannotated masks can be found in [Exp_vessel_mask_reannotate_nii](data%20processing/Exp_vessel_mask_reannotate_nii).*

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
**--inFold_subset**: Splits the test cases evenly into ```num_test_cases / inFold_subset``` parts.
Each part is processed on an individual GPU.

**--seed**: Seed used for inference. 
(For the LiVS dataset with discontinuous annotations, we recommend ensembling inferences with five different seeds, as described in our paper.)

**Optional inference acceleration using DPMSolver++:**

DPM-Solver++ can be enabled by setting the following three parameters in ```run_Infer_Foldx_LiVS.sh```.
```
--dpm_solver True \
--num_ensemble 3 \
--diffusion_steps 30
```

## Post-processing
```
from data_processing import prediction_transpose, ensemble_inference_LiVS

# Transpose segmentation masks
prediction_transpose()

# Sample ensembling for segmentation using five different seeds.
ensemble_inference_LiVS()
```

**Optional noise cancelling:**
```
bash noiseCancel_Foldx_LiVS.sh
```

## Dockerfile
You can simply build the inference image in a WSL2 environment using the Dockerfile in [Dockerfile/GATSegDiff](Dockerfile/GATSegDiff/)
.
```
cd Dockerfile/GATSegDiff
docker build -t image_name:tag .
bash run.sh
```
For quick experimentation, we have released the model weights trained on the [reannotated subset](data%20processing/Exp_vessel_mask_reannotate_nii/) of the LiVS dataset.

You can find the model weights at [this link](https://drive.google.com/drive/folders/1V9NtZingw9XQmFGbGGFOE59vtF6iDeck?usp=drive_link) and download them to ```Dockerfile/GATSegDiff/model```.

## Citation
If you use this work, please cite:
```
@article{ZHANG2025114686,
title = {Continuous and complete liver vessel segmentation with graph-attention guided diffusion},
journal = {Knowledge-Based Systems},
volume = {331},
pages = {114686},
year = {2025},
issn = {0950-7051},
doi = {https://doi.org/10.1016/j.knosys.2025.114686},
url = {https://www.sciencedirect.com/science/article/pii/S0950705125017253},
author = {Xiaotong Zhang and Alexander Broersen and Gonnie {Van Erp} and Silvia Pintea and Jouke Dijkstra},
```
