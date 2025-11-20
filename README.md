# GATSegDiff
The official implementation of 'Continuous and complete liver vessel segmentation with graph-attention guided diffusion'.

## Training
![[Training phase]([training phase.png](https://github.com/ZhangXiaotong015/GATSegDiff/blob/main/training%20phase.png))](https://github.com/ZhangXiaotong015/GATSegDiff/blob/main/training%20phase.png)
Submit a training task to Slurm:

```
sbatch run_Foldx_LiVS.sh
```

## Inference
![[Inference phase]([inference phase.png](https://github.com/ZhangXiaotong015/GATSegDiff/blob/main/inference%20phase.png))](https://github.com/ZhangXiaotong015/GATSegDiff/blob/main/inference%20phase.png)
Submit inference tasks to Slurm:

```
bash run_Infer_Foldx_LiVS.sh
```
--inFold_subset: Splits the test cases evenly into num_test_cases / inFold_subset parts.
Each part is processed on an individual GPU.
