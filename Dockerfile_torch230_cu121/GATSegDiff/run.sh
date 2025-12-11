#!/bin/bash
OUTPUT_DIR="/home/xzhang/inference_project/GATSegDiff/output/Pre-ablation/Portal"
data_test_CT_25D="/home/xzhang/inference_project/GATSegDiff/data/CT_25D"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$data_test_CT_25D"
docker run --rm --gpus "device=0" \
           --tmpfs /dev/shm:rw,noexec,nosuid,size=2g \
    --mount type=bind,src="$OUTPUT_DIR",dst=/output \
    --mount type=bind,src=/mnt/e/xiaotong/WSL/TestData/LiverVesselSeg/Pre-ablation/Portal,dst=/data_test_CT,readonly \
    --mount type=bind,src="$data_test_CT_25D",dst=/data_test_CT_25D \
    -e MODEL_PATH=/app/model/LiVS_reannotated_subset_F0_model160000.pt \
    gat_seg_diff:cuda121 \
    --out_dir /output \
    --data_test_CT /data_test_CT \
    --test_root_path /data_test_CT_25D \
    --batch_size_sample 8 \
    --seed 10 \
    --noise_cancel True \
    --dpm_solver True \
    --diffusion_steps 30 \
    --num_ensemble 3

## Interactive
# OUTPUT_DIR="/home/xzhang/inference_project/GATSegDiff/output/Pre-ablation/Portal"
# data_test_CT_25D="/home/xzhang/inference_project/GATSegDiff/data/CT_25D"
# mkdir -p "$OUTPUT_DIR"
# mkdir -p "$data_test_CT_25D"
# docker run -it --rm --gpus "device=0" --tmpfs /dev/shm:rw,noexec,nosuid,size=2g \
#     --mount type=bind,src="$OUTPUT_DIR",dst=/output \
#     --mount type=bind,src=/mnt/e/xiaotong/WSL/TestData/LiverVesselSeg/Pre-ablation/Portal,dst=/data_test_CT,readonly \
#     --mount type=bind,src="$data_test_CT_25D",dst=/data_test_CT_25D \
#     -e MODEL_PATH=/app/model/LiVS_reannotated_subset_F0_model160000.pt \
#     gat_seg_diff:cuda121 \
#     bash

##### All mounted paths must be exist!!!

    # --dpm_solver True \
    # --diffusion_steps 30 \
    # --num_ensemble 3 \