#!/bin/bash
#SBATCH --job-name=LiVSF0
#SBATCH --output=/home/xzhang2/data1/liver_vessel_segmentation/model/GATSegDiff/slurm_logs/slurm-%j.out
#SBATCH --error=/home/xzhang2/data1/liver_vessel_segmentation/model/GATSegDiff/slurm_logs/slurm-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu-long
#SBATCH --nodelist=node863,node864,node865,node866,node867,node868,node869,node870,node871,node872,node873,node874,node875,node876 \
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=07-00:00:00


hostname
python "/home/xzhang2/data1/liver_vessel_segmentation/model/GATSegDiff/Experiment_train_LiVS.py" \
            --name 'LiVS_3Fold' \
            --batch_size 10 \
            --fold_idx '0' 
            