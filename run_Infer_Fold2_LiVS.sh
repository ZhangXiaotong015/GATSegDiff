# Create an array to store background process IDs
declare -a JOB_IDS
batch_size_sample=50
# Submit five jobs in a loop
inFold_subset=101
seed=50
for ((i=0; i<inFold_subset; i++)); do
    JOB_NAME="s2sub${i}"
    sbatch --job-name=$JOB_NAME \
           --output="/data1/xzhang2/liver_vessel_segmentation/model/GATSegDiff/slurm_logs/seed${seed}_${JOB_NAME}.out" \
           --error="/data1/xzhang2/liver_vessel_segmentation/model/GATSegDiff/slurm_logs/seed${seed}_${JOB_NAME}.err" \
           --ntasks=1 \
           --cpus-per-task=1 \
           --gres=gpu:1 \
           --partition=gpu-long,gpu-medium \
           --nodelist=node863,node864,node865,node866,node867,node868,node869,node870,node871,node872,node873,node874,node875,node876 \
           --mem=32GB \
           --time=1-00:00:00 <<EOF &  # Corrected part here
#!/bin/bash
echo "batch_size_sample in job script: $batch_size_sample"
python "/data1/xzhang2/liver_vessel_segmentation/model/GATSegDiff/Experiment_sample_LiVS.py" \
    --name 'LiVS_3Fold_FullVolume_FC_ensemble5th_seed${seed}' \
    --model_name 'LiVS_3Fold' \
    --batch_size_sample $batch_size_sample \
    --seed ${seed} \
    --fold 3 \
    --fold_idx '2' \
    --inFold_subset $inFold_subset \
    --inFold_sub_idx "$i" \
    --log_interval 20000 \
    --start_log_step 160000 \
    --end_log_step 160000 

EOF

    # Get the job ID of the last submitted job
    JOB_IDS+=($!)
done

# Wait for all background jobs to finish
wait "${JOB_IDS[@]}"

# --dpm_solver True \
# --num_ensemble 3 \
# --diffusion_steps 30 