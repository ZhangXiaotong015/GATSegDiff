# Create an array to store background process IDs
declare -a JOB_IDS
# Submit five jobs in a loop
inFold_subset=101
for ((i=0; i<inFold_subset; i++)); do
    JOB_NAME="ncF2_${i}"
    sbatch --job-name=$JOB_NAME \
           --output="/data1/xzhang2/liver_vessel_segmentation/model/GATSegDiff/slurm_logs/${JOB_NAME}.out" \
           --error="/data1/xzhang2/liver_vessel_segmentation/model/GATSegDiff/slurm_logs/${JOB_NAME}.err" \
           --ntasks=1 \
           --cpus-per-task=1 \
           --partition=gpu-medium,gpu-long,cpu-short,cpu-medium,cpu-long \
           --mem=32GB \
           --time=0-04:00:00 <<EOF &  # Corrected part here
#!/bin/bash
python "/home/xzhang2/data1/liver_vessel_segmentation/model/GATSegDiff/noise_cancel.py" \
    --infer_name 'LiVS_3Fold_FullVolume_FC_5TimesEnsemble_seed_10_20_30_40_50' \
    --fold 3 \
    --fold_idx '2' \
    --inFold_subset $inFold_subset \
    --inFold_sub_idx "$i" \
    --iters 160000
EOF

    # Get the job ID of the last submitted job
    JOB_IDS+=($!)
done

# Wait for all background jobs to finish
wait "${JOB_IDS[@]}"