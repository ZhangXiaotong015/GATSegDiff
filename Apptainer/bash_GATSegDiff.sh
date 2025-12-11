echo "Using GPUs:"
nvidia-smi || echo "No GPU visible"

# # --- Ensure module system is available ---
# source /etc/profile.d/modules.sh

# # --- Load Apptainer module ---
# module load container/apptainer/1.4.1

# --- Base directory for all Apptainer data ---
BASE=/data1/xzhang2/docker_archive
# mkdir -p $BASE/slurm_logs

# --- 1. Set Apptainer tmp & cache to your large persistent directory ---
export APPTAINER_TMPDIR=$BASE/apptainer_tmp
export APPTAINER_CACHEDIR=$BASE/apptainer_cache

mkdir -p "$APPTAINER_TMPDIR"
mkdir -p "$APPTAINER_CACHEDIR"
# chmod 700 "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

# --- 2. Where to store your .sif ---
export IMAGEDIR=$BASE/apptainer_images
mkdir -p "$IMAGEDIR"
# chmod 700 "$IMAGEDIR"

# --- 3. Path to your Docker archive (.tar) ---
TARFILE=$BASE/gat_seg_diff_cuda121.tar

# --- 4. Build SIF only if not already present ---
if [ ! -f "$IMAGEDIR/gat_seg_diff_cuda121.sif" ]; then
    echo "Building SIF from $TARFILE ..."
    apptainer build "$IMAGEDIR/gat_seg_diff_cuda121.sif" docker-archive://$TARFILE
fi

# --- 5. Prepare output and data directories ---
echo "Running container..."

OUT_BASE=$BASE/VesselSeg_GATSegDiff

OUTPUT_DIR="$OUT_BASE/output/Pre-ablation/Portal"
data_test_CT_25D="$OUT_BASE/data/CT_25D"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$data_test_CT_25D"

DATA_CT=$BASE/data/LiverVesselSeg/Pre-ablation/Portal

# SHM=/exports/lkeb-hpc/xzhang/shm
# mkdir -p $SHM

# --- 6. Run GATSegDiff container on GPU ---
apptainer run --nv \
    --bind "$OUTPUT_DIR:/output" \
    --bind "$DATA_CT:/data_test_CT:ro" \
    --bind "$data_test_CT_25D:/data_test_CT_25D" \
    --env MODEL_PATH=/app/model/LiVS_reannotated_subset_F0_model160000.pt \
    "$IMAGEDIR/gat_seg_diff_cuda121.sif" \
        --out_dir /output \
        --data_test_CT /data_test_CT \
        --test_root_path /data_test_CT_25D \
        --batch_size_sample 8 \
        --seed 10 \
        --noise_cancel True \
        --dpm_solver True \
        --diffusion_steps 30 \
        --num_ensemble 3

