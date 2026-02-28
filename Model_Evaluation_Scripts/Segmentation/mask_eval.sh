#!/bin/bash
#SBATCH --job-name=gaash
#SBATCH --partition=highq
##SBATCH --gres=gpu:h100_1g.12gb:1
##SBATCH --gres=gpu:h100_3g.47gb:1
#SBATCH --gres=gpu:h100:1
#SBATCH --qos=highq
##SBATCH --gres=gpu:nvidia_h100_nvl_3g.47gb:1
## SBATCH --time=00:10:00
#SBATCH --output=evaluation_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G

set -e  # stop on error

echo "========== JOB START =========="
date

source ~/.bashrc
conda init
conda activate visionreasoner_backup

INPUT_JSON="/home/gaash/Surgical/Outputs/GPT_5_2/qwen3_predictions.jsonl"
CLEAN_JSON="/home/gaash/Surgical/Outputs/GPT_5_2/qwen3_predictions_cleaned.jsonl"

IMAGE_DIR="/home/scratch-scholars/Tawheed/Combined_1016/"
DATASET_JSON="/home/scratch-scholars/Tawheed/Combined_1016/output.json"
OUTPUT_DIR="/home/gaash/Surgical/Outputs/GPT_5_2"

echo "Checking input files..."

# ---------- FILE VALIDATIONS ----------

if [ ! -f "$INPUT_JSON" ]; then
    echo "ERROR: Predictions file not found:"
    echo "$INPUT_JSON"
    exit 1
fi

if [ ! -s "$INPUT_JSON" ]; then
    echo "ERROR: Predictions file is empty"
    exit 1
fi

if [ ! -f "$DATASET_JSON" ]; then
    echo "ERROR: Dataset JSON not found"
    exit 1
fi

if [ ! -d "$IMAGE_DIR" ]; then
    echo "ERROR: Image directory not found"
    exit 1
fi

echo "Input validation passed."

# ---------- CLEAN OUTPUT CHECK ----------

echo "Running repair_predictions.py..."
python /home/gaash/Surgical/repair_predictions.py \
    --input "$INPUT_JSON" \
    --output "$CLEAN_JSON"

echo "Repair step completed."

# ---------- VALIDATE CLEAN FILE ----------

if [ ! -s "$CLEAN_JSON" ]; then
    echo "ERROR: Cleaned file missing or empty!"
    exit 1
fi

echo "Clean file ready."

# ---------- RUN SAM3 ----------

echo "Running SAM3 mask generation..."

python /home/gaash/Surgical/sam2_mask.py \
  --qwen_results "$CLEAN_JSON" \
  --image_dir "$IMAGE_DIR" \
  --dataset_json "$DATASET_JSON" \
  --output_dir "$OUTPUT_DIR"

echo "SAM3 completed."


echo "========== JOB FINISHED =========="
date