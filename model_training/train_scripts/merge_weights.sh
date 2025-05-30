#!/bin/bash
#SBATCH -A EUHPC_E04_042       # account name
#SBATCH -p boost_usr_prod
#SBATCH --time 24:00:00
#SBATCH --job-name=merge_weights
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=.slurm/merge_weights.out
#SBATCH --error=.slurm/merge_weights.err
#SBATCH --mem=168GB

#source ~/env_HF/bin/activate
source ~/venv/latxa-instruct/bin/activate

export WANDB_MODE=offline
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8
export TOKENIZERS_PARALLELISM=true
export TRANSFORMERS_NO_ADVISORY_WARNINGS=true
export OMP_NUM_THREADS=16
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

export TMPDIR=~/work_dir/.cache_iker


BASE_PATH="~/work_dir/LatxaTxat_models/InstructionModels/"

MODELS=(
    "exp_0_010-fixed"
    "exp_0_011-fixed"
    "exp_0_101-fixed"
    "exp_0_110-fixed"
    "exp_0_111-fixed"
    "exp_1_001-fixed"
    "exp_1_010-fixed"
    "exp_1_011-fixed"
    "exp_1_101-fixed"
    "exp_1_110-fixed"
    "exp_1_111-fixed"
    "exp_2_010-fixed"
    "exp_2_011-fixed"
    "exp_2_100-fixed"
    "exp_2_101-fixed"
    "exp_2_110-fixed"
    "exp_2_111-fixed"
)



echo "Starting weight merging process..."

for MODEL in "${MODELS[@]}"; do
    echo "Processing model: ${MODEL}"
    
    # Find the latest checkpoint directory
    LATEST_CHECKPOINT=$(ls -d ${BASE_PATH}/${MODEL}/checkpoint-* | sort -V | tail -n 1)
    echo "Latest checkpoint found: ${LATEST_CHECKPOINT}"
    
    # Create merged model directory
    MERGED_DIR="${BASE_PATH}/${MODEL}/merged_model"
    echo "Creating merged model directory: ${MERGED_DIR}"
    mkdir -p ${MERGED_DIR}

    # Merge weights
    echo "Merging weights from ${LATEST_CHECKPOINT}/pytorch_model_fsdp_0 to ${MERGED_DIR}"
    accelerate merge-weights ${LATEST_CHECKPOINT}/pytorch_model_fsdp_0 ${MERGED_DIR}

    # Copy trainer state as training results
    cp ${LATEST_CHECKPOINT}/trainer_state.json ${MERGED_DIR}/training_results.json

    # Copy configuration files
    echo "Copying configuration files..."
    for file in \
    config.json \
    generation_config.json \
    tokenizer_config.json \
    special_tokens_map.json \
    tokenizer.json 
    do
        echo "Copying ${file} from ${BASE_PATH}/${MODEL} to ${MERGED_DIR}"
        cp ${BASE_PATH}/${MODEL}/$file ${MERGED_DIR}
    done
    
    echo "Finished processing ${MODEL}"
    echo "----------------------------------------"
done

echo "All models processed successfully!"