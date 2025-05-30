#!/bin/bash
#SBATCH -A EUHPC_E04_042       # account name
#SBATCH -p boost_usr_prod
#SBATCH --time 24:00:00
#SBATCH --job-name=exp_0_111_fixed
#SBATCH --cpus-per-task=16
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --output=.slurm/exp_0_111_fixed.out
#SBATCH --error=.slurm/exp_0_111_fixed.err
#SBATCH --mem=168GB
#SBATCH --exclude=lrdn0310,lrdn0323,lrdn0332,lrdn0340,lrdn0317

#source ~/env_HF/bin/activate
source ~/venv/latxa-instruct/bin/activate

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8
export TOKENIZERS_PARALLELISM=true
export TRANSFORMERS_NO_ADVISORY_WARNINGS=true
export OMP_NUM_THREADS=16

export HF_HUB_OFFLINE=1 # Offline mode for Hugging Face
export WANDB_MODE=offline

#export NCCL_IB_DISABLE=0
#export NCCL_IB_HCA=mlx5
#export NCCL_SOCKET_IFNAME=ib0
# Allow P2P communications between all GPUs within the same node (See: nvidia-smi topo -m)
#export NCCL_P2P_LEVEL=NVL

# Maximum time (in seconds) to wait for NCCL operations before timing out
export NCCL_TIMEOUT=14400 # 2 hours, prevent crashing when loading datasets

# Enable asynchronous error handling for NCCL operations
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

nvidia-smi topo -m # See the topology of the nodes

MASTER_PORT=9327
MAIN_PROCESS_IP=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

srun accelerate launch \
    --num_processes 128 \
    --num_machines 32 \
    --mixed_precision bf16 \
    --dynamo_backend "no" \
    --rdzv_backend c10d \
    --main_process_ip $MAIN_PROCESS_IP \
    --main_process_port $MASTER_PORT \
    --machine_rank $SLURM_NODEID \
    --use_fsdp \
    --fsdp_sharding_strategy HYBRID_SHARD \
    --fsdp_device_mesh "(32,4)" \
    --fsdp_state_dict_type SHARDED_STATE_DICT \
    --fsdp_backward_prefetch BACKWARD_PRE \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_cpu_ram_efficient_loading true \
    -m axolotl.cli.train ~/latxa-instruct/model_training/train_configs//exp_0_111_fixed.yaml
