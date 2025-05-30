#!/bin/bash
#SBATCH -A EUHPC_E04_042       # account name
#SBATCH -p boost_usr_prod
#SBATCH --time 24:00:00
#SBATCH --job-name=70B_exp_2_101_fixed
#SBATCH --cpus-per-task=16
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --output=.slurm/70B_exp_2_101_fixed.out
#SBATCH --error=.slurm/70B_exp_2_101_fixed.err
#SBATCH --mem=494000MB

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
export NCCL_TIMEOUT=14400 # 4 hours, prevent crashing when loading datasets
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600 # 1 hour, prevent HEARTBEAT_TIMEOUT error
# Enable asynchronous error handling for NCCL operations
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Enable the memory allocator to allocate memory in a way that is more likely to be compatible with NCCL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Trying things until it works. 
# > By the look of it, NCCL decided that communication between the two CPUs on each node will be faster using the NICs than using UPI.
# > However, this won't work if your NICs can't talk to each other...
#
# > You can try setting NCCL_NET_DISABLE_INTRA=1 to prevent the NICs from being used for such purpose.
# (https://github.com/NVIDIA/nccl/issues/1405)
export NCCL_NET_DISABLE_INTRA=1

nvidia-smi topo -m # See the topology of the nodes

MASTER_PORT=9327
MAIN_PROCESS_IP=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

# Oscar: Run the command directly from the repository
cd ~/axolotl/src
srun accelerate launch \
    --num_processes 256 \
    --num_machines 64 \
    --mixed_precision bf16 \
    --dynamo_backend "no" \
    --rdzv_backend c10d \
    --main_process_ip $MAIN_PROCESS_IP \
    --main_process_port $MASTER_PORT \
    --machine_rank $SLURM_NODEID \
    --use_fsdp \
    --fsdp_sharding_strategy FULL_SHARD \
    --fsdp_state_dict_type SHARDED_STATE_DICT \
    --fsdp_backward_prefetch BACKWARD_PRE \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_cpu_ram_efficient_loading true \
    --fsdp_sync_module_states true \
    -m axolotl.cli.train ~/latxa-instruct/model_training/train_configs/leonardo/Latxa-Llama-3.1-70B-Instruct-exp_2_101.yaml
