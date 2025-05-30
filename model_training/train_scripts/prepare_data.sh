#!/bin/bash
#SBATCH -A EUHPC_E04_042       # account name
#SBATCH -p boost_usr_prod
#SBATCH --time 5:00:00
#SBATCH --job-name=prepare_data
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=.slurm/prepare_data.out
#SBATCH --error=.slurm/prepare_data.err
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

mkdir -p ~/work_dir/.cache_oscar/hf
export TMPDIR=~/work_dir/.cache_oscar
export HF_HOME=~/work_dir/.cache_oscar/hf



MASTER_PORT=9327
MAIN_PROCESS_IP=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

python -m axolotl.cli.preprocess ~/latxa-instruct/model_training/train_configs//exp_0_010_fixed.yaml
python -m axolotl.cli.preprocess ~/latxa-instruct/model_training/train_configs//exp_0_011_fixed.yaml
python -m axolotl.cli.preprocess ~/latxa-instruct/model_training/train_configs//exp_0_101_fixed.yaml
python -m axolotl.cli.preprocess ~/latxa-instruct/model_training/train_configs//exp_0_110_fixed.yaml

python -m axolotl.cli.preprocess ~/latxa-instruct/model_training/train_configs//exp_0_111_fixed.yaml
python -m axolotl.cli.preprocess ~/latxa-instruct/model_training/train_configs//exp_1_001_fixed.yaml
python -m axolotl.cli.preprocess ~/latxa-instruct/model_training/train_configs//exp_1_010_fixed.yaml

python -m axolotl.cli.preprocess ~/latxa-instruct/model_training/train_configs//exp_1_011_fixed.yaml
python -m axolotl.cli.preprocess ~/latxa-instruct/model_training/train_configs//exp_1_101_fixed.yaml
python -m axolotl.cli.preprocess ~/latxa-instruct/model_training/train_configs//exp_1_110_fixed.yaml

python -m axolotl.cli.preprocess ~/latxa-instruct/model_training/train_configs//exp_1_111_fixed.yaml
python -m axolotl.cli.preprocess ~/latxa-instruct/model_training/train_configs//exp_2_010_fixed.yaml
python -m axolotl.cli.preprocess ~/latxa-instruct/model_training/train_configs//exp_2_011_fixed.yaml

python -m axolotl.cli.preprocess ~/latxa-instruct/model_training/train_configs//exp_2_100_fixed.yaml
python -m axolotl.cli.preprocess ~/latxa-instruct/model_training/train_configs//exp_2_101_fixed.yaml

python -m axolotl.cli.preprocess ~/latxa-instruct/model_training/train_configs//exp_2_110_fixed.yaml
python -m axolotl.cli.preprocess ~/latxa-instruct/model_training/train_configs//exp_2_111_fixed.yaml

python -m axolotl.cli.preprocess ~/latxa-instruct/model_training/train_configs/leonardo/Latxa-Llama-3.1-70B-Instruct-exp_2_101.yaml

