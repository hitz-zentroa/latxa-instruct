#!/bin/bash
#SBATCH --job-name=lm_eval_Claude-3.5-Sonnet
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=10GB
#SBATCH --output=.slurm/lm_eval_Claude-3.5-Sonnet.out
#SBATCH --error=.slurm/lm_eval_Claude-3.5-Sonnet.err

source "${LM_HARNESS_VENV}/bin/activate"

model_name="claude-3-5-sonnet-20241022"
batch_size=10
num_fewshot=5

tasks_selected=(
  "belebele_eus_Latn"
  "bertaqa_eu_global"
  "bertaqa_eu_local"
  "eus_proficiency"
  "eus_reading"
  "eus_trivia"
  "eus_exams_eu"
  "mmlu_eu"
)
for group_name in "${tasks_selected[@]}"; do
  srun python3 -m lm_eval \
    --model anthropic-chat-completions \
    --model_args model=$model_name,max_length=10000,temperature=0.0,max_retries=5 \
    --tasks $group_name \
    --device cpu \
    --output_path ../results/ \
    --batch_size ${batch_size} \
    --num_fewshot ${num_fewshot} \
    --system_instruction "Respond always with a single letter: A, B, C or D." \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --log_samples \
    --mcq_to_generative
done

tasks_selected=(
  "mgsm_native_cot_eu"
  "mgsm_native_cot_en"
  "mgsm_native_cot_es"
)
for group_name in "${tasks_selected[@]}"; do
  python3 -m lm_eval \
    --model anthropic-chat-completions \
    --model_args model=$model_name,max_length=10000,temperature=0.0,max_retries=5,max_gen_toks=1024 \
    --tasks $group_name \
    --device cpu \
    --output_path ../results/ \
    --batch_size ${batch_size} \
    --num_fewshot ${num_fewshot} \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --log_samples
done
