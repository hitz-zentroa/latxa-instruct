#!/bin/bash
#SBATCH --job-name=lm_eval_exp_2_010
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --output=.slurm/lm_eval_exp_2_010.out
#SBATCH --error=.slurm/lm_eval_exp_2_010.err

source "${LM_HARNESS_VENV}/bin/activate"

model_name="HiTZ/Latxa-Llama-3.1-8B-Instruct"
revision="exp_2_010"
batch_size=1

tasks_selected=(
  "arc_eu_challenge"
  "arc_eu_easy"
  "belebele_eus_Latn"
  "bertaqa_eu_global"
  "bertaqa_eu_local"
  "eus_exams_eu"
  "eus_proficiency"
  "eus_reading"
  "eus_trivia"
  "mgsm_native_cot_eu"
  "piqa_eu"
  "xstorycloze_eu"
  "arc_challenge"
  "arc_easy"
  "belebele_eng_Latn"
  "bertaqa_en_global"
  "bertaqa_en_local"
  "mgsm_native_cot_en"
  "piqa"
  "xstorycloze_en"
  "belebele_spa_Latn"
  "openbookqa_es"
  "mgsm_native_cot_es"
  "xstorycloze_es"
  "bl2mp"
  "mmlu_eu"
  "mmlu_en"
  "bbq"
  "basqbbq"
)
for group_name in "${tasks_selected[@]}"; do
    if [[ $group_name == "bbq" || $group_name == "basqbbq" ]]; then
        num_fewshot=4
    else
        num_fewshot=5
    fi
    srun python3 -m lm_eval \
        --model hf \
        --model_args pretrained=$model_name,attn_implementation=flash_attention_2,dtype=bfloat16,revision=$revision \
        --tasks $group_name \
        --device cuda \
        --output_path ../results/HiTZ__Latxa-Llama-3.1-8B-Instruct__exp_2_010/results.json \
        --batch_size ${batch_size} \
        --num_fewshot ${num_fewshot} \
        --log_samples \
        --apply_chat_template \
        --fewshot_as_multiturn
done