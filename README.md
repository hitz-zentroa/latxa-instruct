<p align="center">
    <br>
    <img src="assets/latxa_round.png" style="height: 350px;">
    <br>
    <h1 align="center">Latxa-Instruct: Basque Instruction-Tuned Models and Evaluation Arena</h1>
</p>

<p align="center">
    <a href="https://github.com/hitz-zentroa/latxa-instruct/blob/main/LICENSE"><img alt="GitHub license" src="https://img.shields.io/github/license/hitz-zentroa/latxa-instruct"></a>
    <a href="https://huggingface.co/collections/HiTZ/latxa-65a697e6838b3acc53677304"><img alt="Pretrained Models" src="https://img.shields.io/badge/🤗HuggingFace-Pretrained Models-green"></a>
    <a href="https://www.hitz.eus/en/node/340"><img alt="Blog" src="https://img.shields.io/badge/📒-Blog Post-blue"></a>
    <a href="https://arxiv.org/abs/2403.20266"><img alt="Paper" src="https://img.shields.io/badge/📖-Paper-orange"></a>
    <br>
    <a href="http://www.hitz.eus/"><img src="https://img.shields.io/badge/HiTZ-Basque%20Center%20for%20Language%20Technology-blueviolet"></a>
    <a href="http://www.ixa.eus/?language=en"><img src="https://img.shields.io/badge/IXA-%20NLP%20Group-ff3333"></a>
    <br>
    <br>
</p>

<p align="justify">
<b>Latxa-Instruct</b> is an open-source project for training, evaluating, and benchmarking Basque instruction-tuned language models. It provides:
<ul>
<li>Basque instruction-tuned models based on Llama 3.1 (8B/70B) and Magpie-generated datasets.</li>
<li>A Gradio-based interactive evaluation arena for model comparison, user feedback, and leaderboard generation.</li>
<li>Training and preprocessing scripts for large-scale instruction tuning on the CINECA Leonardo supercomputer.</li>
<li>Open datasets, configs, and tools for reproducible research on Basque LLMs.</li>
</ul>
All models, datasets, and evaluation tools are released under open licenses.
</p>

- 📒 Blog Post: TBA
- 📖 Paper: TBA
- 🤗 Models: [HiTZ/Latxa Instruct](https://huggingface.co/collections/HiTZ/latxa-instruct-682f356091452b0028380804)

---

# Getting Started

## Interactive Evaluation Arena

Launch the Gradio-based frontend to compare models and submit feedback:

```bash
cd frontend
python3 arena_with_user.py
```

- The app will be available at [http://localhost:7887](http://localhost:7887) by default.
- Requires Python 3.9+, Gradio, and Hugging Face Hub.

See [frontend/README.md](frontend/README.md) for details.

## Model Training

Training is performed on the CINECA Leonardo cluster using [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) (a HuggingFace-based framework).

### Steps

1. **Prepare the data**  
   Preprocess instruction datasets and Basque corpora using scripts in [model_training/](model_training/).  
   Example:
   ```bash
   sbatch model_training/train_scripts/prepare_data.sh
   ```
   This will tokenize and cache datasets as specified in the YAML configs.

2. **Train the models**  
   Launch training with the provided SLURM scripts:
   ```bash
   sbatch model_training/train_scripts/Latxa-Llama-3.1-70B-Instruct-exp_2_101.sh
   ```
   See [model_training/README.md](model_training/README.md) for full instructions, environment setup, and troubleshooting.

3. **Merge and transfer checkpoints**  
   After training, merge distributed checkpoints and transfer to the target server using `merge_weights.sh` and `rsync_weights.sh`.

### Training Configs

- All configs are in [model_training/train_configs/](model_training/train_configs/).
- Datasets are in JSONL format, with user/assistant conversations.
- Example config: [exp_1_010_fixed.yaml](model_training/train_configs/exp_1_010_fixed.yaml)

---

# Project Structure

- `frontend/` – Gradio app, API integration, user authentication, scoring, and custom UI.
- `backend/` – Scripts for launching model servers and managing endpoints.
- `model_training/` – Training configs, scripts, and documentation for Axolotl-based training.

---

# Datasets

- **Instruction tuning:** Magpie-generated Basque/English instructions ([see details](https://github.com/magpie-align/magpie)).
- **Pretraining:** Basque corpus (4.3M docs, 4.2B tokens), available on [HuggingFace](https://huggingface.co/datasets/HiTZ/latxa-corpus-v1.1).
- **Evaluation:** Multiple-choice benchmarks (EusProficiency, EusReading, EusTrivia, EusExams).

---

# Citation

If you use Latxa-Instruct, please cite:

```bibtex
@misc{etxaniz2024latxa,
      title={Latxa: An Open Language Model and Evaluation Suite for Basque}, 
      author={Julen Etxaniz and Oscar Sainz and Naiara Perez and Itziar Aldabe and German Rigau and Eneko Agirre and Aitor Ormazabal and Mikel Artetxe and Aitor Soroa},
      year={2024},
      eprint={2403.20266},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

---

# Acknowledgements

This work has been partially supported by the Basque Government (IKER-GAITU project), the Ministerio para la Transformación Digital y de la Función Pública (EU – NextGenerationEU, 2022/TL22/00215335), and trained on the Leonardo supercomputer at CINECA under EuroHPC Joint Undertaking, project EHPC-EXT-2024E01-042.

---