# Latxa-Instruct Evaluation

### Static Benchmarks

The directory [benchmarks/scripts/](benchmarks/scripts/) contains the SLURM scripts to launch static benchmark evaluations with 
EleutherAI's [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) framework.  
Unless specified otherwise, results are written to the [benchmarks/results/](benchmarks/results) directory, which currently contains 
the actual results obtained and reported in the paper.

#### Setup & Execution

1. Create and activate a virtual environment:
    ```shell
    python -m venv /path/to/your/venv
    source /path/to/your/venv/bin/activate
    ```
2. Install LM Evaluation Harness and its dependencies:
    ```shell
    git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
    cd lm-evaluation-harness
    pip install -e .
    ```
3. Set the environment variable (add to your `.bashrc` for persistence):
    ```shell
    export LM_HARNESS_VENV="/path/to/your/venv"
    ```
4. Modify the SLURM scripts as necessary and run them with `sbatch`. For instance:
    ```shell
    sbatch lm_eval_Latxa-Llama-3.1-8B.sh
    ```
   
### Human evaluation: _Ebaluatoia_ Arena

The code to run the arena can be found in [human/frontend/](human/frontend/) and [human/backend/](human/backend/). See the READMEs therein.  
The preference data from Ebaluatoia is available at 🤗 HuggingFace: [HiTZ/ebaluatoia](https://huggingface.co/datasets/HiTZ/ebaluatoia).

### Result analysis

The Jupyter notebook `results.ipynb` contains the code to analyze both the results from the static benchmarks and the arena.  
It also produces the tables and figures included in the paper, which can be found in [figures/](figures/).

#### Setup & Execution

1. Create and activate a virtual environment:
    ```shell
    python -m venv /path/to/your/venv
    source /path/to/your/venv/bin/activate
    ```
2. Install dependencies:
    ```shell
    pip install jupyter numpy pandas seaborn matplotlib scipy
    ```
3. Run the notebook:
    ```shell
    jupyter notebook
    ```