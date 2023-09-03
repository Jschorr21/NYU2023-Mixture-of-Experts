# NYU2023-Mixture-of-Experts-Model

## Overview
This reposistory contains transformer model evaluation scripts and framework for creating and evaluating mixture of experts models on MMLU benchmark. Supports evaluation of individual Hugging Face models using model_eval.py as well as evaluation of a mixture of multiple expert models using mixture_model.py

### Benchmark
This repository evaluates transformer models on the Massive Multitask Language Understanding (MMLU) benchmark dataset. The test split contains thousands of multiple choice questions from 57 different academic subjects ranging from high school mathematics to college genetics to professional law.

### Methodology
For each question in the test set, a prompt string is generated containing 5 example questions and answers from the corresponding subject's dev split along with the question for which the model is to determine the answer. For each question during the mixture model evaluation, the mixture_model.py script averages the probability distributions for the next token generated by each model to produce a composite prediction. All results and metrics are computed at the end.

## Installation
```bash
pip install ./requirements.txt
```
## Usage
For model_eval.py, use the following bash code template to run evaluation on any HF transformer:
```bash
python model_eval.py --hf_model_path PATH_TO_HF_MODEL --output_path PATH_TO_OUTPUT_DATA
```

For mixture_model.py, you can add any number of HF model paths into the mixture model. Include 'n' devices for 'n' HF models :
```bash
export CUDA_VISIBLE_DEVICES=0,1,2...n
python mixture_model.py --hf_model_paths PATH_TO_HF_MODEL_1 PATH_TO_HF_MODEL_2 PATH_TO_HF_MODEL_3 --output_path PATH_TO_OUTPUT_DATA
```

When submitting Slurm jobs to an HPC Cluster, we can use the following job script template: 
```bash
#!/bin/bash
#SBATCH --job-name=JOB_NAME
#SBATCH --nodes=1
#SBATCH --partition=a100_short
#SBATCH --ntasks-per-gpu=16
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --mail-type=END
#SBATCH --mail-user=EMAIL
#SBATCH --output=output_files/Model_Name/output-%j.txt
#SBATCH --error=error_files/Model_Name/error-%j.txt

module load miniconda3/gpu/4.9.2

conda activate env

conda install -y transformers datasets
conda install pytorch -c pytorch -y

python model_eval.py --hf_model_path PATH_TO_HF_MODEL --output_path PATH_TO_OUTPUT_DATA
```

## Results
This repository has been used to evaluate three task-specific Hugging Face transofrmer models, as well as their mixture model. The models used included [medAlpaca-7b](https://huggingface.co/medalpaca/medalpaca-7b), [codeAlpaca-7b](https://huggingface.co/allenai/open-instruct-code-alpaca-7b), and [OG LLaMA-7b](https://huggingface.co/decapoda-research/llama-7b-hf).

## Next Steps
