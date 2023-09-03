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
| Subject                                | medAlpaca-7b | codeAlpaca-7b | OGLLaMA-7b | Mixture Model |
|----------------------------------------|--------------|---------------|------------|---------------|
| Total Accuracy                   | 47.14%        | 34.99%         | 35.39%        | 40.51%         |
| abstract_algebra                       | 33.0%        | 31.0%         | 29.0%      | 28.0%         |
| anatomy                                | 51.11%       | 37.04%        | 39.26%     | 47.41%        |
| astronomy                              | 46.05%       | 33.55%        | 35.53%     | 36.18%        |
| business_ethics                        | 47.0%        | 43.0%         | 40.0%      | 43.0%         |
| clinical_knowledge                     | 60.75%       | 41.13%        | 36.23%     | 46.42%        |
| college_biology                        | 55.56%       | 36.11%        | 37.5%      | 43.75%        |
| college_chemistry                      | 36.0%        | 31.0%         | 36.0%      | 27.0%         |
| college_computer_science                | 25.0%        | 31.0%         | 31.0%      | 28.0%         |
| college_mathematics                     | 27.0%        | 22.0%         | 30.0%      | 26.0%         |
| college_medicine                       | 53.76%       | 30.06%        | 31.79%     | 40.46%        |
| college_physics                        | 34.31%       | 18.63%        | 24.51%     | 28.43%        |
| computer_security                      | 61.0%        | 44.0%         | 46.0%      | 51.0%         |
| conceptual_physics                     | 37.02%       | 35.32%        | 35.74%     | 38.72%        |
| econometrics                           | 29.82%       | 25.44%        | 26.32%     | 28.07%        |
| electrical_engineering                  | 40.69%       | 26.21%        | 26.9%      | 31.72%        |
| elementary_mathematics                  | 26.19%       | 26.98%        | 26.19%     | 26.98%        |
| formal_logic                           | 19.05%       | 24.6%         | 21.43%     | 21.43%        |
| global_facts                           | 35.0%        | 30.0%         | 30.0%      | 33.0%         |
| high_school_biology                    | 58.71%       | 33.55%        | 34.84%     | 45.48%        |
| high_school_chemistry                  | 45.81%       | 26.11%        | 29.56%     | 29.56%        |
| high_school_computer_science            | 39.0%        | 28.0%         | 34.0%      | 34.0%         |
| high_school_european_history           | 54.55%       | 40.0%         | 38.18%     | 52.12%        |
| high_school_geography                  | 54.55%       | 41.92%        | 33.84%     | 38.89%        |
| high_school_government_and_politics    | 64.25%       | 40.93%        | 42.49%     | 49.22%        |
| high_school_macroeconomics             | 42.56%       | 33.59%        | 34.1%      | 34.1%         |
| high_school_mathematics                | 25.56%       | 21.11%        | 26.3%      | 23.7%         |
| high_school_microeconomics             | 37.39%       | 34.03%        | 33.19%     | 33.61%        |
| high_school_physics                    | 27.81%       | 27.15%        | 27.81%     | 28.48%        |
| high_school_psychology                 | 72.29%       | 44.4%         | 45.69%     | 61.65%        |
| high_school_statistics                | 23.15%       | 31.02%        | 33.33%     | 24.54%        |
| high_school_us_history                 | 64.71%       | 45.59%        | 39.22%     | 58.33%        |
| high_school_world_history              | 64.98%       | 38.82%        | 41.23%     | 60.07%        |
| human_aging                           | 63.68%       | 33.01%        | 40.51%     | 50.45%        |
| human_sexuality                       | 56.49%       | 37.51%        | 39.86%     | 50.67%        |
| international_law                     | 61.98%       | 38.37%        | 42.05%     | 49.0%         |
| jurisprudence                         | 52.78%       | 36.24%        | 35.03%     | 42.51%        |
| logical_fallacies                     | 50.92%       | 33.68%        | 33.53%     | 40.0%         |
| machine_learning                      | 32.14%       | 29.96%        | 31.66%     | 31.38%        |
| management                           | 49.51%       | 36.89%        | 39.68%     | 41.71%        |
| marketing                             | 66.24%       | 38.27%        | 42.81%     | 47.55%        |
| medical_genetics                      | 65.0%        | 32.69%        | 37.17%     | 49.75%        |
| miscellaneous                         | 63.22%       | 34.89%        | 37.0%      | 46.77%        |
| moral_disputes                        | 47.69%       | 32.46%        | 33.09%     | 34.32%        |
| moral_scenarios                       | 24.25%       | 27.38%        | 26.0%      | 25.75%        |
| nutrition                             | 58.82%       | 37.13%        | 37.16%     | 44.21%        |
| philosophy                            | 49.84%       | 32.97%        | 34.33%     | 36.99%        |
| prehistory                            | 52.78%       | 33.0%         | 34.1%      | 41.18%        |
| professional_accounting              | 34.75%       | 32.25%        | 31.75%     | 33.75%        |
| professional_law                     | 36.25%       | 32.25%        | 34.75%     | 33.0%         |
| professional_medicine                | 66.91%       | 36.25%        | 36.51%     | 49.06%        |
| professional_psychology              | 56.37%       | 35.94%        | 36.86%     | 43.74%        |
| public_relations                     | 53.64%       | 33.86%        | 35.78%     | 40.35%        |
| security_studies                     | 45.31%       | 33.37%        | 34.19%     | 38.2%         |
| sociology                            | 51.24%       | 33.96%        | 34.3%      | 38.67%        |
| us_foreign_policy                    | 62.0%        | 35.58%        | 38.37%     | 44.89%        |
| virology                             | 53.01%       | 28.36%        | 30.49%     | 35.06%        |
| world_religions                      | 51.46%       | 30.82%        | 33.18%     | 39.15%        |


## Next Steps
While this experiment provides a lot of insight into how transformer models can be combined to increase performence, there is still more research to be done for our ultimate goal of learning the best way to train a Large Language Model. Some adaptations we can try in order to further improve performence in the future include: 

  *Fine tuning a model to weight the average of the models differently depending on the nature of the question

  *Adding more models to the mixture to explore when (and if) returns diminish when using more models/more granular and subspecialized expert models.

  *Manually pretraining models using extrememly specific distinct sets of corpus for each model. This might help us answer the question of whether the domain specificity arises through pertaining corpus or if a model becomes domain specific during fine-tuning.

This experiment has been a baseline for future exploration into the nature of LLMs and how we can best train and utilize them
