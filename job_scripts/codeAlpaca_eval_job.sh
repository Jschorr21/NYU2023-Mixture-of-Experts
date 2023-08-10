#!/bin/bash
#SBATCH --job-name=codeAlpaca_job
#SBATCH --nodes=1
#SBATCH --partition=a100_short
#SBATCH --ntasks-per-gpu=16
#SBATCH --time=10:00:00
#SBATCH --exclude=a100-4023
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --mail-type=END
#SBATCH --mail-user=jacob.t.schorr@vanderbilt.edu
#SBATCH --output=output_files/codeAlpaca/codeAlpaca-output-%j.txt
#SBATCH --error=error_files/codeAlpaca/codeAlpaca-error-%j.txt

module load miniconda3/gpu/4.9.2

conda activate env

conda install -y transformers datasets
conda install pytorch -c pytorch -y

python model_eval.py --hf_model_path=/gpfs/data/oermannlab/users/js11268/open-instruct-code-alpaca-7b/model-final-diff --output_path=data/codeAlpaca_Eval_Results_updated_bug.json
