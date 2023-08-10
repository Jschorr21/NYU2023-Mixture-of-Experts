#!/bin/bash
#SBATCH --job-name=medAlpaca_job
#SBATCH --nodes=1
#SBATCH --partition=a100_short
#SBATCH --ntasks-per-gpu=16
#SBATCH --exclude=a100-4023
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --mail-type=END
#SBATCH --mail-user=jacob.t.schorr@vanderbilt.edu
#SBATCH --output=output_files/medAlpaca/output-%j.txt
#SBATCH --error=error_files/medAlpaca/error-%j.txt

module load miniconda3/gpu/4.9.2

conda activate env

conda install -y transformers datasets
conda install pytorch -c pytorch -y

python model_eval.py --hf_model_path=medalpaca/medalpaca-7b --output_path=data/medAlpaca_Eval_Results_updated_bug2.json




