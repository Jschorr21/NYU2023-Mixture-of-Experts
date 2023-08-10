#!/bin/bash
#SBATCH --job-name=OGLLaMA_job
#SBATCH --nodes=1
#SBATCH --partition=a100_short
#SBATCH --exclude=a100-4023
#SBATCH --ntasks-per-gpu=16
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --mail-type=END
#SBATCH --mail-user=jacob.t.schorr@vanderbilt.edu
#SBATCH --output=output_files/OGllama/output-%j.txt
#SBATCH --error=error_files/OGLlama/error-%j.txt

module load miniconda3/gpu/4.9.2

conda activate env

conda install -y transformers datasets
conda install pytorch -c pytorch -y

python model_eval.py --hf_model_path=/gpfs/data/oermannlab/public_data/llama_models_hf/7B-hf --output_path=data/OGLLaMA_Eval_Results_updated_bug.json




