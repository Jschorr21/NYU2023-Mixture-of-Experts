#!/bin/bash
#SBATCH --job-name=Mixture_Model_job
#SBATCH --nodes=1
#SBATCH --exclude=a100-4023
#SBATCH --partition=a100_short
#SBATCH --ntasks-per-gpu=16
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:3
#SBATCH --mem=350G
#SBATCH --mail-type=END
#SBATCH --mail-user=jacob.t.schorr@vanderbilt.edu
#SBATCH --output=./output_files/MixModel/output-%j.txt
#SBATCH --error=./error_files/MixModel/error-%j.txt

module load miniconda3/gpu/4.9.2

conda activate env
export CUDA_VISIBLE_DEVICES=0,1,2
conda install -y transformers datasets
conda install pytorch -c pytorch -y

python mixture_model.py --hf_model_paths medalpaca/medalpaca-7b /gpfs/data/oermannlab/public_data/llama_models_hf/7B-hf /gpfs/data/oermannlab/users/js11268/open-instruct-code-alpaca-7b/model-final-diff --output_path data/Mixture_Model_Eval_Results_updated_bug2.json




