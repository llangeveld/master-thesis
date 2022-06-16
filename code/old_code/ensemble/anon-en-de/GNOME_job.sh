#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --output=GNOME.out
#SBATCH --mem=64GB
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name=GNOME-anon-en-de

# Get the resources needed
echo "Loading module..."
module purge
module load PyTorch/1.10.0-fosscuda-2020b 
module rm SciPy-bundle/2020.11-fosscuda-2020b

source /data/s3225143/.envs/thesis/bin/activate

#echo "Loading requirements..."
#pip install -q -r ./requirements.txt
export TRANSFORMERS_CACHE=/data/s3225143/cache
export TORCH_HOME=/data/s3225143/cache

echo "Starting program..."
bash GNOME.sh

deactivate
