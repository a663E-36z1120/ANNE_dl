#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gres=gpu:1 # request a GPU
#SBATCH --tasks-per-node=1 
#SBATCH --cpus-per-task=1 # change this parameter to 2,4,6,... and increase "--num_workers" accordingly to see the effect on performance
#SBATCH --mem=8G      
#SBATCH --time=2:00:00
#SBATCH --output=output_%j.out
#SBATCH --error=error_%j.out
#SBATCH --account=def-alim

module load python # Using Default Python version - Make sure to choose a version that suits your application
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install torch torchvision --no-index
pip install --no-index numpy==1.25.0
pip install matplotlib --no-index
pip install scipy --no-index

cd /home/$USER/projects/def-alim/$USER/ANNE_dl
pip install -r requirements.txt
python -u ./models/train.py

python -u ./models/train.py