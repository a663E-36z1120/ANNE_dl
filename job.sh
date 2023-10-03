#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32000M
#SBATCH --time=2:00:00
#SBATCH --account=def-alim
#SBATCH --output=output_%j.out
#SBATCH --error=error_%j.out

module load python
module load scipy-stack

source ./pytorchENV/bin/activate
# pip uninstall numpy -y
pip install --no-index numpy==1.25.0
pip install matplotlib --no-index
pip install scipy --no-index
pip install -r requirements.txt

cd /home/$USER/projects/def-alim/$USER/ANNE_dl
python -u ./models/train.py
