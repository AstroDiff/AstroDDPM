sbatch <<EOF
#!/bin/bash
#SBATCH -J MN1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --constraint="h100|a100"
#SBATCH -o /mnt/home/dheurtel/astroddpm/astroddpm/log/MN1.out
source ~/.bashrc
source /mnt/home/dheurtel/venv/genv_DL/bin/activate
python train_moment_network.py --epochs 30000
EOF