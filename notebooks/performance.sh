sbatch <<EOF
#!/bin/bash
#SBATCH -J performance
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=h100
#SBATCH -o /mnt/home/dheurtel/astroddpm/astroddpm/log/performance.log
source ~/.bashrc
source /mnt/home/dheurtel/venv/genv_DL/bin/activate
python performance.py
EOF