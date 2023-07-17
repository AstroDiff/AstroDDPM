#!/bin/bash
#SBATCH -J DDPM17
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --constraint=a100-80gb
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH -o ddpm1.17.log

source ~/.bashrc
conda activate FIexplore
python train.py -flagfile /mnt/home/dheurtel/astro_generative/config/DDPM_1.17.txt