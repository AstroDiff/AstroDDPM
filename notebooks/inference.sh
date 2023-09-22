for phi_0 in 0.2 0.4 0.6 0.8 ;do
   for phi_1 in 0.2 0.4 0.6 0.8 ;do
sbatch <<EOF
#!/bin/bash
#SBATCH -J inference_${phi_0}_${phi_1}
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=a100
#SBATCH -o /mnt/home/dheurtel/astroddpm/astroddpm/log/inference_${phi_0}_${phi_1}.log
source ~/.bashrc
source /mnt/home/dheurtel/venv/genv_DL/bin/activate
python inference.py --phi_target ${phi_0} ${phi_1}
EOF
    done
done