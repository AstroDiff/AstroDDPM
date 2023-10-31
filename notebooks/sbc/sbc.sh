for model_id in ContinuousSBM_ContinuousVPSDE_I_BPROJ_bottleneck_16_firstc_6_phi_beta_cosine ;do
    for count in 0 1 2 3 4 ;do
sbatch <<EOF
#!/bin/bash
#SBATCH -J sbc_${model_id}_${count}
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --constraint='h100|a100'
#SBATCH -o /mnt/home/dheurtel/astroddpm/astroddpm/log/sbc_${model_id}_${count}.log
source ~/.bashrc
source /mnt/home/dheurtel/venv/genv_DL/bin/activate
python sbc.py --model_id ${model_id} --num_chain=256 --num_sample=1000 --noise_level=0.1 --save_path=/mnt/home/dheurtel/ceph/04_inference/sbc/${model_id}_${count}.pt
EOF
    done
done