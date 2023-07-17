for norm in LN BN GN None DN; do
    for sizemin in 32 16 8; do
sbatch <<EOF
#!/bin/bash
#SBATCH -J MHD_DDPM_${norm}_bottleneck${sizemin}_toy
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --constraint=a100
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH -o logs_output/MHD_DDPM_${norm}_bottleneck${sizemin}_toy.log
source ~/.bashrc
conda activate FIexplore
python train.py --model_id=MHD_DDPM_${norm}_bottleneck${sizemin}_toy \
--num_epochs=5000 \
--save_step_epoch=200 \
--sample_step_epoch=200 \
--num_sample=16 \
--source_dir=/mnt/home/dheurtel/ceph/00_exploration_data/toy \
--padding_mode=zeros \
--normalization=${norm} \
--size_min=${sizemin} \
--network=unet \
--sample_folder=/mnt/home/dheurtel/ceph/20_samples/artificial_architecture_exps \
--ckpt_folder=/mnt/home/dheurtel/ceph/10_checkpoints/artificial_architecture_exps \
--lr_scheduler=stepLR
EOF
    printf "MHD_DDPM_${norm}_bottleneck${sizemin}_toy:\n\t--model_id=MHD_DDPM_${norm}_bottleneck${sizemin}_toy\n\t--num_epochs=5000 \n\t--save_step_epoch=200 \n\t--sample_step_epoch=200 \n\t--num_samples=16 \n\t--source_dir=/mnt/home/dheurtel/ceph/00_exploration_data/toy\n\t--normalization=${norm} \n\t--size_min=${sizemin} \n\t--network=unet \n\t--lr_scheduler=stepLR\n\t--padding_mode=zeros\n\t--sample_folder=/mnt/home/dheurtel/ceph/20_samples/artificial_architecture_exps\n\t--ckpt_folder=/mnt/home/dheurtel/ceph/10_checkpoints/artificial_architecture_exps\n" >> /mnt/home/dheurtel/astro_generative/config/exp1.txt
    done
done 