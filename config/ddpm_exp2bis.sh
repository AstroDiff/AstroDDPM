for norm in BN None; do
    for sizemin in 32 8; do
        for wd in 1e-6; do
            for eps in 1e-5 1e-3; do

sbatch <<EOF
#!/bin/bash
#SBATCH -J MHD_DDPM_${norm}_bottleneck${sizemin}_wd${wd}_epsnorm${eps}_dens
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --constraint=a100
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH -o logs_output/MHD_DDPM_${norm}_bottleneck${sizemin}_wd${wd}_epsnorm${eps}_dens.log
source ~/.bashrc
conda activate FIexplore
python train.py --model_id=MHD_DDPM_${norm}_bottleneck${sizemin}_wd${wd}_epsnorm${eps}_dens \
--num_epochs=3000 \
--save_step_epoch=200 \
--sample_step_epoch=200 \
--num_sample=16 \
--source_dir=/mnt/home/dheurtel/ceph/00_exploration_data/density/b_proj \
--padding_mode=circular \
--normalization=${norm} \
--size_min=${sizemin} \
--optimizer=AdamW \
--weight_decay=${wd} \
--eps_norm=${eps} \
--network=unet \
--sample_folder=/mnt/home/dheurtel/ceph/20_samples/artificial_architecture_exps \
--ckpt_folder=/mnt/home/dheurtel/ceph/10_checkpoints/artificial_architecture_exps \
--lr_scheduler=stepLR
EOF
    printf "MHD_DDPM_${norm}_bottleneck${sizemin}_wd${wd}_epsnorm${eps}_dens:\n\t--model_id=MHD_DDPM_${norm}_bottleneck${sizemin}_wd${wd}_epsnorm${eps}_dens\n\t--num_epochs=3000\n\t--save_step_epoch=200\n\t--sample_step_epoch=200\n\t--num_sample=16\n\t--source_dir=/mnt/home/dheurtel/ceph/00_exploration_data/density/b_proj\n\t--padding_mode=circular\n\t--normalization=${norm}\n\t--size_min=${sizemin}\n\t--optimizer=AdamW\n\t--weight_decay=${wd}\n\t--eps_norm=${eps}\n\t--network=unet\n\t--sample_folder=/mnt/home/dheurtel/ceph/20_samples/artificial_architecture_exps\n\t--ckpt_folder=/mnt/home/dheurtel/ceph/10_checkpoints/artificial_architecture_exps\n\t--lr_scheduler=stepLR\n" >> /mnt/home/dheurtel/astro_generative/config/exp2.txt
            done
        done
    done
done 