for n_steps in 2000 4000 ; do
    for sizemin in 32 16 8; do
sbatch <<EOF
#!/bin/bash
#SBATCH -J MHD_DDPM_skip_GN_bottleneck${sizemin}_dens
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --constraint=a100-80gb,sxm4
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH -o logs_output/MHD_DDPM_skip_GN_bottleneck${sizemin}_lr1e-3_diff${n_steps}_dens.log
source ~/.bashrc
conda activate FIexplore
python train.py --model_id=MHD_DDPM_skip_GN_bottleneck${sizemin}_lr1e-3_diff${n_steps}_dens \
--n_steps=${n_steps} \
--num_epochs=10000 \
--save_step_epoch=200 \
--sample_step_epoch=200 \
--num_sample=16 \
--source_dir=/mnt/home/dheurtel/ceph/00_exploration_data/density/b_proj \
--padding_mode=circular \
--normalization=GN \
--size_min=${sizemin} \
--lr=1e-3 \
--optimizer=Adam \
--network=ResUNet \
--skip_rescale \
--sample_folder=/mnt/home/dheurtel/ceph/20_samples/artificial_architecture_exps \
--ckpt_folder=/mnt/home/dheurtel/ceph/10_checkpoints/artificial_architecture_exps \
--lr_scheduler=stepLR
EOF
    printf "MHD_DDPM_skip_GN_bottleneck${sizemin}_lr1e-3_diff${n_steps}_dens:\n\t--model_id=MHD_DDPM_skip_GN_bottleneck${sizemin}_lr1e-3_diff${n_steps}_dens\n\t--n_steps=${n_steps}\n\t--num_epochs=3000\n\t--save_step_epoch=200\n\t--sample_step_epoch=200\n\t--num_sample=16\n\t--source_dir=/mnt/home/dheurtel/ceph/00_exploration_data/density/b_proj\n\t--padding_mode=circular\n\t--normalization=GN\n\t--size_min=${sizemin}\n\t--lr=1e-3\n\t--optimizer=Adam\n\t--network=ResUNet\n\t--skip_rescale\n\t--sample_folder=/mnt/home/dheurtel/ceph/20_samples/artificial_architecture_exps\n\t--ckpt_folder=/mnt/home/dheurtel/ceph/10_checkpoints/artificial_architecture_exps\n\t--lr_scheduler=stepLR\n" >> /mnt/home/dheurtel/astro_generative/config/exp3bis.txt
    done
done