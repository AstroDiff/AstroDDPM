for sizemin in 32 16 ; do
    for num_epochs in 2000 5000 10000; do
sbatch <<EOF
#!/bin/bash
#SBATCH -J PLANCKN_DDPM_skip_GN_bottleneck${sizemin}_epochs${num_epochs}_dim2
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH -o logs_output/PLANCKN_DDPM_skip_GN_bottleneck${sizemin}_epochs${num_epochs}_dim2.log
source ~/.bashrc
conda activate FIexplore
python train.py --model_id=PLANCKN_DDPM_skip_GN_bottleneck${sizemin}_epochs${num_epochs}_dim2 \
--n_steps=1000 \
--num_epochs=${num_epochs} \
--in_channel=2 \
--save_step_epoch=200 \
--sample_step_epoch=200 \
--num_sample=16 \
--source_dir=/mnt/home/dheurtel/ceph/00_exploration_data/planck_cham_noise \
--padding_mode=zeros \
--normalization=GN \
--size_min=${sizemin} \
--lr=1e-3 \
--optimizer=Adam \
--network=ResUNet \
--skip_rescale \
--sample_folder=/mnt/home/dheurtel/ceph/20_samples/artificial_architecture_exps \
--ckpt_folder=/mnt/home/dheurtel/ceph/10_checkpoints/artificial_architecture_exps \
--random_rotate=False \
--no_lognorm \
--test_set_len=30 \
--lr_scheduler=stepLR
EOF
    printf "PLANCKN_DDPM_skip_GN_bottleneck${sizemin}_epochs${num_epochs}_dim2:\n\t--model_id=PLANCKN_DDPM_skip_GN_bottleneck${sizemin}_epochs${num_epochs}_dim2\n\t--n_steps=1000\n\t--in_channel=2\n\t--num_epochs=${num_epochs}\n\t--save_step_epoch=200\n\t--sample_step_epoch=200\n\t--num_sample=16\n\t--test_set_len=30\n\t--source_dir=/mnt/home/dheurtel/ceph/00_exploration_data/planck_cham_noise\n\t--padding_mode=zeros\n\t--normalization=GN\n\t--size_min=${sizemin}\n\t--lr=1e-3\n\t--random_rotate=False\n\t--no_lognorm\n\t--optimizer=Adam\n\t--network=ResUNet\n\t--skip_rescale\n\t--sample_folder=/mnt/home/dheurtel/ceph/20_samples/artificial_architecture_exps\n\t--ckpt_folder=/mnt/home/dheurtel/ceph/10_checkpoints/artificial_architecture_exps\n\t--lr_scheduler=stepLR\n" >> /mnt/home/dheurtel/astro_generative/config/exp5.txt
    done
done