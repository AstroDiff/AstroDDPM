for sizemin in 32 ; do
    for diff in 8000 10000; do
sbatch <<EOF
#!/bin/bash
#SBATCH -J GRF2_DDPM_skip_GN_bottleneck${sizemin}_diffusion${diff}
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH -o logs_output/GRF2_DDPM_skip_GN_bottleneck${sizemin}_diffusion${diff}.log
source ~/.bashrc
conda activate FIexplore
python train.py --model_id=GRF2_DDPM_skip_GN_bottleneck${sizemin}_diffusion${diff} \
--n_steps=${diff} \
--num_epochs=10000 \
--in_channel=1 \
--save_step_epoch=200 \
--sample_step_epoch=200 \
--num_sample=16 \
--num_result_sample=64 \
--source_dir=/mnt/home/dheurtel/ceph/00_exploration_data/grf/power2 \
--padding_mode=circular \
--normalization=GN \
--size_min=${sizemin} \
--lr=1e-3 \
--optimizer=Adam \
--network=ResUNet \
--skip_rescale \
--sample_folder=/mnt/home/dheurtel/ceph/20_samples/artificial_architecture_exps \
--ckpt_folder=/mnt/home/dheurtel/ceph/10_checkpoints/artificial_architecture_exps \
--random_rotate=True \
--no_lognorm \
--test_set_len=128 \
--lr_scheduler=stepLR
EOF
    printf "GRF2_DDPM_skip_GN_bottleneck${sizemin}_diffusion${diff}:\n\t--model_id=GRF2_DDPM_skip_GN_bottleneck${sizemin}_diffusion${diff}\n\t--n_steps=${diff}\n\t--in_channel=1\n\t--num_epochs=10000\n\t--save_step_epoch=200\n\t--sample_step_epoch=200\n\t--num_sample=16\n\t--num_result_sample=64\n\t--test_set_len=128\n\t--source_dir=/mnt/home/dheurtel/ceph/00_exploration_data/grf/power2\n\t--padding_mode=circular\n\t--normalization=GN\n\t--size_min=${sizemin}\n\t--lr=1e-3\n\t--random_rotate=True\n\t--no_lognorm\n\t--optimizer=Adam\n\t--network=ResUNet\n\t--skip_rescale\n\t--sample_folder=/mnt/home/dheurtel/ceph/20_samples/artificial_architecture_exps\n\t--ckpt_folder=/mnt/home/dheurtel/ceph/10_checkpoints/artificial_architecture_exps\n\t--lr_scheduler=stepLR\n" >> /mnt/home/dheurtel/astro_generative/config/exp6.txt
    done
done


for sizemin in 32 ; do
    for diff in 8000 10000; do
sbatch <<EOF
#!/bin/bash
#SBATCH -J GRFSAME_DDPM_skip_GN_bottleneck${sizemin}_diffusion${diff}
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH -o logs_output/GRFSAME_DDPM_skip_GN_bottleneck${sizemin}_diffusion${diff}.log
source ~/.bashrc
conda activate FIexplore
python train.py --model_id=GRFSAME_DDPM_skip_GN_bottleneck${sizemin}_diffusion${diff} \
--n_steps=${diff} \
--num_epochs=10000 \
--in_channel=1 \
--save_step_epoch=200 \
--sample_step_epoch=200 \
--num_sample=16 \
--num_result_sample=64 \
--source_dir=/mnt/home/dheurtel/ceph/00_exploration_data/grf/same \
--padding_mode=circular \
--normalization=GN \
--size_min=${sizemin} \
--lr=1e-3 \
--optimizer=Adam \
--network=ResUNet \
--skip_rescale \
--sample_folder=/mnt/home/dheurtel/ceph/20_samples/artificial_architecture_exps \
--ckpt_folder=/mnt/home/dheurtel/ceph/10_checkpoints/artificial_architecture_exps \
--random_rotate=True \
--no_lognorm \
--test_set_len=128 \
--lr_scheduler=stepLR
EOF
    printf "GRFSAME_DDPM_skip_GN_bottleneck${sizemin}_diffusion${diff}:\n\t--model_id=GRFSAME_DDPM_skip_GN_bottleneck${sizemin}_diffusion${diff}\n\t--n_steps=${diff}\n\t--in_channel=1\n\t--num_epochs=10000\n\t--save_step_epoch=200\n\t--sample_step_epoch=200\n\t--num_sample=16\n\t--num_result_sample=64\n\t--test_set_len=128\n\t--source_dir=/mnt/home/dheurtel/ceph/00_exploration_data/grf/same\n\t--padding_mode=circular\n\t--normalization=GN\n\t--size_min=${sizemin}\n\t--lr=1e-3\n\t--random_rotate=True\n\t--no_lognorm\n\t--optimizer=Adam\n\t--network=ResUNet\n\t--skip_rescale\n\t--sample_folder=/mnt/home/dheurtel/ceph/20_samples/artificial_architecture_exps\n\t--ckpt_folder=/mnt/home/dheurtel/ceph/10_checkpoints/artificial_architecture_exps\n\t--lr_scheduler=stepLR\n" >> /mnt/home/dheurtel/astro_generative/config/exp7.txt
    done
done