for sizemin in 32 16 8; do
    for diff in 1000 2000 4000; do
sbatch <<EOF
#!/bin/bash
#SBATCH -J MHDpower2_SigmaDDPM_skip_GN_bottleneck${sizemin}_diffusion${diff}
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH -o logs_output/MHDpower2_SigmaDDPM_skip_GN_bottleneck${sizemin}_diffusion${diff}.log
source ~/.bashrc
conda activate FIexplore
python train.py --model_id=MHDpower2_SigmaDDPM_skip_GN_bottleneck${sizemin}_diffusion${diff} \
--n_steps=${diff} \
--diffusion_mode=SigmaDDPM \
--num_epochs=10000 \
--in_channel=1 \
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
--random_rotate=True \
--test_set_len=95  \
--power_spectrum=/mnt/home/dheurtel/ceph/00_exploration_data/power_spectra/power2.npy \
--lr_scheduler=stepLR
EOF
    printf "MHDpower2_SigmaDDPM_skip_GN_bottleneck${sizemin}_diffusion${diff}:\n\t--model_id=MHDpower2_SigmaDDPM_skip_GN_bottleneck${sizemin}_diffusion${diff}\n\t--n_steps=${diff}\n\t--in_channel=1\n\t--num_epochs=10000\n\t--save_step_epoch=200\n\t--sample_step_epoch=200\n\t--diffusion_mode=SigmaDDPM\n\t--num_sample=16\n\t--test_set_len=128\n\t--source_dir=/mnt/home/dheurtel/ceph/00_exploration_data/density/b_proj\n\t--padding_mode=circular\n\t--normalization=GN\n\t--size_min=${sizemin}\n\t--lr=1e-3\n\t--random_rotate=True\n\t--optimizer=Adam\n\t--network=ResUNet\n\t--skip_rescale\n\t--power_spectrum=/mnt/home/dheurtel/ceph/00_exploration_data/power_spectra/power2.npy\n\t--sample_folder=/mnt/home/dheurtel/ceph/20_samples/artificial_architecture_exps\n\t--ckpt_folder=/mnt/home/dheurtel/ceph/10_checkpoints/artificial_architecture_exps\n\t--lr_scheduler=stepLR\n" >> /mnt/home/dheurtel/astro_generative/config/exp8.txt
    done
done