for sizemin in 32 8; do 
    for diff in 1000; do 
        for wd in 0 1e-3 1e-4; do 
            for first_c in 10 6; do
sbatch <<EOF
#!/bin/bash
#SBATCH -J MHDsame_SigmaDDPM_skip_GN_bottleneck${sizemin}_diffusion${diff}_wd${wd}_csize${first_c}
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH -o logs_output/MHDsame_SigmaDDPM_skip_GN_bottleneck${sizemin}_diffusion${diff}_wd${wd}_csize${first_c}.log
source ~/.bashrc
conda activate FIexplore
python train.py --model_id=MHDsame_SigmaDDPM_skip_GN_bottleneck${sizemin}_diffusion${diff}_wd${wd}_csize${first_c} \
--n_steps=${diff} \
--diffusion_mode=SigmaDDPM \
--num_epochs=10000 \
--in_channel=1 \
--first_c_mult=${first_c} \
--save_step_epoch=200 \
--sample_step_epoch=200 \
--num_sample=16 \
--source_dir=/mnt/home/dheurtel/ceph/00_exploration_data/density/b_proj \
--padding_mode=circular \
--normalization=GN \
--size_min=${sizemin} \
--lr=1e-3 \
--optimizer=AdamW \
--weight_decay=${wd} \
--warmup=100 \
--network=ResUNet \
--skip_rescale \
--sample_folder=/mnt/home/dheurtel/ceph/20_samples/artificial_architecture_exps \
--ckpt_folder=/mnt/home/dheurtel/ceph/10_checkpoints/artificial_architecture_exps \
--random_rotate=True \
--test_set_len=95  \
--power_spectrum=/mnt/home/dheurtel/ceph/00_exploration_data/power_spectra/same.npy \
--lr_scheduler=stepLR
EOF
    printf "MHDsame_SigmaDDPM_skip_GN_bottleneck${sizemin}_diffusion${diff}_wd${wd}_csize${first_c}:\n\t--model_id=MHDsame_SigmaDDPM_skip_GN_bottleneck${sizemin}_diffusion${diff}_wd${wd}_csize${first_c}\n\t--n_steps=${diff}\n\t--in_channel=1\n\t--first_c_mult=${first_c}\n\t--num_epochs=10000\n\t--save_step_epoch=200\n\t--sample_step_epoch=200\n\t--diffusion_mode=SigmaDDPM\n\t--num_sample=16\n\t--test_set_len=128\n\t--source_dir=/mnt/home/dheurtel/ceph/00_exploration_data/density/b_proj\n\t--padding_mode=circular\n\t--normalization=GN\n\t--size_min=${sizemin}\n\t--lr=1e-3\n\t--random_rotate=True\n\t--optimizer=AdamW\n\t--weight_decay=${wd}\n\t--warmup=100\n\t--network=ResUNet\n\t--skip_rescale\n\t--power_spectrum=/mnt/home/dheurtel/ceph/00_exploration_data/power_spectra/same.npy\n\t--sample_folder=/mnt/home/dheurtel/ceph/20_samples/artificial_architecture_exps\n\t--ckpt_folder=/mnt/home/dheurtel/ceph/10_checkpoints/artificial_architecture_exps\n\t--lr_scheduler=stepLR\n" >> /mnt/home/dheurtel/astro_generative/config/exp10.txt
            done
        done
    done
done