for model_id in DiscreteSBM_SigmaVPSDE_MHD_BPROJ_N_1000_bottleneck_32 DiscreteSBM_VPSDE_MHD_BPROJ_N_4000_bottleneck_32; do 
sbatch <<EOF
#!/bin/bash
#SBATCH -J ${model_id}
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --constraint=a100
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=a100
#SBATCH -o /mnt/home/dheurtel/astroddpm/astroddpm/log/comparaison_${model_id}.log
source ~/.bashrc
source /mnt/home/dheurtel/venv/genv_DL/bin/activate
python build_comparison.py --model_id ${model_id} --num_samples 20 --noise_min 2e-2 --noise_max 1e2 --num_levels 20 --noise_interp log --methods all --batch_size 20
EOF
    echo "Submitted ${model_id}"
done
