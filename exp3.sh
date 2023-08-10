for model_id in DiscreteSBM_ND_SigmaVPSDE_MHD_BPROJ_N_1000_bottleneck_32 DiscreteSBM_ND_SigmaVPSDE_MHD_BPROJ_N_1000_bottleneck_16 DiscreteSBM_ND_SigmaVPSDE_MHD_BPROJ_N_4000_bottleneck_32 DiscreteSBM_ND_SigmaVPSDE_MHD_BPROJ_N_4000_bottleneck_16 DiscreteSBM_ND_SigmaVPSDE_MHD_BPROJ_N_8000_bottleneck_32 DiscreteSBM_ND_SigmaVPSDE_MHD_BPROJ_N_8000_bottleneck_16 DiscreteSBM_ND_VPSDE_MHD_BPROJ_N_1000_bottleneck_32 DiscreteSBM_ND_VPSDE_MHD_BPROJ_N_1000_bottleneck_16 DiscreteSBM_ND_VPSDE_MHD_BPROJ_N_4000_bottleneck_32 DiscreteSBM_ND_VPSDE_MHD_BPROJ_N_4000_bottleneck_16 DiscreteSBM_ND_VPSDE_MHD_BPROJ_N_8000_bottleneck_32 DiscreteSBM_ND_VPSDE_MHD_BPROJ_N_8000_bottleneck_16 ;do 
sbatch <<EOF
#!/bin/bash
#SBATCH -J ${model_id}
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH -o /mnt/home/dheurtel/astroddpm/astroddpm/log/${model_id}.log
source ~/.bashrc
source /mnt/home/dheurtel/venv/genv_DL/bin/activate
python train.py --model_id ${model_id} --ckpt_folder /mnt/home/dheurtel/astroddpm/astroddpm/config/exp3 --all_models True
EOF
    echo "Submitted ${model_id}"
done