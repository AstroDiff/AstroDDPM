for model_id in DiscreteSBM_MultiSigmaVPSDE_I_BPROJ_N_2000_bottleneck_32_firstc_6 DiscreteSBM_MultiSigmaVPSDE_I_BPROJ_N_4000_bottleneck_32_firstc_6 DiscreteSBM_MultiSigmaVPSDE_I_BPROJ_N_8000_bottleneck_32_firstc_6 DiscreteSBM_MultiSigmaVPSDE_I_BPROJ_N_2000_bottleneck_32_firstc_10 DiscreteSBM_MultiSigmaVPSDE_I_BPROJ_N_4000_bottleneck_32_firstc_10 DiscreteSBM_MultiSigmaVPSDE_I_BPROJ_N_8000_bottleneck_32_firstc_10 DiscreteSBM_MultiSigmaVPSDE_I_BPROJ_N_2000_bottleneck_16_firstc_6 DiscreteSBM_MultiSigmaVPSDE_I_BPROJ_N_4000_bottleneck_16_firstc_6 DiscreteSBM_MultiSigmaVPSDE_I_BPROJ_N_8000_bottleneck_16_firstc_6 DiscreteSBM_MultiSigmaVPSDE_I_BPROJ_N_2000_bottleneck_16_firstc_10 DiscreteSBM_MultiSigmaVPSDE_I_BPROJ_N_4000_bottleneck_16_firstc_10 DiscreteSBM_MultiSigmaVPSDE_I_BPROJ_N_8000_bottleneck_16_firstc_10 ;do 
sbatch <<EOF
#!/bin/bash
#SBATCH -J ${model_id}
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --constraint="h100|a100"
#SBATCH -o /mnt/home/dheurtel/astroddpm/astroddpm/log/${model_id}.log
source ~/.bashrc
source /mnt/home/dheurtel/venv/genv_DL/bin/activate
python train.py --model_id ${model_id} --config_folder /mnt/home/dheurtel/astroddpm/astroddpm/config/exp8 --all_models True
EOF
done