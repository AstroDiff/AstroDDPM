for model_id in ContinuousSBM_FFResUNet_bottleneck_32_firstc_10_phi_beta_linear_betamax_2.0_betamin_0.01 ContinuousSBM_FFResUNet_bottleneck_32_firstc_10_phi_beta_linear_betamax_2.0_betamin_0.005 ContinuousSBM_FFResUNet_bottleneck_32_firstc_10_phi_beta_linear_betamax_2.0_betamin_0.001 ContinuousSBM_FFResUNet_bottleneck_32_firstc_10_phi_beta_cosine_betamax_2.0_betamin_0.01 ContinuousSBM_FFResUNet_bottleneck_32_firstc_10_phi_beta_cosine_betamax_2.0_betamin_0.005 ContinuousSBM_FFResUNet_bottleneck_32_firstc_10_phi_beta_cosine_betamax_2.0_betamin_0.001 ;do 
sbatch <<EOF
#!/bin/bash
#SBATCH -J ${model_id}
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=96:00:00
#SBATCH --partition=gpu
#SBATCH --constraint="h100|a100"
#SBATCH -o /mnt/home/dheurtel/astroddpm/astroddpm/log/${model_id}.log
source ~/.bashrc
source /mnt/home/dheurtel/venv/genv_DL/bin/activate
python train.py --model_id ${model_id} --config_folder /mnt/home/dheurtel/astroddpm/astroddpm/config/exp17 --all_models True
EOF
done