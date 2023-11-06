for model_id in ContinuousSBM_ContinuousVPSDE_I_BPROJ_bottleneck_32_firstc_4_phi_beta_linear_betamax_0.5 ContinuousSBM_ContinuousVPSDE_I_BPROJ_bottleneck_32_firstc_4_phi_beta_cosine_betamax_0.5 ContinuousSBM_ContinuousVPSDE_I_BPROJ_bottleneck_32_firstc_6_phi_beta_linear_betamax_0.5 ContinuousSBM_ContinuousVPSDE_I_BPROJ_bottleneck_32_firstc_6_phi_beta_cosine_betamax_0.5 ContinuousSBM_ContinuousVPSDE_I_BPROJ_bottleneck_32_firstc_10_phi_beta_linear_betamax_0.5 ContinuousSBM_ContinuousVPSDE_I_BPROJ_bottleneck_32_firstc_10_phi_beta_cosine_betamax_0.5 ;do 
sbatch <<EOF
#!/bin/bash
#SBATCH -J ${model_id}
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --constraint="h100|a100"
#SBATCH -o /mnt/home/dheurtel/astroddpm/astroddpm/log/${model_id}.log
source ~/.bashrc
source /mnt/home/dheurtel/venv/genv_DL/bin/activate
python train.py --model_id ${model_id} --config_folder /mnt/home/dheurtel/astroddpm/astroddpm/config/exp14 --all_models True
EOF
done