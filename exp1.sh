for model_id in DiscreteSBM_VPSDE_MHD_BPROJ_N_1000_bottleneck_32 ;do 
sbatch <<EOF
#!/bin/bash
#SBATCH -J ${model_id}
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH -o log/${model_id}_teeeeest.log
source ~/.bashrc
source /mnt/home/dheurtel/venv/genv_DL/bin/activate
python train.py --bordel
EOF
    echo "Submitted ${model_id}"
done