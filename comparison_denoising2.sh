for model_id in DiscreteSBM_SigmaVPSDE_MHD_BPROJ_N_1000_bottleneck_32 DiscreteSBM_VPSDE_MHD_BPROJ_N_4000_bottleneck_32 ;do 

python build_comparison.py --model_id ${model_id} --num_samples 20 --noise_min 2e-2 --noise_max 1e2 --num_levels 20 --noise_interp 'log' --methods 'all' --batch_size 20

done
