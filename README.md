# VPIDM
This is the official repository for  [variance-preserving-based interpolation diffusion models for speech enhancement](https://arxiv.org/abs/2306.08527), in which we apply the diffusion model to speech enhancement (denoising) task.

# Preparation
Install requirements in requirements.txt via
```
pip install -r requirements.txt
````

# Training 
```
python train.py --base_dir <your vbd dataset dir>
                --gpus 4
                --no_wandb
                --sde vpsde
                --eta 1.5
                --beta-max 2
                --N 25
                --t_eps 4e-2
                --logdir <your log dir>
```

# Enhancing
```
python enhancement.py --test_dir exp_dir
                      --corrector_step 0
                      --N 25
                      --enhanced_dir /outputs
                      --ckpt <your checkpoint best.ckpt>
```
