# （VPIDM) Variance Preserving Interpolation Diffusion Model for Speech Enhancement  
Official PyTorch implementation of "**[A Variance-Preserving Interpolation Approach for Diffusion Models with Applications to Single Channel Speech Enhancement and Recognition](https://ieeexplore.ieee.org/abstract/document/10547426)**" which has been accepted by TASLP 2024. 
The short version was accepted by Interspeech 2023, found in "**[variance-preserving-based interpolation diffusion models for speech enhancement](https://arxiv.org/abs/2306.08527)**", in which we apply the diffusion model to the speech enhancement (denoising) task. （Diffusion Models for Speech Enhancement）
# Listening Demo
We provide a listening Demo [here](https://zelokuo.github.io/VPIDM_demo) using the model trained on the DNS corpus.
# Introduction
VPIDM in this project is for single-channel speech enhancement/denoising tasks. Overview of training and inferring stages are presented as follows
## The training stage
<div align="center">
<image src="figures/Training Stage.png"  width="500" alt="Overview of training Stage" />
</div>
  
## The Inferring/Enhancing stage
<div align="center">
<image src="figures/Infering Stage.png"  width="1200" alt="Illustration of Inferring Stage" />
</div>

# Preparation
Install requirements in requirements.txt via
```
pip install -r requirements.txt
```

# Training from Scratch
Prepare your dataset dir in the form of 
```
.../mydataset/train/clean
.../mydataset/train/noisy
.../mydataset/valid/clean
.../mydataset/valid/noisy
```
Use the following command for training your VPIDM
```
python train.py --base_dir <your dataset dir, e.g., .../mydataset/ >
                --gpus 4
                --no_wandb
                --sde vpsde
                --eta 1.5
                --beta-max 2
                --N 25
                --t_eps 4e-2
                --logdir <your log dir>
```
Our code is also compatible with [SGMSE+](https://github.com/sp-uhh/sgmse) 
```
python train.py -base_dir <your dataset dir, e.g., .../mydataset/ >
                --gpus 4
                --no_wandb
                --sde ouve
                --theta 1.5
                --N 30
                --t_eps 0.03
                --logdir <your log dir>
```



# Enhancing
For denoising via the VPIDM
```
python enhancement.py --test_dir <test_noisy_dir>
                      --corrector_step 0
                      --N 25
                      --enhanced_dir <outputs_dir>
                      --ckpt <your checkpoint best.ckpt>
```
For denoising via the SGMSE+
```
python enhancement.py --test_dir <test_noisy_dir>
                      --corrector_step 1
                      --N 30
                      --enhanced_dir <outputs_dir>
                      --ckpt <your checkpoint best.ckpt>
```
## Checkpoints 
We release the checkpoint trained on the VoiceBank+Demand dataset [here](https://drive.google.com/file/d/1nkzdsd-LjJNObRHZObh_R6yGeRtxk1If/view?usp=drive_link).

We release the checkpoint trained on the DNS corpus (only using additive noises) [here](https://drive.google.com/file/d/1jDj5daO81nY_9vMNrU_FrVH40a3aj1HR/view?usp=sharing).

# Thanks and Citations
This code is mainly built on the [SGMSE+](https://github.com/sp-uhh/sgmse). grateful for their open-source spirit.
If you find this project helpful, please kindly cite the following papers. 
```
@ARTICLE{Guo_VPIDM,
  author={Guo, Zilu and Wang, Qing and Du, Jun and Pan, Jia and Liu, Qing-Feng and Lee, Chin-Hui},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing}, 
  title={A Variance-Preserving Interpolation Approach for Diffusion Models With Applications to Single Channel Speech Enhancement and Recognition}, 
  year={2024},
  volume={32},
  number={},
  pages={3025-3038},
  keywords={Speech processing;Noise;Noise measurement;Interpolation;Speech enhancement;Task analysis;Mathematical models;Speech enhancement;speech denoising;diffusion model;score-based;interpolating diffusion model},
  doi={10.1109/TASLP.2024.3407533}}
```
```
@inproceedings{guo23_interspeech,
  author={Zilu Guo and Jun Du and Chin-Hui Lee and Yu Gao and Wenbin Zhang},
  title={{Variance-Preserving-Based Interpolation Diffusion Models for Speech Enhancement}},
  year=2023,
  booktitle={Proc. INTERSPEECH 2023},
  pages={1065--1069},
  doi={10.21437/Interspeech.2023-1265}
}
```
```
@article{richter2023speech,
  title={Speech Enhancement and Dereverberation with Diffusion-based Generative Models},
  author={Richter, Julius and Welker, Simon and Lemercier, Jean-Marie and Lay, Bunlong and Gerkmann, Timo},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  volume={31},
  pages={2351-2364},
  year={2023},
  doi={10.1109/TASLP.2023.3285241}
}
```
