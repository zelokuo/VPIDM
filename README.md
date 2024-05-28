# （VPIDM) Variance Preserving Interpolation Diffusion Model for Speech Enhancement  
Official PyTorch implementation of "**[A Variance-Preserving Interpolation Approach for Diffusion Models with Applications to Single Channel Speech Enhancement and Recognition](https://arxiv.org/abs/2405.16952)**" which has been accepted by TASLP 2024. 
[variance-preserving-based interpolation diffusion models for speech enhancement](https://arxiv.org/abs/2306.08527), in which we apply the diffusion model to the speech enhancement (denoising) task. （Diffusion Models for Speech Enhancement）
# Listening Demo
We provide a listening Demo [here](https://zelokuo.github.io/VPIDM_demo) using the model trained on the DNS corpus.
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
@misc{guo2024variancepreserving,
      title={A Variance-Preserving Interpolation Approach for Diffusion Models with Applications to Single Channel Speech Enhancement and Recognition}, 
      author={Zilu Guo and Qing Wang and Jun Du and Jia Pan and Qing-Feng Liu and Chin-Hui},
      year={2024},
      eprint={2405.16952},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}
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
