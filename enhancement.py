import glob
from argparse import ArgumentParser
import os
from os.path import join

import torch, torchaudio
from soundfile import write
from torchaudio import load
from tqdm import tqdm

from sgmse.model import ScoreModel
from sgmse.util.other import ensure_dir, pad_spec

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--test_dir", type=str, required=True, help='Directory containing the test data (must have subdirectory noisy/)')
    parser.add_argument("--enhanced_dir", type=str, required=True, help='Directory containing the enhanced data')
    parser.add_argument("--clean_dir", type=str, default=None, help='clean directory')
    parser.add_argument("--ckpt", type=str,  help='Path to model checkpoint.')
    parser.add_argument("--corrector", type=str, choices=("ald", "langevin", "none"), default="ald", help="Corrector class for the PC sampler.")
    parser.add_argument("--corrector_steps", type=int, default=1, help="Number of corrector steps")
    parser.add_argument("--snr", type=float, default=0.5, help="SNR value for (annealed) Langevin dynmaics.")
    parser.add_argument("--samplerate", type=int, default=16000, help="sample rate of the test audio.")
    parser.add_argument("--resolution", type=float, default=None, help="time interval")
    parser.add_argument("--N", type=int, default=30, help="Number of reverse steps")
    parser.add_argument("--T", type=float, default=None, help="Tiem of reverse")
    parser.add_argument("--t_eps", type=float, default=0.03, help="the minima time stamp")
    parser.add_argument("--probability_flow", action='store_true', default=False, help="probability flow")
    parser.add_argument("--t_eps_c", type=float, default=0.001, help="the minima state index")
    args = parser.parse_args()

    noisy_dir = join(args.test_dir, 'noisy/')
    if not os.path.exists(noisy_dir):
        noisy_dir = args.test_dir
    clean_dir = join(args.test_dir, 'clean/')
    if not os.path.exists(clean_dir):
        clean_dir =  noisy_dir
    checkpoint_file = args.ckpt
    corrector_cls = args.corrector
    corrector_steps = args.corrector_steps
    probability_flow = args.probability_flow

    target_dir = args.enhanced_dir
    resolution = args.resolution
    samplerate = args.samplerate
    ensure_dir(target_dir)
    t_eps_c = args.t_eps_c
    t_eps = args.t_eps


    # Settings
    sr = 16000
    snr = args.snr
    N = args.N
    T = args.T
    resampler = torchaudio.transforms.Resample(samplerate, sr)

    # Load score model 
    model = ScoreModel.load_from_checkpoint(checkpoint_file, base_dir='', batch_size=16, num_workers=0, kwargs=dict(gpu=False))
    model.eval(no_ema=False)
    model.cuda()


    noisy_files = sorted(glob.glob('{}/*.wav'.format(noisy_dir)))
    clean_files = sorted(glob.glob('{}/*.wav'.format(clean_dir)))
    i = 0
    for noisy_file in tqdm(noisy_files):
        clean_file = clean_files[i]
        i += 1
        filename = noisy_file.split('/')[-1]
        cleanname = clean_file.split('/')[-1]
        
        name = os.path.splitext(filename)[0]


        # Load wav
        y, _ = load(noisy_file.strip()) 
        x, _ = load(clean_file.strip()) 
        if sr != samplerate:
            x = resampler(x)
            y = resampler(y)
        T_orig = y.size(1)   

        # Normalize
        norm_factor = y.abs().max()
        y = y / norm_factor
        x = x / norm_factor
        
        # Prepare DNN input
        Y = torch.unsqueeze(model._forward_transform(model._stft(y.cuda())), 0)
        Y = pad_spec(Y)
        # Reverse sampling
        sampler = model.get_pc_sampler(
                'reverse_diffusion', corrector_cls, Y.cuda(), X=None,  N=N, T=T,  
                probability_flow = probability_flow,
                resolution=resolution,
                T_orig=T_orig,
                corrector_steps=corrector_steps, snr=snr)
        sample, _ = sampler()
        
        # Backward transform in time domain
        x_hat = model.to_audio(sample.squeeze(), T_orig)

        # Renormalize
        x_hat = x_hat * norm_factor

        # Write enhanced wav file
        filename = join(target_dir, filename)
        write(filename, x_hat.cpu().numpy(), 16000)
        torch.cuda.empty_cache()
