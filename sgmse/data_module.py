
from os.path import join
import torch, torchaudio
import pytorch_lightning as pl
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from glob import glob
from torchaudio import load
import numpy as np
import torch.nn.functional as F
import random


def get_window(window_type, window_length):
    if window_type == 'sqrthann':
        return torch.sqrt(torch.hann_window(window_length, periodic=True))
    elif window_type == 'hann':
        return torch.hann_window(window_length, periodic=True)
    else:
        raise NotImplementedError(f"Window type {window_type} not implemented!")

def apply_echo(clean_wav, rir_wav):
    n_clean, n_rir = clean_wav.shape[-1], rir_wav.shape[-1]
    n = n_clean + n_rir - 1
    y_spec = torch.fft.rfft(clean_wav, n=n)*torch.fft.rfft(rir_wav, n=n)
    y_wav = torch.fft.irfft(y_spec, n=n)
    start = (n_rir - 1)//2
    y_wav = y_wav[..., start:start + n_clean]
    r1 = (clean_wav**2).mean()
    y_wav = (y_wav/((y_wav**2).mean() + 1e-8))*r1
    m2 = y_wav.abs().max()
    if m2 < 1e-5 :
        y_wav = clean_wav
    return y_wav




class Specs(Dataset):
    def __init__(self, data_dir, subset, dummy, shuffle_spec, num_frames,
            format='default', normalize="noisy", spec_transform=None,
            rir=False, rir_prob=0.5, no_reverb_as_target=False,
            stft_kwargs=None, **ignored_kwargs):

        # Read file paths according to file naming format.
        if format == "default":
            self.clean_files = sorted(glob(join(data_dir, subset) + '/clean/*.wav'))
            self.noisy_files = sorted(glob(join(data_dir, subset) + '/noisy/*.wav'))
        elif format == 'librispeech':
            if subset == 'train':
                mode = 'train'
            else:
                mode = 'dev'
            with open(data_dir + f'librispeech_{mode}_clean_paths.txt', 'r') as f:
                self.clean_files = f.readlines()

            with open(data_dir + f'librispeech_{mode}_noisy_paths.txt', 'r') as f:
                self.noisy_files = f.readlines()
        elif format == 'chime4':
            if subset == 'train':
                mode = 'train'
            else:
                mode = 'dev_simu'
            with open(data_dir + f'chime4_{mode}_clean.txt', 'r') as f:
                self.clean_files = f.readlines()

            with open(data_dir + f'chime4_{mode}_noisy.txt', 'r') as f:
                self.noisy_files = f.readlines()

        elif format == 'dns':
            if subset == 'train':
                with open(data_dir + 'dns_clean.txt', 'r') as f:
                    self.clean_files = f.readlines()
                with open(data_dir + 'dns_noisy.txt', 'r') as f:
                    self.noisy_files = f.readlines()
                with open(data_dir + 'dns_noise.txt', 'r') as f:
                    self.noise_files = f.readlines()

                with open(data_dir + 'dns_rir_16k_sml_with_rt60.txt', 'r') as f:
                    self.rir_files = f.readlines()
            else:
                with open(data_dir + f'chime4_dev_simu_clean.txt', 'r') as f:
                #with open(data_dir + f'dns3_synthetic_no_reverb_clean.txt', 'r') as f:
                #with open(data_dir + f'dns3_synthetic_clean.txt', 'r') as f:
                    self.clean_files = f.readlines()

                with open(data_dir + f'chime4_dev_simu_noisy.txt', 'r') as f:
                #with open(data_dir + f'dns3_synthetic_no_reverb_noisy.txt', 'r') as f:
                #with open(data_dir + f'dns3_synthetic_noisy.txt', 'r') as f:
                    self.noisy_files = f.readlines()
 
        else:
            # Feel free to add your own directory format
            raise NotImplementedError(f"Directory format {format} unknown!")

        self.dummy = dummy
        self.num_frames = num_frames
        self.shuffle_spec = shuffle_spec
        self.normalize = normalize
        self.spec_transform = spec_transform
        self.rir = rir
        self.rir_prob = rir_prob
        self.no_reverb_as_target = no_reverb_as_target and rir 
        self.clean_noise_pair = True if format == 'dns' and subset == 'train'  else False
        self.format = format
        self.subset = subset

        assert all(k in stft_kwargs.keys() for k in ["n_fft", "hop_length", "center", "window"]), "misconfigured STFT kwargs"
        self.stft_kwargs = stft_kwargs
        self.hop_length = self.stft_kwargs["hop_length"]
        assert self.stft_kwargs.get("center", None) == True, "'center' must be True for current implementation"


    def __prepare_wave(self, wav, isclean=False, path=None):
        target_len = (self.num_frames - 1) * self.hop_length
        original = wav.clone()
        current_len = wav.size(-1)
        pad = max(target_len - current_len, 0)
        if pad == 0:
            # extract random part of the audio file
            if self.shuffle_spec:
                start = int(np.random.uniform(0, current_len-target_len))
            else:
                    start = int((current_len-target_len)/2)
            wav = wav[..., start:start+target_len]

            if wav.abs().max() < 1e-3 and self.clean_noise_pair and isclean :
                original = torchaudio.functional.vad(original, sample_rate=16000) 
                if original.shape[-1] >= target_len:
                    wav = wav[..., :target_len]
        else:
            # pad audio if the length T is smaller than num_frames
            wav = F.pad(wav, (pad//2, pad//2+(pad%2)), mode='constant')

        if isclean and self.rir and np.random.rand() < self.rir_prob :
            rir_path = np.random.choice(self.rir_files).strip()
            rir_path, rt60 = rir_path.split(',')
            rir_wav, _ = load(rir_path)
            reverb_wav = apply_echo(wav, rir_wav)
            rt60 = float(rt60)
            if rt60 > 0.25:
                rts_rir_wav, _ = RTS(rir_wav, rt60)
                wav = apply_echo(wav, rts_rir_wav)
            return reverb_wav, wav
        
        return wav, wav


    def __getitem__(self, i):
        x_path = self.clean_files[i].strip()
        y_path = np.random.choice(self.noise_files).strip() if self.clean_noise_pair else self.noisy_files[i].strip()
        x, _ = load(x_path)
        y, _ = load(y_path)


        if self.clean_noise_pair:
            x, x0 = self.__prepare_wave(x, True, path=x_path)
            y, _ = self.__prepare_wave(y, path=y_path)
            snr = 10**(- random.randrange(-5, 26)/20)
            z = x + torch.sqrt(torch.sum(x**2)/(torch.sum(y**2) + 1e-8))*snr*y
            if torch.any(torch.isnan(z)):
                z = x + y
                print('nan', torch.mean(y**2))
            y = z
            x = x0 if self.no_reverb_as_target else x
        else:
            target_len = (self.num_frames - 1) * self.hop_length
            current_len = x.size(-1)
            pad = max(target_len - current_len, 0)
            if pad == 0:
                # extract random part of the audio file
                if self.shuffle_spec:
                    start = int(np.random.uniform(0, current_len-target_len))
                else:
                    start = int((current_len-target_len)/2)
                x = x[..., start:start+target_len]
                y = y[..., start:start+target_len]
            else:
                # pad audio if the length T is smaller than num_frames
                x = F.pad(x, (pad//2, pad//2+(pad%2)), mode='constant')
                y = F.pad(y, (pad//2, pad//2+(pad%2)), mode='constant')



        if self.normalize == "noisy":
            normfac = y.abs().max()
        elif self.normalize == "clean":
            normfac = x.abs().max()
        elif self.normalize == "not":
            normfac = 1.0
        x = x / (normfac + 1e-8)
        y = y / (normfac + 1e-8)


        if torch.any(torch.isnan(x)) or torch.any(torch.isnan(y)):
            print('audio', self.clean_files[i])
            print('y', torch.any(torch.isnan(y)), normfac)
            print('x', torch.any(torch.isnan(x)))
        X = torch.stft(x, **self.stft_kwargs)
        Y = torch.stft(y, **self.stft_kwargs)

        X, Y = self.spec_transform(X), self.spec_transform(Y)
        if torch.any(torch.isnan(X)) or torch.any(torch.isnan(Y)):
            print(self.clean_files[i])
            print(self.noisy_files[i])
        return X, Y

    def __len__(self):
        if self.dummy:
            # for debugging shrink the data set size
            return int(len(self.clean_files)/200)
        else:
            return len(self.clean_files)


class SpecsDataModule(pl.LightningDataModule):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--base_dir", type=str, required=True, help="The base directory of the dataset. Should contain `train`, `valid` and `test` subdirectories, each of which contain `clean` and `noisy` subdirectories.")
        parser.add_argument("--format", type=str, choices=("default", "dns", "librispeech", "chime4"), default="default", help="Read file paths according to file naming format.")
        parser.add_argument("--batch_size", type=int, default=8, help="The batch size. 8 by default.")
        parser.add_argument("--n_fft", type=int, default=510, help="Number of FFT bins. 510 by default.")   # to assure 256 freq bins
        parser.add_argument("--win_length", type=int, default=None, help="Number of window_length. 510 by default.")   # to assure 256 freq bins
        parser.add_argument("--hop_length", type=int, default=128, help="Window hop length. 128 by default.")
        parser.add_argument("--num_frames", type=int, default=256, help="Number of frames for the dataset. 256 by default.")
        parser.add_argument("--window", type=str, choices=("sqrthann", "hann"), default="hann", help="The window function to use for the STFT. 'hann' by default.")
        parser.add_argument("--num_workers", type=int, default=4, help="Number of workers to use for DataLoaders. 4 by default.")
        parser.add_argument("--dummy", action="store_true", help="Use reduced dummy dataset for prototyping.")
        parser.add_argument("--spec_factor", type=float, default=0.15, help="Factor to multiply complex STFT coefficients by. 0.15 by default.")
        parser.add_argument("--spec_abs_exponent", type=float, default=0.5, help="Exponent e for the transformation abs(z)**e * exp(1j*angle(z)). 0.5 by default.")
        parser.add_argument("--normalize", type=str, choices=("clean", "noisy", "not"), default="noisy", help="Normalize the input waveforms by the clean signal, the noisy signal, or not at all.")
        parser.add_argument("--transform_type", type=str, choices=("exponent", "log", "none", "scale", "complex_exponent"), default="exponent", help="Spectogram transformation for input representation.")
        parser.add_argument("--rir", action='store_true', help="If use room impulse response")
        parser.add_argument("--no_reverb_as_target", action='store_true', help="Use the clean wave as target rather than the reverbed version")
        parser.add_argument("--rir_prob", type=float, default=0.5, help="The probability of applying room impulse response, if the 'rir' is ture.")
        return parser

    def __init__(
        self, base_dir, format='default', batch_size=8,
        n_fft=510, win_length=None, hop_length=128, num_frames=256, window='hann',
        num_workers=4, dummy=False, spec_factor=0.15, spec_abs_exponent=0.5,
        rir=False, rir_prob=0.5, no_reverb_as_target=False,
        gpu=True, normalize='noisy', transform_type="exponent", **kwargs
    ):
        super().__init__()
        self.base_dir = base_dir
        self.format = format
        self.batch_size = batch_size
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.num_frames = num_frames
        win_len = win_length if win_length else n_fft
        self.window = get_window(window, win_len) 
        self.windows = {}
        self.num_workers = num_workers
        self.dummy = dummy
        self.spec_factor = spec_factor
        self.spec_abs_exponent = spec_abs_exponent
        self.gpu = gpu
        self.normalize = normalize
        self.transform_type = transform_type
        self.rir = rir
        self.rir_prob = rir_prob
        self.no_reverb_as_target = no_reverb_as_target
        self.kwargs = kwargs

    def setup(self, stage=None):
        specs_kwargs = dict(
            stft_kwargs=self.stft_kwargs, num_frames=self.num_frames,
            spec_transform=self.spec_fwd, **self.kwargs
        )
        if stage == 'fit' or stage is None:
            self.train_set = Specs(data_dir=self.base_dir, subset='train',
                dummy=self.dummy, shuffle_spec=True, format=self.format,
                rir=self.rir, rir_prob=self.rir_prob, no_reverb_as_target=self.no_reverb_as_target, 
                normalize=self.normalize, **specs_kwargs)
            self.valid_set = Specs(data_dir=self.base_dir, subset='valid',
                dummy=self.dummy, shuffle_spec=False, format=self.format,
                normalize=self.normalize, **specs_kwargs)
        if stage == 'test' or stage is None:
            self.test_set = Specs(data_dir=self.base_dir, subset='test',
                dummy=self.dummy, shuffle_spec=False, format=self.format,
                normalize=self.normalize, **specs_kwargs)

    def spec_fwd(self, spec):
        if self.transform_type == "exponent":
            if self.spec_abs_exponent != 1:
                # only do this calculation if spec_exponent != 1, otherwise it's quite a bit of wasted computation
                # and introduced numerical error
                e = self.spec_abs_exponent
                spec = spec.abs()**e * torch.exp(1j * spec.angle())
            spec = spec * self.spec_factor
        elif self.transform_type == 'complex_exponent':
            if self.spec_abs_exponent != 1:
                e = self.spec_abs_exponent
                re, im = spec.real, spec.imag
                spec = re.sign()*(re.abs()**e) + 1j*im.sign()*(im.abs()**e)
            spec = spec * self.spec_factor
        elif self.transform_type == "log":
            spec = torch.log(1 + spec.abs()) * torch.exp(1j * spec.angle())
            spec = spec * self.spec_factor
        elif self.transform_type == "none":
            spec = spec
        elif self.transform_type == 'scale':
            spec = spec/50
        return spec

    def spec_back(self, spec):
        if self.transform_type == "exponent":
            spec = spec / self.spec_factor
            if self.spec_abs_exponent != 1:
                e = self.spec_abs_exponent
                spec = spec.abs()**(1/e) * torch.exp(1j * spec.angle())
        elif self.transform_type == 'complex_exponent':
            spec = spec / self.spec_factor
            if self.spec_abs_exponent != 1:
                e = 1/self.spec_abs_exponent
                re, im = spec.real, spec.imag
                spec = re.sign()*(re.abs()**e) + 1j*im.sign()*(im.abs()**e)
        elif self.transform_type == "log":
            spec = spec / self.spec_factor
            spec = (torch.exp(spec.abs()) - 1) * torch.exp(1j * spec.angle())
        elif self.transform_type == "none":
            spec = spec
        elif self.transform_type == 'scale':
            spec = spec*50
        return spec

    @property
    def stft_kwargs(self):
        return {**self.istft_kwargs, "return_complex": True}

    @property
    def istft_kwargs(self):
        return dict(
            n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
            window=self.window, center=True
        )

    def _get_window(self, x):
        """
        Retrieve an appropriate window for the given tensor x, matching the device.
        Caches the retrieved windows so that only one window tensor will be allocated per device.
        """
        window = self.windows.get(x.device, None)
        if window is None:
            window = self.window.to(x.device)
            self.windows[x.device] = window
        return window

    def stft(self, sig):
        window = self._get_window(sig)
        return torch.stft(sig, **{**self.stft_kwargs, "window": window})

    def istft(self, spec, length=None):
        window = self._get_window(spec)
        return torch.istft(spec, **{**self.istft_kwargs, "window": window, "length": length})

    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=self.gpu, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_set, batch_size=1,
            num_workers=self.num_workers, pin_memory=self.gpu, shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set, batch_size=1,
            num_workers=self.num_workers, pin_memory=self.gpu, shuffle=False
        )
