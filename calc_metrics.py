from os.path import join 
from glob import glob
from argparse import ArgumentParser
from soundfile import read
from tqdm import tqdm
from pesq import pesq as pesq1
from sepm import composite
import pandas as pd
import numpy as np
import os

from pystoi import stoi

from sgmse.util.other import energy_ratios, mean_std

def pesq2(sr, x, x_method, band_mode):
    pesq_mos = pesq1(sr, x, x_method, band_mode)

    pesq_mos = 4.6607 - np.log((4.999 - pesq_mos)/(pesq_mos - 0.999))
    pesq_mos = pesq_mos / 1.4945
    return pesq_mos



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--test_dir", type=str, required=True, help='Directory containing the original test data (must have subdirectories clean/ and noisy/)')
    parser.add_argument("--enhanced_dir", type=str, required=True, help='Directory containing the enhanced data')
    parser.add_argument("--txt", action='store_true', help='Directory containing the enhanced data')
    parser.add_argument("--band_mode", type=str, default='wb', help='band mode "wb" or "nb" ')
    parser.add_argument("--no_extend", action='store_true', help='comptute stoi use extend or not" ')
    parser.add_argument("--pesq_mode", type=int, default=1, help='use pesq or pypesq ')
    parser.add_argument("--N", type=int, default=None, help='n-th step')
    parser.add_argument("--compare_to_real", action='store_true', help='compare the signal to the real target')
    args = parser.parse_args()

    test_dir = args.test_dir
    band_mode = args.band_mode
    enhanced_dir = args.enhanced_dir
    pesq = pesq1 if args.pesq_mode == 1 else pesq2
    extend = not args.no_extend
    compare_to_real = args.compare_to_real
    N_sample = args.N
    if args.txt:
        with open(test_dir, 'r') as f:
            clean_dir = f.readlines()
        with open(enhanced_dir, 'r') as f:
            noisy_dir = f.readlines()
    else:
        clean_dir = join(test_dir, "clean/")
        noisy_dir = join(test_dir, "noisy/")

    data = {"filename": [], "pesq": [], "estoi": [], "si_sdr": [], "si_sir": [],  "si_sar": [], "ssnr":[], "sig":[], "bak":[], "ovl":[]}
    sr = 16000

    # Evaluate standard metrics
    if not args.txt:
        noisy_files = sorted(glob('{}/*.wav'.format(noisy_dir)))
    else:
        noisy_files = noisy_dir
    i = 0 
    for  noisy_file in tqdm(noisy_files):
        if args.txt:
            x, _ = read(clean_dir[i].strip())
            y, _ = read(noisy_dir[i].strip())
            i += 1
            x_method = y
            filename = os.path.basename(noisy_file.strip())
        else:
            filename = noisy_file.split('/')[-1]
            x, _ = read(join(clean_dir, filename))
            y, _ = read(noisy_file)
            x_method, _ = read(join(enhanced_dir, filename))
        n = y - x 
        if compare_to_real:
            #x = x + (1 - np.exp(-1.5*(1 - 0.04)*N_sample/25 + 0.04))*n
            x_method = x + (1 - np.exp(-1.5*(25 - N_sample)/25))*n

        sdr, sir, sar = energy_ratios(x_method, x, n)
        ssnr, pesq_mos, sig, bak, ovl = composite(x, x_method, sr)
        data["filename"].append(filename)
        data["pesq"].append(pesq(sr, x, x_method, band_mode))
        #data["pesq"].append(pesq_mos)
        data["ssnr"].append(ssnr)
        data["estoi"].append(stoi(x, x_method, sr, extended=extend))
        data["si_sdr"].append(sdr)
        data["si_sir"].append(sir)
        data["si_sar"].append(sar)
        data["sig"].append(sig)
        data["bak"].append(bak)
        data["ovl"].append(ovl)


    # Save results as DataFrame    
    df = pd.DataFrame(data)

    # POLQA evaluation  -  requires POLQA license and server, uncomment at your own peril.
    # This is batch processed for speed reasons and thus runs outside the for loop.
    # if not basic:
    #     clean_files = sorted(glob('{}/*.wav'.format(clean_dir)))
    #     enhanced_files = sorted(glob('{}/*.wav'.format(enhanced_dir)))
    #     clean_audios = [read(clean_file)[0] for clean_file in clean_files]
    #     enhanced_audios = [read(enhanced_file)[0] for enhanced_file in enhanced_files]
    #     polqa_vals = polqa(clean_audios, enhanced_audios, 16000, save_to=None)
    #     polqa_vals = [val[1] for val in polqa_vals]
    #     # Add POLQA column to DataFrame
    #     df['polqa'] = polqa_vals

    # Print results
    print(enhanced_dir)
    #print("POLQA: {:.2f} ± {:.2f}".format(*mean_std(df["polqa"].to_numpy())))
    print("PESQ: {:.2f} ± {:.2f}".format(*mean_std(df["pesq"].to_numpy())))
    print("FewSNR: {:.2f} ± {:.2f}".format(*mean_std(df["ssnr"].to_numpy())))
    print("ESTOI: {:.4f} ± {:.4f}".format(*mean_std(df["estoi"].to_numpy())))
    print("SI-SDR: {:.1f} ± {:.1f}".format(*mean_std(df["si_sdr"].to_numpy())))
    print("SI-SIR: {:.1f} ± {:.1f}".format(*mean_std(df["si_sir"].to_numpy())))
    print("SI-SAR: {:.1f} ± {:.1f}".format(*mean_std(df["si_sar"].to_numpy())))
    print("Csig: {:.2f} ± {:.2f}".format(*mean_std(df["sig"].to_numpy())))
    print("Cbak: {:.2f} ± {:.2f}".format(*mean_std(df["bak"].to_numpy())))
    print("Covl: {:.2f} ± {:.2f}".format(*mean_std(df["ovl"].to_numpy())))

    # Save DataFrame as csv file
    ofile = 'compare_to_real_results.csv' if compare_to_real else '_results.csv' 
    if args.txt:
        result_dir = os.path.splitext(os.path.basename(enhanced_dir))[0] + ofile
    else:
        result_dir = join(enhanced_dir, ofile)
    df.to_csv(result_dir, index=False)
