# General imports
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import time
from datetime import datetime


from benchmark.PSNR import meanPSNR
#from benchmark.SSIM import meanSSIM
from benchmark.speed import speedMetric

def evaluation(in_path, out_path, verbose):
    begin_time = datetime.now()
    # Upscale

    label_path = os.path.join(in_path, 'labels/')
    in_path = os.path.join(in_path, 'without/')

    # Compute metrics
    if verbose:
        print("Computing PSNR (Y channel)", flush=True)
    mean_PSNRY, n2 = meanPSNR(label_path, in_path, color_mode='Y', verbose=verbose)
    if verbose:
        print("Computing PSNR (RGB channels)", flush=True)
    mean_PSNRRGB, _ = meanPSNR(label_path, in_path, color_mode='RGB', verbose=verbose)
    if verbose:
        print("Computing PSNR (YCbCr channels)", flush=True)
    mean_PSNRYCbCr, _ = meanPSNR(label_path, in_path, color_mode='YCbCr', verbose=verbose)
    #if verbose:
       # print("Computing SSIM", flush=True)
    #mean_SSIM, n3 = meanSSIM(label_path, in_path, verbose=verbose)

    benchmark_file = out_path + "Benchmark" + str(begin_time).replace(':', '-') + ".txt"
    with open(os.path.abspath(benchmark_file),'w') as file:
        file.write('---Performance Metrics info---' + os.linesep)
        file.write('Number of images used for PSNR: ' + str(n2) + os.linesep)
        file.write('Mean PSNR(Y): ' + str(mean_PSNRY) + ' dB' + os.linesep)
        file.write('Mean PSNR(YCbCr): ' + str(mean_PSNRYCbCr) + ' dB' + os.linesep)
        file.write('Mean PSNR(RGB): ' + str(mean_PSNRRGB) + ' dB' + os.linesep)
        file.write('Number of images used for SSIM: ' + str(n3) + os.linesep)
        #file.write('Mean SSIM(Y): ' + str(mean_SSIM) + os.linesep)

    if verbose:
        print("Benchmark saved to: " + benchmark_file, flush=True)
    return

if __name__ == "__main__":
    pass # ToDo
