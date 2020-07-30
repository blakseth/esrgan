"""
metrics.py

Written by Eirik Vesterkjær, 2019.
Modified by Sindre Stenen Blakseth, 2020.

Implements metrics to be used for evaluating the performance of ESRGAN.
"""

#----------------------------------------------------------------------------
# Package imports
import argparse
import math
import numpy as np
import torch

from metrics.LPIPS.models import PerceptualLoss

#----------------------------------------------------------------------------
# PSNR metric

def calculate_psnr(hr: np.ndarray, sr: np.ndarray) -> float:
    w, h, ch = hr.shape[0], hr.shape[1], hr.shape[2]

    sr = sr.reshape(w * h * ch)
    hr = hr.reshape(w * h * ch)

    MSE  = np.square( (hr - sr) ).sum(axis=0) / (w * h * ch)
    MSE  = MSE.item()
    if np.max(hr)**2 > 1.0:
        print("WARNING: PSNR input data is not scaled as expected.")
    R_sq = 1.0 #np.max(hr)**2 #255.0*255.0 # R is max fluctuation, and data is cv2 img: int [0, 255] -> R² = 255²
    eps  = 1e-8          # PSNR is usually ~< 50 so this should not impact the result much
    if R_sq > 0:
        psnr = 10 * math.log10(R_sq / (MSE + eps))
        #print("PSNR:", psnr)
        return psnr
    else:
        print("WARNING: PSNR was undefined, so MSE is printed instead.")
        return MSE

#----------------------------------------------------------------------------
# LPIPS metric

# TODO: Why use alexnet instead of the other possible nets?
lpips_model = PerceptualLoss(model='net-lin', net='alex', use_gpu=torch.cuda.is_available(), gpu_ids=[0])

def calculate_lpips(hr: torch.Tensor, sr: torch.Tensor):
    lpips = lpips_model.forward(hr, sr).item()
    #print("LPIPS:", lpips)
    return lpips

#----------------------------------------------------------------------------