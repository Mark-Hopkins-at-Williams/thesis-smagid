import sys
import cv2
import numpy as np
import torch
from torch.autograd import Variable

def print_bar(curr, maxval):
    barsize = min(maxval, 50)
    if curr % (maxval / barsize) == 0:
        numblocks = curr / (maxval / barsize)
        bar = "[" + "=" * numblocks + ">" + " " * (barsize + 1 - numblocks) + "]\r"
        sys.stdout.write(bar)
        sys.stdout.flush()

def log_factorial(k):
    approx = k * torch.log(k) - k + torch.log(k * (1 + 4 * k * (1 + 2 * k))) / 6. + np.log(np.pi) / 2.
    approx[k == 0] = 0
    return approx

def poisson_logpmf(k, mu):
    return k * torch.log(mu) - mu - log_factorial(k)

def poisson_pmf(k, mu):
    return torch.exp(poisson_logpmf(k, mu))

def gaussian_reparam(mu, logvar, z_size, dtype):
    eps = Variable(torch.randn(z_size).type(dtype))
    z = mu + eps * torch.exp(0.5 * logvar) # log(var) = log(stdev^2) so 1/2 needed
    return z

def kl_divergence(mu, logvar, z_size):
    var = torch.exp(logvar)
    kl = 0.5 * (var - 1 - logvar + torch.pow(mu, 2))
    return kl

def prep_write(x):
    x = torch.clamp(x, 0, 1)
    x = torch.cat([y for y in x.view(26, 64, 64)]).detach().cpu().numpy()
    x = np.floor(255 * x).astype(np.uint8)
    return x

def binarize(datum):
    binarized = datum.clone()
    threshold = 255
    binarized[binarized < threshold] = 0
    binarized[binarized >= threshold] = 1
    return binarized

def smart_binarize(datum):
    datumnp = np.floor(255 * datum.detach().cpu().numpy()).astype(np.uint8)
    rounded = cv2.adaptiveThreshold(datumnp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 2)
    #rounded = cv2.threshold(datumnp,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    return (rounded / 255.).astype(np.float32)

def build_charmap():
    infile = open("char_counts.txt")
    chars = {}
    for line in infile:
        tup,count = line.strip().split('\t')
        if int(count) < 100:
            break
        name = tup.split("'")[1]
        chars[name] = len(chars)
    return chars
