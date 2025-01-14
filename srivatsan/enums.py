from enum import Enum
import torch

class Dataset(Enum):
    cap64 = 'Capitals64'

    def __str__(self):
        return self.value

class Mode(Enum):
    debug = 'debug'
    val = 'val'
    test = 'test'
    toy = 'toy'

    def __str__(self):
        return self.value

class Optimizer(Enum):
    adam = "Adam"
    sgd = "SGD"

    def __str__(self):
        return self.value

class Metric(Enum):
    nll_mf = 'nll_mf'
    mse = 'mse'
    ssim = 'ssim'

    def __str__(self):
        return self.value

class Model(Enum):
    font_vae = 'font_vae'
    nearest_neighbor = 'nearest_neighbor'

    def __str__(self):
        return self.value

class Arch(Enum):
    linear = 'linear'
    conv = 'conv'

    def __str__(self):
        return self.value

class Loss(Enum):
    bernoulli = 'bernoulli'
    dct_l2 = 'dct_l2'
    dct_l1 = 'dct_l1'
    dct_cauchy = 'dct_cauchy'
    dct_t = 'dct_t'

    def __str__(self):
        return self.value
