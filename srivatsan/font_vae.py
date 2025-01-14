import random
import sys

from imageio import imwrite
import numpy as np
from skimage.measure import block_reduce
from scipy.spatial import distance_matrix
import torch
from torch.nn import Parameter
from torch.autograd import Variable

from decoder import Decoder
from encoder import Encoder
from enums import Metric, Optimizer
from model import Model
from set_paths import paths
from utils import *

class FontVAE(Model):
    def initialize(self):
        # define hyperprameters
        self.curr_epoch = -1
        self.hyperparams = {'reg' : self.config.regularization,
                            'dyn' : self.config.dynamic_params,
                            'z' : self.config.z_size,
                            'h' : self.config.hidden_size,
                            'beta' : self.config.beta,
                            'kl' : self.config.kl_threshold,
                            'loss' : self.config.loss,
                            'lr' : self.config.learn_rate}
        # initialize variables
        self.encoder = Encoder(self.config)
        self.decoder = Decoder(self.config) 
        self.y = Parameter(torch.randn(26, self.config.z_size).type(self.config.float))
        # set up optimizer
        self.params = list(self.parameters())
        if self.config.optim == Optimizer.adam:
            optimizer_type = torch.optim.Adam
        elif self.config.optim == Optimizer.sgd:
            optimizer_type = torch.optim.SGD 
        self.opt = optimizer_type(self.params, lr=self.config.learn_rate, weight_decay=self.config.regularization)

    def noisy_decode(self, mu, logvar):
        if self.training:
            z_hat = gaussian_reparam(mu, logvar, self.config.z_size, self.config.float)
            datum_hat = self.decoder.forward(z_hat, self.y)
        else:
            if self.config.sample_test == 0:
                datum_hat = self.decoder.forward(mu, self.y)
            else:
                samples = []
                for i in range(self.config.sample_test):
                    z_hat = gaussian_reparam(mu, logvar, self.config.z_size, self.config.float)
                    samples.append(self.decoder.forward(z_hat, self.y))
                datum_hat = sum(samples) / len(samples)
        return datum_hat

    def forward(self, datum, beta=1.0, filename=None):
        # computes the approximate marginal log likelihood of observation
        if self.training and filename is not None:
            mask = self.get_mask(filename)
        else:
            mask = None
        mu, logvar = self.encoder.forward(datum, self.y, mask=mask)
        kl = kl_divergence(mu, logvar, self.config.z_size)
        if self.training:
            kl = torch.max(kl, torch.zeros_like(kl) + self.config.kl_threshold).sum()
        else:
            kl = kl.sum()
        datum_hat = self.noisy_decode(mu, logvar)
        recon = self.decoder.score(datum, datum_hat)
        #print "Recon", recon
        #print "KL", beta * kl
        loss = recon - beta * kl
        # add regularization terms
        if self.training:
            reg = self.config.lamb * torch.abs(datum_hat).sum()
            loss -= reg # subtract because it gets flipped to compute NLL
        return loss

    def train_step(self, font):
        filename, datum = font
        datum = Variable(self.config.float(datum))
        # compute expected marginal negative log-likelihood
        obj = -self.forward(datum, beta=self.config.beta, filename=filename)
        # compute gradients and update
        self.opt.zero_grad()
        obj.backward()
        self.opt.step()

    def standardize_pixel_intensities(self):
        return True

    def reconstruct(self, datum, mask):
        mu, logvar = self.encoder.forward(datum, self.y, mask=mask)
        datum_hat = self.noisy_decode(mu, logvar)
        if self.config.binarize:
            datum_hat *= 255
        return datum_hat

    def visualize_reconstructions(self, corpus, prefix, max_fonts=None):
        if max_fonts is None:
            max_fonts = len(corpus)
        for i,font in enumerate(corpus[:max_fonts]):
            filename, datum = font
            datum = Variable(self.config.float(datum))
            mask = self.get_mask(filename)
            datum_hat = self.reconstruct(datum, mask)
            datum /= 255.
            datum_hat /= 255.
            # write reconstructions to file
            if self.config.output_binarization:
                datum_hat = self.config.float(smart_binarize(datum_hat))
            inter = datum * mask + datum_hat * (1 - mask)
            path = "{}/{}/{}/recon_{}_e{}_b{}_{}_{}".format(paths['images'], self.config.mode, self.config.model, prefix, self.curr_epoch, self.config.blanks, i, filename)
            imwrite(path, prep_write(inter))

    def get_mask(self, filename):
        mask = Variable(torch.ones(26, 1)).type(self.config.float)
        if self.training:
            # mask out each character with some probability
            charprob = 0.7
            for i in range(26):
                if random.random() <= charprob:
                    mask[i, 0] = 0
            # make sure at least one is observed
            if sum(mask) == 0:
                mask[random.choice(range(26)), 0] = 1
        else:
            if filename in self.config.test_dict:
                blank_ids = self.config.test_dict[filename][:self.config.blanks]
            else:
                blank_ids = range(26)
                random.shuffle(blank_ids)
                blank_ids = blank_ids[:self.config.blanks]
            mask[blank_ids] = 0
        return mask

    def sample_fonts(self):
        num_samples = 100
        for i in range(100):
            sample = self.decoder.sample(self.y)
            imwrite("{}/{}/{}/sample_e{}_{}.png".format(paths['images'], self.config.mode, self.config.model, self.curr_epoch, i), sample)
