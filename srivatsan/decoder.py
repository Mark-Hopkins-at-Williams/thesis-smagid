import random
import sys

import numpy as np
import torch
from torch.nn import Parameter, ModuleList
from torch.autograd import Variable

from dct import apply_matrix, dct_matrix
from enums import *
from model import Model
from set_paths import paths
from utils import poisson_logpmf, prep_write, binarize

class Decoder(Model):

    def initialize(self):
        if not self.config.loss == Loss.bernoulli:
            self.dct = dct_matrix(64).type(self.config.float)
        if self.config.decoder_type == Arch.linear:
            self.font_dec_lin = torch.nn.Sequential(
                                    torch.nn.Linear(self.config.z_size * 2, self.config.hidden_size),
                                    torch.nn.ReLU(inplace=True),
                                    torch.nn.Linear(self.config.hidden_size, self.config.hidden_size),
                                    torch.nn.ReLU(inplace=True),
                                    torch.nn.Linear(self.config.hidden_size, self.config.hidden_size),
                                    torch.nn.ReLU(inplace=True),
                                    torch.nn.Linear(self.config.hidden_size, self.config.hidden_size),
                                    torch.nn.ReLU(inplace=True),
                                    torch.nn.Linear(self.config.hidden_size, self.config.hidden_size),
                                    torch.nn.ReLU(inplace=True),
                                    torch.nn.Linear(self.config.hidden_size, 64 * 64),
                                ).type(self.config.float)
            if not self.config.binarize:
                self.char_dec_lin[-2].bias.data += 255
        else:
            self.char_dec_resize = torch.nn.Sequential(
                                    torch.nn.Linear(self.config.z_size * 2, self.config.hidden_size),
                                    torch.nn.ReLU(inplace=True),
                                    torch.nn.Linear(self.config.hidden_size, 256 * 8 * 8),
                                    torch.nn.ReLU(inplace=True),
                                   ).type(self.config.float)
            if self.config.dynamic_params:
                self.weight0 = torch.nn.Sequential(
                                        torch.nn.Linear(self.config.z_size, self.config.hidden_size),
                                        torch.nn.ReLU(inplace=True),
                                        torch.nn.Linear(self.config.hidden_size, 256 * 256 * 5 * 5),
                                       ).type(self.config.float)
                self.bias0 = torch.nn.Sequential(
                                        torch.nn.Linear(self.config.z_size, self.config.hidden_size),
                                        torch.nn.ReLU(inplace=True),
                                        torch.nn.Linear(self.config.hidden_size, 256),
                                       ).type(self.config.float)
                self.conv0 = torch.nn.Sequential(
                                        torch.nn.Conv2d(256, 256, 5, padding=2),
                                        torch.nn.InstanceNorm2d(256),
                                        torch.nn.ReLU(inplace=True),
                                        torch.nn.Conv2d(256, 256, 5, padding=2),
                                        torch.nn.InstanceNorm2d(256),
                                        torch.nn.ReLU(inplace=True)
                                       ).type(self.config.float)
                self.weight1 = torch.nn.Sequential(
                                        torch.nn.Linear(self.config.z_size, self.config.hidden_size),
                                        torch.nn.ReLU(inplace=True),
                                        torch.nn.Linear(self.config.hidden_size, 256 * 128 * 5 * 5),
                                       ).type(self.config.float)
                self.bias1 = torch.nn.Sequential(
                                        torch.nn.Linear(self.config.z_size, self.config.hidden_size),
                                        torch.nn.ReLU(inplace=True),
                                        torch.nn.Linear(self.config.hidden_size, 128),
                                       ).type(self.config.float)
                self.conv1 = torch.nn.Sequential(
                                        torch.nn.Conv2d(128, 128, 5, padding=2),
                                        torch.nn.InstanceNorm2d(128),
                                        torch.nn.ReLU(inplace=True),
                                        torch.nn.Conv2d(128, 128, 5, padding=2),
                                        torch.nn.InstanceNorm2d(128),
                                        torch.nn.ReLU(inplace=True)
                                       ).type(self.config.float)
                self.weight2 = torch.nn.Sequential(
                                        torch.nn.Linear(self.config.z_size, self.config.hidden_size),
                                        torch.nn.ReLU(inplace=True),
                                        torch.nn.Linear(self.config.hidden_size, 128 * 64 * 5 * 5),
                                       ).type(self.config.float)
                self.bias2 = torch.nn.Sequential(
                                        torch.nn.Linear(self.config.z_size, self.config.hidden_size),
                                        torch.nn.ReLU(inplace=True),
                                        torch.nn.Linear(self.config.hidden_size, 64),
                                       ).type(self.config.float)
                self.conv2 = torch.nn.Sequential(
                                        torch.nn.Conv2d(64, 64, 5, padding=2),
                                        torch.nn.InstanceNorm2d(64),
                                        torch.nn.ReLU(inplace=True),
                                        torch.nn.Conv2d(64, 64, 5, padding=2),
                                        torch.nn.InstanceNorm2d(64),
                                        torch.nn.ReLU(inplace=True)
                                       ).type(self.config.float)
                self.weight3 = torch.nn.Sequential(
                                        torch.nn.Linear(self.config.z_size, self.config.hidden_size),
                                        torch.nn.ReLU(inplace=True),
                                        torch.nn.Linear(self.config.hidden_size, 64 * 32 * 5 * 5),
                                       ).type(self.config.float)
                self.bias3 = torch.nn.Sequential(
                                        torch.nn.Linear(self.config.z_size, self.config.hidden_size),
                                        torch.nn.ReLU(inplace=True),
                                        torch.nn.Linear(self.config.hidden_size, 32),
                                       ).type(self.config.float)
                self.conv3 = torch.nn.Sequential(
                                        torch.nn.Conv2d(32, 32, 5, padding=2),
                                        torch.nn.InstanceNorm2d(32),
                                        torch.nn.ReLU(inplace=True),
                                        torch.nn.Conv2d(32, 1, 5, padding=2),
                                       ).type(self.config.float)
            else:
                self.font_dec_conv = torch.nn.Sequential(
                                        torch.nn.ConvTranspose2d(256, 256, 5, stride=2, padding=2, output_padding=1),
                                        torch.nn.ReLU(inplace=True),
                                        torch.nn.ConvTranspose2d(256, 128, 5, stride=2, padding=2, output_padding=1),
                                        torch.nn.ReLU(inplace=True),
                                        torch.nn.ConvTranspose2d(128, 32, 5, stride=2, padding=2, output_padding=1),
                                        torch.nn.ReLU(inplace=True),
                                        torch.nn.ConvTranspose2d(32, 1, 5, stride=1, padding=2, output_padding=0),
                                     ).type(self.config.float)
        self.pixel_rnn = torch.nn.LSTM(self.config.z_size * 2, self.config.hidden_size, batch_first=True).type(self.config.float)
        self.pixel_reshape = torch.nn.Linear(self.config.hidden_size, 1).type(self.config.float)

    def forward(self, z_hat, y):
        # pass through y (character embeddings) with z (font style)
        if self.config.decoder_type == Arch.conv and self.config.dynamic_params:
            embed_cat = torch.cat([y, 0 * z_hat.view(1, -1).repeat(26, 1)], 1)
        else:
            embed_cat = torch.cat([y, z_hat.view(1, -1).repeat(26, 1)], 1)
        if self.config.decoder_type == Arch.linear:
            output = self.font_dec_lin(embed_cat)
        else:
            # project embedding to 256x8x8
            output = self.char_dec_resize(embed_cat).view(26, 256, 8, 8)
            if self.config.dynamic_params:
                # compute the parameters for convolutions
                weight0 = self.weight0(z_hat).view(256, 256, 5, 5)
                bias0 = self.bias0(z_hat).view(256)
                weight1 = self.weight1(z_hat).view(256, 128, 5, 5)
                bias1 = self.bias1(z_hat).view(128)
                weight2 = self.weight2(z_hat).view(128, 64, 5, 5)
                bias2 = self.bias2(z_hat).view(64)
                weight3 = self.weight3(z_hat).view(64, 32, 5, 5)
                bias3 = self.bias3(z_hat).view(32)
                # perform convolutions
                h0 = torch.nn.functional.conv_transpose2d(output, weight0, bias=bias0, stride=2, padding=2, output_padding=1)
                h0 = torch.nn.functional.instance_norm(h0)
                a0 = torch.nn.functional.relu(h0, inplace=True)
                a0 = self.conv0(a0)
                h1 = torch.nn.functional.conv_transpose2d(a0, weight1, bias=bias1, stride=2, padding=2, output_padding=1)
                h1 = torch.nn.functional.instance_norm(h1)
                a1 = torch.nn.functional.relu(h1, inplace=True)
                a1 = self.conv1(a1)
                h2 = torch.nn.functional.conv_transpose2d(a1, weight2, bias=bias2, stride=2, padding=2, output_padding=1)
                h2 = torch.nn.functional.instance_norm(h2)
                a2 = torch.nn.functional.relu(h2, inplace=True)
                a2 = self.conv2(a2)
                h3 = torch.nn.functional.conv_transpose2d(a2, weight3, bias=bias3, stride=1, padding=2, output_padding=0)
                h3 = torch.nn.functional.instance_norm(h3)
                a3 = torch.nn.functional.relu(h3, inplace=True)
                output = self.conv3(a3)
            else:
                output = self.font_dec_conv(output)
        '''
        # run a PixelRNN to clean up output
        output = torch.nn.functional.relu(output, inplace=True)
        output, (hn,cn) = self.pixel_rnn(torch.cat([z_hat.view(1, 1, -1).repeat(26, 64 * 64, 1), y.view(26, 1, -1).repeat(1, 64 * 64, 1)], 2))
        output = self.pixel_reshape(output)
        '''
        if self.config.dynamic_params:
            output /= 10. # scale down so activations not saturated at init
        if self.config.loss == Loss.bernoulli:
            output = torch.sigmoid(output)
            output = torch.clamp(output, 1e-6, 1-1e-6)
        elif self.config.binarize:
            pass
        else:
            output = torch.nn.functional.softplus(output)
        return output.view(26, 64 * 64)

    def score(self, datum, datum_hat):
        """Computes loss of reconstructed font data. Except in the case
        of bernoulli loss, a 2-Dimensional Discrete Cosine Transform
        (2D DCT-II) is applied to better compute loss based on edges."""
        if self.config.binarize:
            if self.config.loss == Loss.bernoulli:
                datum = binarize(datum)
                return torch.log(datum_hat)[datum == 1].sum() + torch.log(1 - datum_hat)[datum == 0].sum()
            else:
                datum_dct = apply_matrix(self.dct, datum.view(26, 64, 64) / 255.)
                datum_hat_dct = apply_matrix(self.dct, datum_hat.view(26, 64, 64))
                if self.config.loss == Loss.dct_l2:
                    return -((datum_dct - datum_hat_dct) ** 2).sum()
                elif self.config.loss == Loss.dct_l1:
                    return -torch.abs(datum_dct - datum_hat_dct).sum()
                elif self.config.loss == Loss.dct_cauchy:
                    sigma = 0.001
                    return -10000. * torch.log(1 + 0.5 * ((datum_dct - datum_hat_dct) ** 2).sum() / (sigma ** 2))
        else:
            pmf = poisson_logpmf(datum, datum_hat)
            return pmf.sum()

    def sample(self, y):
        """Passes a random sample of z-sized data through decoder."""
        z = Variable(torch.randn(self.config.z_size).type(self.config.float))
        return prep_write(self.forward(z, y))
