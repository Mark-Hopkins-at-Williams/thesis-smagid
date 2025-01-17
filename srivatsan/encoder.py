import random
import sys

import numpy as np
import torch
from torch.nn import Parameter
from torch.autograd import Variable

from enums import Metric, Optimizer, Arch
from model import Model
from utils import binarize

class Encoder(Model):
    
    def initialize(self):
        if self.config.encoder_type == Arch.linear:
            self.char_enc_lin = torch.nn.Sequential(
                                    torch.nn.Linear(64 * 64 + self.config.z_size, self.config.hidden_size),
                                    torch.nn.ReLU(inplace=True),
                                    torch.nn.Linear(self.config.hidden_size, self.config.hidden_size),
                                    torch.nn.ReLU(inplace=True),
                                    torch.nn.Linear(self.config.hidden_size, self.config.hidden_size),
                                    torch.nn.ReLU(inplace=True),
                                    torch.nn.Linear(self.config.hidden_size, 1024),
                                    torch.nn.ReLU(inplace=True)
                                ).type(self.config.float)
        else:
            self.char_enc_con = torch.nn.Sequential(
                                    torch.nn.Conv2d(1 + self.config.z_size, 64, 5, padding=2),
                                    torch.nn.MaxPool2d(2, stride=2),
                                    torch.nn.InstanceNorm2d(64),
                                    torch.nn.ReLU(inplace=True),
                                    torch.nn.Conv2d(64, 128, 5, padding=2),
                                    torch.nn.MaxPool2d(2, stride=2),
                                    torch.nn.InstanceNorm2d(128),
                                    torch.nn.ReLU(inplace=True),
                                    torch.nn.Conv2d(128, 256, 5, padding=2),
                                    torch.nn.MaxPool2d(2, stride=2),
                                    torch.nn.InstanceNorm2d(256),
                                    torch.nn.ReLU(inplace=True),
                                ).type(self.config.float)
            self.char_enc_resize = torch.nn.Sequential(
                                        torch.nn.Linear(256 * 8 * 8, 1024),
                                        torch.nn.ReLU(inplace=True)
                                    ).type(self.config.float)
        self.font_enc_lin = torch.nn.Sequential(
                                torch.nn.Linear(1024, self.config.hidden_size),
                                torch.nn.ReLU(inplace=True),
                                torch.nn.Linear(self.config.hidden_size, self.config.hidden_size),
                                torch.nn.ReLU(inplace=True),
                                torch.nn.Linear(self.config.hidden_size, self.config.hidden_size),
                                torch.nn.ReLU(inplace=True),
                                torch.nn.Linear(self.config.hidden_size, self.config.z_size * 2)
                            ).type(self.config.float)

    def forward(self, datum, y, mask=None):
        if self.config.binarize:
            datum = binarize(datum)
        if self.config.zca:
            datum = self.config.proc.preprocess(datum)
        if self.config.encoder_type == Arch.linear:
            datum = torch.cat([datum, y], 1) # take (26x4096) and cat on (26x32)
            char_out = self.char_enc_lin(datum)
        else:
            datum = torch.cat([datum.view(26, 1, 64, 64), y.view(26, -1, 1, 1).repeat(1, 1, 64, 64)], 1)
            char_out = self.char_enc_resize(self.char_enc_con(datum).view(26, -1))
        if mask is not None:
            char_out_avg = (char_out * mask).max(0)[0]
        else:
            char_out_avg = char_out.max(0)[0]
        font_out = self.font_enc_lin(char_out_avg)
        mu, logvar = font_out.split(self.config.z_size)
        return mu, logvar
