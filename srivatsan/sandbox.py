import torch
from font_vae import FontVAE
from arguments import parse_args
import os
from imageio import imread
import numpy as np
from utils import *

config = parse_args()
config.float = torch.cuda.FloatTensor
config.long = torch.cuda.LongTensor

model = FontVAE(config)

# load google fonts+
fonts = []
path = '/mnt/storage/smagid/Capitals64/gfonts+'
for filename in os.listdir(path):
    font_image = imread(path + '/' + filename)
    letters = np.split(font_image, 26, axis=1)
    letters = [letter.reshape([-1]).tolist() for letter in letters]
    fonts.append((filename, letters))
checkpoint_dict = torch.load('e350.pt')
model.load_state_dict(checkpoint_dict['state_dict'], strict=False)

encoder = model.encoder

zhats = dict()

for font in fonts:
    filename, datum = font
    datum = Variable(model.config.float(datum))
    mu, logvar = encoder.forward(datum, model.y, mask=None)
    zhat = gaussian_reparam(mu, logvar, model.config.z_size, model.config.float)
    zhats[filename] = zhat