import torch
from font_vae import FontVAE
from arguments import parse_args
import os
from imageio import imread
import numpy as np
from utils import *
import pandas as pd
from tqdm import tqdm

ATTRIBUTES_CSV = '../gfont-attributes/fonts-attributes.csv'
COMMON_WORDS = ["Bold", "Italic", "ExtraBold", "Thin", "SemiBold", "ExtraBold", "Regular", "Medium", "Light", "ExtraLight", "10pt", "Black"]

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

config = parse_args()
config.float = torch.cuda.FloatTensor
config.long = torch.cuda.LongTensor

model = FontVAE(config)

# load google fonts+
fonts = []
path = '/mnt/storage/smagid/fonts/png-grouped/google'
for filename in os.listdir(path):
    font_image = imread(path + '/' + filename)
    letters = np.split(font_image, 26, axis=1)
    letters = [letter.reshape([-1]).tolist() for letter in letters]
    fontName = filename.split('.')[0]
    fonts.append((fontName, letters))
checkpoint_dict = torch.load('e350.pt')
model.load_state_dict(checkpoint_dict['state_dict'], strict=False)

encoder = model.encoder

# attributes = pd.read_csv(ATTRIBUTES_CSV)

# # only do google fonts
# googleFonts = []
# for font in fonts:
#     adaptedFontName = ' '.join([word for word in font[0].split() if word not in COMMON_WORDS])
#     if adaptedFontName in attributes['font'].values:
#         # print(f'YES: {adaptedFontName}')
#         googleFonts.append((adaptedFontName, font[1]))
#     # else:
#     #     print(f'NO: {adaptedFontName}')

print(f"length is {len(fonts)}")

googleFontNames = [font[0] for font in fonts]

# catch duplicates:
for fontName in googleFontNames:
    if fonts.count(fontName) > 1:
        print(f"Noting duplicate: {fontName}")

zhats = pd.DataFrame({'font': googleFontNames})

for font in tqdm(fonts):
    # adaptedFontName = ' '.join([word for word in font[0].split() if word not in COMMON_WORDS])
    fontName = font[0]
    if fontName in googleFontNames: # redundant now
        filename, datum = font
        datum = Variable(model.config.float(datum))
        mu, logvar = encoder.forward(datum, model.y, mask=None)
        zhat = gaussian_reparam(mu, logvar, model.config.z_size, model.config.float)
        for i, value in enumerate(zhat.tolist()):
            zhats.loc[zhats['font'] == fontName, f'z{i}'] = value

zhats.to_csv("zhats.csv", index=False)