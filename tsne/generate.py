"""
Generates data for an interactive T-SNE plot of nn-font vector averages.
"""

import torch
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
from sklearn.manifold import TSNE
from fontdataset import FontDataset # local import
from styletransfer import StyleTransfer # local import

CSV_DATA = '/mnt/storage/smagid/thesis-smagid/fontdata/data.csv'
IMG_DATA = '/mnt/storage/smagid/thesis-smagid/fontdata/images'
MODEL_FILE = '/mnt/storage/smagid/thesis-smagid/weights/styletransfer.pth'
IMG_SIZE = 128
AVERAGES_FILE = 'data/averages.pt'
TSNE_OUTPUT_FILE = 'data/tnse.csv'

if __name__ == "__main__":
    gpu = torch.device("cuda")
    if os.path.exists(AVERAGES_FILE):
        averages = torch.load(AVERAGES_FILE, map_location=gpu)
    else:
        # dataset initialization
        dataset = FontDataset(csv_file = CSV_DATA, img_dir = IMG_DATA, img_size = IMG_SIZE)
        # model initialization
        model = StyleTransfer(img_size = IMG_SIZE).to(gpu)
        model.load_state_dict(torch.load(MODEL_FILE, weights_only=True, map_location=gpu))
        model.eval()
        # only generating encodings
        encoder = model.encoder.to(gpu)
        # populate dictionary of encodings by font
        encodings = {}
        for i in tqdm(range(len(dataset))):
            (image, font, letter) = dataset[i]
            reshaped = image.reshape(IMG_SIZE*IMG_SIZE).to(gpu)
            letter_embedding = model.letter1_embedding(letter.to(gpu))
            encoding = encoder(torch.cat([reshaped, letter_embedding], dim=0))
            if font not in encodings:
                encodings[font] = [encoding]
            else:
                encodings[font].append(encoding)
        # take averages by font
        averages = {}
        for font in encodings:
            vectors = torch.stack(encodings[font]).to(gpu)
            average = vectors.mean(dim=0)
            averages[font] = average
        # save for reloading
        torch.save(averages, AVERAGES_FILE)
    # apply t-sne
    values = np.stack([value.cpu().detach().numpy() for value in averages.values()])
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_out = tsne.fit_transform(values)
    # put into a pandas dataframe and save data
    row_names = list(averages.keys())
    tsne_df = pd.DataFrame(tsne_out, index=row_names, columns=["tSNE-1", "tSNE-2"])
    tsne_df.to_csv(TSNE_OUTPUT_FILE, index=True)