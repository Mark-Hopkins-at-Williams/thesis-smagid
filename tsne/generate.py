"""
Generates data for an interactive T-SNE plot of nn-font vector averages.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import Embedding, Parameter
import torchvision
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
import os
import faiss
from tqdm import tqdm
import numpy as np
from sklearn.manifold import TSNE
from fontdataset import FontDataset # local import
from styletransfer import StyleTransfer # local import

CSV_DATA = '/mnt/storage/smagid/thesis-smagid/fontdata/data.csv'
IMG_DATA = '/mnt/storage/smagid/thesis-smagid/fontdata/images'
MODEL_FILE = # fill this in
IMG_SIZE = 64
OUTPUT_FILE = 'tnse.npy'

if __name__ == "__main__":
    # dataset initialization
    dataset = FontDataset(csv_file = CSV_DATA, img_dir = IMG_DATA, img_size = IMG_SIZE)
    # model initialization
    model = StyleTransfer(img_size = IMG_SIZE)
    model.load_state_dict(torch.load(MODEL_FILE, weights_only=True))
    model.eval()
    # only generating encodings
    encoder = model.encoder
    # populate dictionary of encodings by font
    encodings = {}
    for i in tqdm(range(len(dataset))):
        (image, font, letter) = dataset[i]
        reshaped = image.reshape(IMG_SIZE*IMG_SIZE)
        letter_embedding = model.letter1_embedding(letter)
        encoding = encoder(torch.cat([reshaped, letter_embedding], dim=0))
        if font not in encodings:
            encodings[font] = [encoding]
        else:
            encodings[font].append(encoding)
    # take averages by font
    averages = {}
    for font in encodings:
        vectors = torch.stack(encodings[font])
        average = vectors.mean(dim=0)
        averages[font] = average
    
    # can we keep font names when we make the numpy array?
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced_vectors = tsne.fit_transform(vectors)
    np.save(OUTPUT_FILE, reduced_vectors)