"""
Creates dataset from directory of images after generation.

"""

import csv
import os
from tqdm import tqdm

DATABASE_FILE = "data.csv"
IMAGE_DIR = 'images'

data = [['font', 'character', 'ASCII', 'path']]

fontFolders = subfolders = [f.path for f in os.scandir(IMAGE_DIR) if f.is_dir()]
for folder in tqdm(fontFolders):
    fontName = folder.split('/')[-1]
    images = [item for item in os.listdir(folder) if os.path.isfile(os.path.join(folder, item))]
    for image in images:
        ascii = image.split('.')[0]
        character = chr(int(ascii))
        path = os.path.join(folder, image)
        data.append([fontName, character, ascii, path])

with open(DATABASE_FILE, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)