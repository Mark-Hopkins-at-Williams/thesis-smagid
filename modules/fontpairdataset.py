import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image

LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

class FontPairDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform = None):
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.letters = list(LETTERS)
        all_fonts = self.data_frame['font'].unique()
        df = self.data_frame
        self.fonts = []
        
        # cache structure: {(font, letter): image}
        self.cache = dict()        
        
        try:
            with open('alphabetic_fonts.txt') as reader:
                for line in reader:
                    self.fonts.append(line.strip())
        except FileNotFoundError:        
            for font in all_fonts:
                sub_df = df[(df['font'] == font) & (df['character'].isin(self.letters))] 
                if len(sub_df) == 52:                
                    self.fonts.append(font)
            with open('alphabetic_fonts.txt', 'w') as writer:
                for font in self.fonts:
                    writer.write(f'{font}\n')
                

    def __len__(self):
        return (len(self.letters) ** 2) * len(self.fonts)

    def __getitem__(self, idx):
        which_letter2 = idx % len(self.letters)
        which_letter1 = (idx // len(self.letters)) % len(self.letters)
        which_font = (idx // (len(self.letters) ** 2)) % len(self.fonts)
        df = self.data_frame

        letter1 = df[(df['font'] == self.fonts[which_font]) & (df['character']==self.letters[which_letter1])]
        letter1_label = torch.tensor(self.letters.index(letter1.iloc[0]['character']))
        letter2 = df[(df['font'] == self.fonts[which_font]) & (df['character']==self.letters[which_letter2])]
        letter2_label = torch.tensor(self.letters.index(letter2.iloc[0]['character']))

        if (which_font, which_letter1) not in self.cache:
            letter1_img_name = os.path.join(self.img_dir, letter1.iloc[0]['path'])
            letter1_image = Image.open(letter1_img_name)
            self.cache[(which_font, which_letter1)] = letter1_image
        else:
            letter1_image = self.cache[(which_font, which_letter1)]

        if (which_font, which_letter2) not in self.cache:
            letter2_img_name = os.path.join(self.img_dir, letter2.iloc[0]['path'])
            letter2_image = Image.open(letter2_img_name)
            self.cache[(which_font, which_letter2)] = letter2_image
        else:
            letter2_image = self.cache[(which_font, which_letter2)]
            
        if self.transform:
            letter1_image = self.transform(letter1_image)
            letter2_image = self.transform(letter2_image)

        return (letter1_image, letter1_label, letter2_label), letter2_image