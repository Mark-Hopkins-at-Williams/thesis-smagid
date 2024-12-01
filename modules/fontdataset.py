import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms

LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
ALPHA_FONTS_DIR = '/mnt/storage/smagid/thesis-smagid/fontdata/alphabetic_fonts.txt'

class FontDataset(Dataset):
    def __init__(self, csv_file, img_dir, img_size):
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(), 
            transforms.Normalize((0.5), (0.5)),
        ])
        df = self.data_frame
        all_fonts = df['font'].unique()
        self.fonts = []
        self.letters = list(LETTERS)

        try:
            with open(ALPHA_FONTS_DIR) as reader:
                for line in reader:
                    self.fonts.append(line.strip())
        except FileNotFoundError:        
            for font in tqdm(all_fonts):
                sub_df = df[(df['font'] == font) & (df['character'].isin(self.letters))] 
                if len(sub_df) == 52:                
                    self.fonts.append(font)
            with open(ALPHA_FONTS_DIR, 'w') as writer:
                for font in self.fonts:
                    writer.write(f'{font}\n')
    
    def __len__(self):
        return len(self.letters) * len(self.fonts)

    def __getitem__(self, idx):
        which_letter = idx % len(self.letters)
        which_font = (idx // len(self.letters)) % len(self.fonts)
        df = self.data_frame

        letter = df[(df['font'] == self.fonts[which_font]) & (df['character']==self.letters[which_letter])]
        img_name = os.path.join(self.img_dir, letter.iloc[0]['path'])
        image = Image.open(img_name)
        letter_label = torch.tensor(self.letters.index(letter.iloc[0]['character']))
        font = self.fonts[which_font]
        
        if self.transform:
            image = self.transform(image)

        return image, font, letter_label