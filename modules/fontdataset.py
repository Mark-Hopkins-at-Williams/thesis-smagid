import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms

class FontDataset(Dataset):
    def __init__(self, data_dir, img_size):
        self.data_frame = pd.read_csv(os.path.join(data_dir, 'data.csv'))
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(), 
            transforms.Normalize((0.5), (0.5)),
        ])
        fonts = self.data_frame['font'].unique()
    
    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        item = self.data_frame.iloc[idx]
        letter = chr(item['ascii'])
        font = item['font']
        img_name = item['path']
        image = Image.open(img_name)
        
        if self.transform:
            image = self.transform(image)

        return image