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
import plotly.express as px

IMG_SIZE = 128
CSV_DATA = '/mnt/storage/smagid/thesis-smagid/fontdata/data.csv'
IMG_DIR = '/mnt/storage/smagid/thesis-smagid/fontdata/images'
MODEL_FILE = 'new-weights.pth'
OUTPUT_FILE = 'output.txt'
LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

class FontDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform = None):
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        all_fonts = self.data_frame['font'].unique()
        df = self.data_frame
        self.fonts = []
        self.letters = list(LETTERS)

        try:
            with open('alphabetic_fonts.txt') as reader:
                for line in reader:
                    self.fonts.append(line.strip())
        except FileNotFoundError:        
            for font in tqdm(all_fonts):
                sub_df = df[(df['font'] == font) & (df['character'].isin(self.letters))] 
                if len(sub_df) == 52:                
                    self.fonts.append(font)
            with open('alphabetic_fonts.txt', 'w') as writer:
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
    
class DeepAutoencoder(torch.nn.Module): 
    def __init__(self): 
        super().__init__()         
        self.letter_embedding_dim = 10
        self.encoder = torch.nn.Sequential( 
            torch.nn.Linear(IMG_SIZE*IMG_SIZE + self.letter_embedding_dim, 6272), 
            torch.nn.ReLU(),
            torch.nn.Linear(6272, 3136), 
            torch.nn.ReLU(),
            torch.nn.Linear(3136, 1568), 
            torch.nn.ReLU(),
            torch.nn.Linear(1568, 784), 
            torch.nn.ReLU(), 
            torch.nn.Linear(784, 256), 
            torch.nn.ReLU(), 
            torch.nn.Linear(256, 128), 
            torch.nn.ReLU(), 
            torch.nn.Linear(128, 64), 
            torch.nn.ReLU(), 
            torch.nn.Linear(64, 10) 
        ) 
          
        self.decoder = torch.nn.Sequential( 
            torch.nn.Linear(10+self.letter_embedding_dim, 64), 
            torch.nn.ReLU(), 
            torch.nn.Linear(64, 128), 
            torch.nn.ReLU(), 
            torch.nn.Linear(128, 256), 
            torch.nn.ReLU(), 
            torch.nn.Linear(256, 784),
            torch.nn.ReLU(), 
            torch.nn.Linear(784, 1568),
            torch.nn.ReLU(), 
            torch.nn.Linear(1568, 3136),
            torch.nn.ReLU(), 
            torch.nn.Linear(3136, 6272),
            torch.nn.ReLU(), 
            torch.nn.Linear(6272, IMG_SIZE*IMG_SIZE), 
            torch.nn.Sigmoid() 
        ) 
        self.letter1_embedding = Embedding(num_embeddings=52, embedding_dim=10)
        self.letter2_embedding = Embedding(num_embeddings=52, embedding_dim=10)
        
  
    def forward(self, letter1_img, letter1_label, letter2_label): 
        try:
            letter1_embed = self.letter1_embedding(letter1_label) 
        except Exception:
            print(letter1_label.device)
            print(self.letter1_embedding.weight.device)
            
        letter2_embed = self.letter2_embedding(letter2_label)       
        encoded = self.encoder(torch.cat([letter1_img, letter1_embed], dim=1))        
        decoded = self.decoder(torch.cat([encoded, letter2_embed], dim=1)) 
        return decoded 
    
if __name__ == "__main__":  
    avg_file = "averages.npy"
    if os.path.exists(avg_file):
        vectors = np.load(avg_file)
    else:
        transformations = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            torchvision.transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(), 
            transforms.Normalize((0.5), (0.5)),
        ])

        dataset = FontDataset(csv_file = CSV_DATA, img_dir = IMG_DIR,
                                    transform = transformations)
        
        model = DeepAutoencoder()
        model.load_state_dict(torch.load(MODEL_FILE, weights_only=True))
        model.eval()
        encoder = model.encoder
        
        embeddings = {}

        for i in tqdm(range(len(dataset))):
            symbol = dataset[i]
            font = symbol[1]
            image = symbol[0]
            letter = symbol[2]
            reshaped = image.reshape(IMG_SIZE*IMG_SIZE)
            letter_embedding = model.letter1_embedding(letter) 
            encoding = encoder(torch.cat([reshaped, letter_embedding], dim=0))
            if font not in embeddings:
                embeddings[font] = [encoding]
            else:
                embeddings[font].append(encoding)

        averages = {}

        for font in embeddings:
            vectors = torch.stack(embeddings[font])
            average = vectors.mean(dim=0)
            averages[font] = average
        fonts = list(averages.keys())
        vectors = torch.stack(list(averages.values())).detach().numpy().astype('float32')
        np.save("averages.npy", vectors)

    tsne_file = "tsne-out.npy"
    if os.path.exists(tsne_file):
        reduced_vectors = np.load(tsne_file)
    else:
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        reduced_vectors = tsne.fit_transform(vectors)
        np.save(tsne_file, reduced_vectors)

    with open('alphabetic_fonts.txt', 'r') as file:
        labels = [line.strip() for line in file]

    fig = px.scatter(
        x=reduced_vectors[:, 0],
        y=reduced_vectors[:, 1],
        hover_name=labels,
        title="TypeFace Space T-SNE Map"
    )

    fig.update_layout(
    xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, title=''),
    yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, title=''),
    plot_bgcolor='#f2f4fa'
)

    fig.show()

