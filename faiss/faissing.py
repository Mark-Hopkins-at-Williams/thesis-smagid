import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
import os
import faiss
from tqdm import tqdm
import numpy as np

IMG_SIZE = 128
CSV_DATA = '/mnt/storage/smagid/thesis-smagid/fontdata/data.csv'
IMG_DIR = '/mnt/storage/smagid/thesis-smagid/fontdata/images'
MODEL_FILE = 'model_weights.pth'
OUTPUT_FILE = 'output.txt'

class FontDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform = None):
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data_frame.iloc[idx, 2])
        image = Image.open(img_name)
        label = self.data_frame.iloc[idx, 1]
        font = self.data_frame.iloc[idx, 0]
        
        if self.transform:
            image = self.transform(image)

        return image, font, label
    
class DeepAutoencoder(torch.nn.Module): 
    def __init__(self): 
        super().__init__()         
        self.encoder = torch.nn.Sequential( 
            torch.nn.Linear(IMG_SIZE*IMG_SIZE, 6272), 
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
            torch.nn.Linear(10, 64), 
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
  
    def forward(self, x): 
        encoded = self.encoder(x) 
        decoded = self.decoder(encoded) 
        return decoded 
    
if __name__ == "__main__":
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
        reshaped = image.reshape(IMG_SIZE*IMG_SIZE)
        encoding = encoder(reshaped)
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
    d = vectors.shape[1]
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    index = faiss.IndexFlatL2(d)
    index.add(vectors)

    distances, indices = index.search(vectors, 2) # second closest to exclude self

    nearest_neighbors = {}

    for i, font in enumerate(fonts):
        nearest_index = indices[i, 1]
        nearest_font = fonts[nearest_index]
        nearest_distance = distances[i, 1]
        nearest_neighbors[font] = (nearest_font, nearest_distance)

    with open(OUTPUT_FILE, "w") as f:
        for i, font in enumerate(fonts):
            nearest_index = indices[i, 1] # second closest neighbor
            nearest_font = fonts[nearest_index]
            nearest_distance = distances[i, 1]
            f.write(f"Nearest neighbor of '{font}' is '{nearest_font}' with distance {nearest_distance:.2f}\n")