import os
import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

IMG_DIR = 'fontdata/images'
CSV_DATA = 'fontdata/data.csv'
IMG_SIZE = 28
DATA_RANGE = 500
BATCH_SIZE = 256
NUM_EPOCHS = 100
  
# Initializing the transform for the dataset 
transform = torchvision.transforms.Compose([ 
    torchvision.transforms.ToTensor(), 
    torchvision.transforms.Normalize((0.5), (0.5)) 
]) 
  
class FontDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform = None, range = None):
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.range = range

        if self.range:
            self.data_frame.truncate(after = range)
    
    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data_frame.iloc[idx, 2])
        image = Image.open(img_name)
        label = self.data_frame.iloc[idx, 1]
        
        if self.transform:
            image = self.transform(image)\

        return image, label

# defining transforms
transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(), 
    transforms.Normalize((0.5), (0.5)) 
])
  
# create and load dataset
font_dataset = FontDataset(csv_file = CSV_DATA, img_dir = IMG_DIR,
                            transform = transforms, range = DATA_RANGE)
train_loader = torch.utils.data.DataLoader( 
    font_dataset, batch_size=BATCH_SIZE) 
  
# Printing 25 random images from the training dataset 
# random_samples = np.random.randint( 
#     1, len(font_dataset), (25))

# for idx in range(random_samples.shape[0]): 
#     plt.subplot(5, 5, idx + 1) 
#     plt.imshow(font_dataset[idx][0][0].numpy(), cmap='gray') 
#     plt.title(font_dataset[idx][1]) 
#     plt.axis('off') 

# plt.tight_layout() 
# plt.savefig('test_data.png')

class DeepAutoencoder(torch.nn.Module): 
    def __init__(self): 
        super().__init__()         
        self.encoder = torch.nn.Sequential( 
            torch.nn.Linear(IMG_SIZE * IMG_SIZE, 256), 
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
            torch.nn.Linear(256, IMG_SIZE * IMG_SIZE), 
            torch.nn.Sigmoid() 
        ) 
  
    def forward(self, x): 
        encoded = self.encoder(x) 
        decoded = self.decoder(encoded) 
        return decoded 

# Instantiating the model and hyperparameters 
model = DeepAutoencoder() 
criterion = torch.nn.MSELoss() 
num_epochs = NUM_EPOCHS
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# List that will store the training loss 
train_loss = [] 

# Dictionary that will store the 
# different images and outputs for  
# various epochs 
outputs = {}

batch_size = len(train_loader)

# Transfer to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)

# Training loop starts 
for epoch in tqdm(range(num_epochs)): 
        
    # Initializing variable for storing loss
    running_loss = 0
      
    # Iterating over the training dataset 
    for batch in train_loader: 
        
        # Loading image(s) and 
        # reshaping it into a 1-d vector 
        img, _ = batch # are we using the labels?
        img = img.reshape(-1, IMG_SIZE*IMG_SIZE)

        # put data on cuda
        img = img.to(device)
          
        # Generating output 
        out = model(img) 
          
        # Calculating loss 
        loss = criterion(out, img) 
          
        # Updating weights according 
        # to the calculated loss 
        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step() 
          
        # Incrementing loss 
        running_loss += loss.item() 
      
    # Averaging out loss over entire batch 
    running_loss /= batch_size 
    train_loss.append(running_loss) 
      
    # Storing useful images and 
    # reconstructed outputs for the last batch 
    outputs[epoch+1] = {'img': img, 'out': out} 

# Plotting the training loss 
plt.plot(range(1,num_epochs+1),train_loss) 
plt.xlabel("Number of epochs") 
plt.ylabel("Training Loss") 
plt.savefig(loss.png)