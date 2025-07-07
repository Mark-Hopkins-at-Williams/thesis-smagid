import os
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import sys

CSV_DATA = '/mnt/storage/smagid/fonts/png-individual/data.csv'
IMG_SIZE = 64
DATA_RANGE = None
BATCH_SIZE = 256
NUM_EPOCHS = 100

class FontDataset(Dataset):
    def __init__(self, csv_file, transform = None, range = None):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform
        self.range = range

        if self.range:
            self.data_frame.truncate(after = range)
    
    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_path = self.data_frame.iloc[idx, 4]
        image = Image.open(img_path)
        label = self.data_frame.iloc[idx, 2]
        
        if self.transform:
            image = self.transform(image)

        return image, label



class DeepAutoencoder(torch.nn.Module): 
    def __init__(self): 
        super().__init__()         
        self.encoder = torch.nn.Sequential( 
            torch.nn.Linear(4096, 2048), 
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 1024), 
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512), 
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256), 
            torch.nn.ReLU(), 
            torch.nn.Linear(256, 128), 
            torch.nn.ReLU(), 
            torch.nn.Linear(128, 64), 
            torch.nn.ReLU(), 
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(), 
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(), 
            torch.nn.Linear(16, 6)
        ) 
          
        self.decoder = torch.nn.Sequential( 
            torch.nn.Linear(6, 16), 
            torch.nn.ReLU(), 
            torch.nn.Linear(16, 32), 
            torch.nn.ReLU(), 
            torch.nn.Linear(32, 64), 
            torch.nn.ReLU(), 
            torch.nn.Linear(64, 128), 
            torch.nn.ReLU(), 
            torch.nn.Linear(128, 256), 
            torch.nn.ReLU(), 
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(), 
            torch.nn.Linear(512, 1024),
            torch.nn.ReLU(), 
            torch.nn.Linear(1024, 2048),
            torch.nn.ReLU(), 
            torch.nn.Linear(2048, 4096),
            torch.nn.Sigmoid() 
        ) 
  
    def forward(self, x): 
        encoded = self.encoder(x) 
        decoded = self.decoder(encoded) 
        return decoded 


def image_progress(dev_set, model, step):
    """Saves an image of model progress on a sample of devset."""
    sample = 5
    img, _ = dev_set
    img = img.reshape(-1, IMG_SIZE*IMG_SIZE)
    img = img.to("cuda")
    original = img.reshape(-1,IMG_SIZE,IMG_SIZE)[0]
    model.eval()
    with torch.no_grad():
        out = model(img)
        out = out.to("cuda")
    trained = out.reshape(-1,IMG_SIZE,IMG_SIZE)[0:sample]
    original = img.reshape(-1,IMG_SIZE,IMG_SIZE)[0:sample]
    plt.clf()
    for i in range(sample):
        plt.subplot(sample, 2, 2*i + 1)
        plt.imshow(original[i].cpu().detach().numpy(), cmap='gray')
        plt.title("original")
        plt.axis('off')
        plt.subplot(sample, 2, 2*i + 2)
        plt.imshow(trained[i].cpu().detach().numpy(), cmap='gray')
        plt.title("trained")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'images/{step}.png')




if __name__ == "__main__":

    transformations = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        torchvision.transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(), 
        transforms.Normalize((0.5), (0.5)),
    ])
    
    # Loading data, reserving 10% for a devset
    full_dataset = FontDataset(csv_file = CSV_DATA,
                                transform = transformations, range = DATA_RANGE)
    train_size = int(0.9 * len(full_dataset))
    dev_size = len(full_dataset) - train_size
    train_dataset, dev_dataset = random_split(full_dataset, [train_size, dev_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE) 
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE) 

    model = DeepAutoencoder() 
    criterion = torch.nn.MSELoss() 
    num_epochs = NUM_EPOCHS
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_loss = [] 

    num_batches = len(train_loader)

    # Transfering model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    sample_data = next(iter(dev_loader))
    record_every = 250

    steps = 0 # Running count of steps

    for epoch in range(num_epochs): 
        model.train()
        running_loss = 0 # Running loss count
        for i, batch in tqdm(enumerate(train_loader)): 
             
            # Unpacking batch
            img, _ = batch
            img = img.reshape(-1, IMG_SIZE*IMG_SIZE)
            img = img.to(device)

            # Running batch through model and update weights
            out = model(img) 
            loss = criterion(out, img)
            optimizer.zero_grad() 
            loss.backward() 
            optimizer.step() 

            # increment loss
            running_loss += loss.item() 

            # Recording performance metrics every record_every updates
            if i % record_every == 0:
                image_progress(sample_data, model, steps)
                if i > 0:
                    loss_per_batch = running_loss / i if i > 0 else 0
                    print(f'Train loss: {loss_per_batch}')
                    sys.stdout.flush()

            # Keeping track of steps
            steps += 1
        model.eval()
        torch.save(model.state_dict(), f'weights.pth')
        
        # Averaging out loss over entire batch 
        running_loss /= num_batches
        train_loss.append(running_loss) 

    # Plotting the training loss 
    plt.plot(range(1,num_epochs+1),train_loss) 
    plt.xlabel("Number of epochs") 
    plt.ylabel("Training Loss") 
    plt.savefig('loss.png')