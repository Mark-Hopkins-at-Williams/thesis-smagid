import os
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import torchvision
import torchvision.transforms as transforms
import torch
from torch.nn import Embedding, Parameter
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import sys

IMG_DIR = 'fontdata/images'
CSV_DATA = 'fontdata/data.csv'
IMG_SIZE = 128
BATCH_SIZE = 256
NUM_EPOCHS = 100
LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
GPU = "cuda"
TEST_IMG_DIR = 'new-new-images'
WEIGHTS_DIR = 'new-new-weights'
IMG_EVERY = 10
WEIGHTS_EVERY = 50

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
            for font in tqdm(all_fonts):
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


def image_progress(dev_set, model, step):
    """Saves an image of model progress on a sample of devset."""
    
    (letter1_image, letter1_label, letter2_label), letter2_image = dev_set
    original = letter1_image.clone()[0].squeeze()
    letter1_image = letter1_image.reshape(-1, IMG_SIZE*IMG_SIZE).to(GPU)
    letter1_label = letter1_label.to(GPU)
    letter2_label = letter2_label.to(GPU)
    model.eval()
    with torch.no_grad():
        out = model(letter1_image, letter1_label, letter2_label)
        out = out.to(GPU)
    trained = out.reshape(-1,IMG_SIZE,IMG_SIZE)[0]
    plt.clf()    
    plt.subplot(1, 3, 1)
    plt.imshow(original.cpu().detach().numpy(), cmap='gray')
    plt.title(f"original ({LETTERS[letter1_label[0].item()]})")
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(trained.cpu().detach().numpy(), cmap='gray')
    plt.title(f"trained ({LETTERS[letter2_label[0].item()]})")
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(letter2_image[0].squeeze().cpu().detach().numpy(), cmap='gray')
    plt.title(f"goal ({LETTERS[letter2_label[0].item()]})")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{TEST_IMG_DIR}/{step}.png')




if __name__ == "__main__":

    transformations = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        torchvision.transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(), 
        transforms.Normalize((0.5), (0.5)),
    ])
    
    # Loading data, reserving 10% for a devset
    full_dataset = FontPairDataset(csv_file = CSV_DATA, img_dir = IMG_DIR,
                                   transform = transformations)
    train_size = int(0.98 * len(full_dataset))
    dev_size = len(full_dataset) - train_size
    train_dataset, dev_dataset = random_split(full_dataset, [train_size, dev_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE) 
    print(f'train size: {len(train_dataset)}')
    print(f'dev size:   {len(dev_dataset)}')
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=True) 

    model = DeepAutoencoder() 
    criterion = torch.nn.MSELoss() 
    num_epochs = NUM_EPOCHS
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_loss = [] 

    num_batches = len(train_loader)

    print(len(train_loader))

    model.to(GPU)

    steps = 0 # Running count of steps
    
    for epoch in range(num_epochs): 
        running_loss = 0 # Running loss count
        for i, batch in tqdm(enumerate(train_loader)): 
             
            # Unpacking batch
            (letter1_image, letter1_label, letter2_label), letter2_image = batch
            letter1_image = letter1_image.reshape(-1, IMG_SIZE*IMG_SIZE).to(GPU)
            letter1_label = letter1_label.to(GPU)
            letter2_label = letter2_label.to(GPU)

            # Running batch through model and update weights
            out = model(letter1_image, letter1_label, letter2_label) 
            letter2_image = letter2_image.reshape(-1, IMG_SIZE*IMG_SIZE).to(GPU)
            loss = criterion(out, letter2_image)
            optimizer.zero_grad() 
            loss.backward() 
            optimizer.step() 

            # increment loss
            running_loss += loss.item() 

            # Recording performance metrics every IMAGE_EVERY updates
            if i % IMG_EVERY == 0:
                sample_data = next(iter(dev_loader))
                image_progress(sample_data, model, steps // IMG_EVERY)
                if i > 0:
                    loss_per_batch = running_loss / i if i > 0 else 0
                    print(f'Train loss: {loss_per_batch}')
                    sys.stdout.flush()

            # Saving weights every WEIGHTS_EVERY updates
            if i % WEIGHTS_EVERY == 0 and i != 0:
                model.eval()
                torch.save(model.state_dict(), f'{WEIGHTS_DIR}/weights-{i}.pth')
                model.train()

            # Keeping track of steps
            steps += 1
        
        # Averaging out loss over entire batch 
        running_loss /= num_batches
        train_loss.append(running_loss) 

    # Plotting the training loss 
    plt.plot(range(1,num_epochs+1),train_loss) 
    plt.xlabel("Number of epochs") 
    plt.ylabel("Training Loss") 
    plt.savefig('loss.png')