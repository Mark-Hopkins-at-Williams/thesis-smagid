import os
import sys
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from fontpairdataset import FontPairDataset
from fontresnet import FontResNet
from imageprogress import image_progress

IMG_DIR = '../fontdata/images'
CSV_DATA = '../fontdata/data.csv'
IMG_SIZE = 64
BATCH_SIZE = 256
NUM_EPOCHS = 100
GPU = "cuda"
TEST_IMG_DIR = 'images'
WEIGHTS_DIR = 'weights'
IMG_EVERY = 10
LOSS_EVERY = 10
WEIGHTS_EVERY = 500

if __name__ == "__main__":
    for directory in [TEST_IMG_DIR, WEIGHTS_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)

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

    model = FontResNet(image_size=64) 
    criterion = torch.nn.MSELoss() 
    num_epochs = NUM_EPOCHS
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_loss = [] 
    dev_loss = []

    num_batches = len(train_loader)

    model.to(GPU)

    steps = 0 # Running count of steps

    running_loss = 0
    
    for epoch in range(num_epochs): 
        for i, batch in tqdm(enumerate(train_loader)): 
             
            # Unpacking batch
            (letter1_image, letter1_label, letter2_label), letter2_image = batch
            letter1_image = letter1_image.to(GPU)
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
                image_progress(IMG_SIZE, sample_data, model, steps // IMG_EVERY)
                if i > 0:
                    loss_per_batch = running_loss / i if i > 0 else 0
                    print(f'Train loss: {loss_per_batch}')
                    sys.stdout.flush()

            # Saving weights every WEIGHTS_EVERY updates
            if i % WEIGHTS_EVERY == 0 and i != 0:
                model.eval()
                torch.save(model.state_dict(), f'{WEIGHTS_DIR}/weights-{i}.pth')
                model.train()

            if i % LOSS_EVERY == 0:
                # plot average training loss
                running_loss /= LOSS_EVERY
                train_loss.append(running_loss) 
                running_loss = 0
                plt.clf()
                plt.plot(range(1,LOSS_EVERY * len(train_loss)+1, LOSS_EVERY), train_loss) 
                plt.xlabel("Number of iterations") 
                plt.ylabel("Training Loss")
                plt.tight_layout() 
                plt.savefig('train_loss.png')
                # plot dev loss
                model.eval()
                with torch.no_grad():
                    sample_data = next(iter(dev_loader))
                    (s_letter1_image, s_letter1_label, s_letter2_label), s_letter2_image = sample_data
                    s_letter1_image = s_letter1_image.to(GPU)
                    s_letter1_label = s_letter1_label.to(GPU)
                    s_letter2_label = s_letter2_label.to(GPU)
                    out = model(s_letter1_image, s_letter1_label, s_letter2_label)
                    s_letter2_image = s_letter2_image.reshape(-1, IMG_SIZE*IMG_SIZE).to(GPU)
                    loss = criterion(out, s_letter2_image).cpu().detach().numpy()
                    dev_loss.append(loss)
                plt.clf()
                plt.plot(range(1,LOSS_EVERY * len(dev_loss)+1, LOSS_EVERY), dev_loss) 
                plt.xlabel("Number of iterations") 
                plt.ylabel("Dev Loss") 
                plt.tight_layout()
                plt.savefig('dev_loss.png')


            # Keeping track of steps
            steps += 1
