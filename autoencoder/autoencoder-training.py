import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import pandas as pd
from fontdataset import FontDataset # local import
from fontautoencoder import FontAutoencoder # local import
from imageprogress import imageProgress # local import
from plotloss import plotLoss # local import

DATA_DIR = '/mnt/storage/smagid/thesis-smagid/fontdata/'
IMG_SIZE = 64
DATA_RANGE = None
BATCH_SIZE = 256
GPU = torch.device("cuda")
IMG_EVERY = 10
IMG_OUT_DIR = 'images'
LOSS_EVERY = 50
WEIGHTS_EVERY = 500

if __name__ == "__main__":    
    # load data, reserving 2% for test dev set
    full_dataset = FontDataset(data_dir = DATA_DIR, img_size = IMG_SIZE)
    train_size = int(0.98 * len(full_dataset))
    dev_size = len(full_dataset) - train_size
    train_dataset, dev_dataset = random_split(full_dataset, [train_size, dev_size])
    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE) 
    dev_loader = DataLoader(dev_dataset, batch_size = BATCH_SIZE)
    # set up model training
    model = FontAutoencoder() 
    criterion = torch.nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_loss = []
    num_batches = len(train_loader)
    # transfer model to gpu
    model.to(GPU)
    # keeping track of losses
    trainLoss = pd.DataFrame(columns=["Steps", "Loss"])
    devLoss = pd.DataFrame(columns=["Steps", "Loss"])
    # start training
    steps = 0
    while True: # ignore epochs for now
        for batch in tqdm(enumerate(train_loader)):
            img = batch[1].reshape(-1, IMG_SIZE*IMG_SIZE).to(GPU)
            # run through model
            out = model(img) 
            loss = criterion(out, img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # record image every IMG_EVERY
            # if i % IMG_EVERY == 0:
            #     sample_data = next(iter(dev_loader))
            #     imageProgress(IMG_SIZE, sample_data, model, IMG_EVERY, IMG_OUT_DIR, "auto")
            # save weights evert WEIGHTS_EVERY
            if steps % WEIGHTS_EVERY == 0 and steps > 0:
                model.eval()
                torch.save(model.state_dict(), 'weights.pth')
                model.train()
            # plot loss every LOSS_EVERY
            if steps % LOSS_EVERY == 0:
                # plot training loss
                newRow = pd.DataFrame({"Steps": steps, "Loss": loss.cpu().detach().numpy()}, index=[0])
                trainLoss = pd.concat([trainLoss, newRow], ignore_index=True)
                plotLoss(data = trainLoss, type = "train")
                # plot dev loss
                model.eval()
                with torch.no_grad():
                    sample_data = next(iter(dev_loader))
                    img = sample_data[0]
                    img = img.reshape(-1, IMG_SIZE*IMG_SIZE).to(GPU)
                    out = model(img)
                    loss = criterion(out, img)
                    newRow = pd.DataFrame({"Steps": steps, "Loss": loss.cpu().detach().numpy()}, index=[0])
                    devLoss = pd.concat([devLoss, newRow], ignore_index=True)
                plotLoss(data = devLoss, type = "dev")
            # update steps
            steps += 1