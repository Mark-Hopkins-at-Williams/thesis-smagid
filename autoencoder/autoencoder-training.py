import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from fontdataset import FontDataset # local import
from fontautoencoder import FontAutoencoder # local import
from imageprogress import imageProgress # local import
from plotloss import plotLoss # local import

CSV_DATA = '/mnt/storage/smagid/thesis-smagid/fontdata/data.csv'
IMG_DATA = '/mnt/storage/smagid/thesis-smagid/fontdata/images'
IMG_SIZE = 128
DATA_RANGE = None
BATCH_SIZE = 256
GPU = torch.device("cuda")
IMG_EVERY = 10
IMG_OUT_DIR = 'images'
LOSS_EVERY = 10
WEIGHTS_EVERY = 500

if __name__ == "__main__":    
    # load data, reserving 2% for test dev set
    full_dataset = FontDataset(csv_file = CSV_DATA, img_dir = IMG_DATA, img_size = IMG_SIZE)
    train_size = int(0.98 * len(full_dataset))
    dev_size = len(full_dataset) - train_size
    train_dataset, dev_dataset = random_split(full_dataset, [train_size, dev_size])
    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE) 
    dev_loader = DataLoader(dev_dataset, batch_size = BATCH_SIZE)
    # set up model training
    model = FontAutoencoder(image_size = IMG_SIZE) 
    criterion = torch.nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_loss = [] 
    num_batches = len(train_loader)
    # transfer model to gpu
    model.to(GPU)
    # keeping track of losses
    running_loss = 0
    train_loss = [] 
    dev_loss = []
    # start training
    steps = 0
    while True: # ignore epochs for now
        for i, batch in tqdm(enumerate(train_loader)): 
            # unpack batch
            img = batch[0]
            img = img.reshape(-1, IMG_SIZE*IMG_SIZE).to(GPU)
            # run through model
            out = model(img) 
            loss = criterion(out, img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # increment loss
            running_loss += loss.item() 
            # record image every IMG_EVERY
            if i % IMG_EVERY == 0:
                sample_data = next(iter(dev_loader))
                imageProgress(IMG_SIZE, sample_data, model, IMG_EVERY, IMG_OUT_DIR, "auto")
            # save weights evert WEIGHTS_EVERY
            if i % WEIGHTS_EVERY == 0 and i != 0:
                model.eval()
                torch.save(model.state_dict(), 'weights.pth')
                model.train()
            # plot loss every LOSS_EVERY
            if i % LOSS_EVERY == 0:
                # plot avg training loss
                running_loss /= LOSS_EVERY
                train_loss.append(running_loss) 
                running_loss = 0
                plotLoss(data = train_loss, type = "train", step = LOSS_EVERY)
                # plot dev loss
                model.eval()
                with torch.no_grad():
                    sample_data = next(iter(dev_loader))
                    img = sample_data[0]
                    img = img.reshape(-1, IMG_SIZE*IMG_SIZE).to(GPU)
                    out = model(img)
                    loss = criterion(out, img)
                    dev_loss.append(loss)
                plotLoss(data = dev_loss, type = "dev", step = LOSS_EVERY)
            # update steps
            steps += 1