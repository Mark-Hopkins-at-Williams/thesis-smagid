import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from fontpairdataset import FontPairDataset # local import
from styletransfer import StyleTransfer # local import
from imageprogress import imageProgress # local import
from plotloss import plotLoss # local import


CSV_DATA = '/mnt/storage/smagid/thesis-smagid/fontdata/data.csv'
IMG_DATA = '/mnt/storage/smagid/thesis-smagid/fontdata/images'
IMG_SIZE = 64
BATCH_SIZE = 256
NUM_EPOCHS = 100
GPU = torch.device("cuda")
IMG_OUT_DIR = 'images'
IMG_EVERY = 10
LOSS_EVERY = 10
WEIGHTS_EVERY = 500

if __name__ == "__main__":    
    # load data, reserving 2% for test dev set
    full_dataset = FontPairDataset(csv_file = CSV_DATA, img_dir = IMG_DATA, image_size = IMG_SIZE)
    train_size = int(0.98 * len(full_dataset))
    dev_size = len(full_dataset) - train_size
    train_dataset, dev_dataset = random_split(full_dataset, [train_size, dev_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE) 
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=True) 
    # set up model training
    model = StyleTransfer() 
    criterion = torch.nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
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
            # unpacking batch
            (letter1_image, letter1_label, letter2_label), letter2_image = batch
            letter1_image = letter1_image.reshape(-1, IMG_SIZE*IMG_SIZE).to(GPU)
            letter1_label = letter1_label.to(GPU)
            letter2_label = letter2_label.to(GPU)
            # running batch through model and update weights
            out = model(letter1_image, letter1_label, letter2_label) 
            letter2_image = letter2_image.reshape(-1, IMG_SIZE*IMG_SIZE).to(GPU)
            loss = criterion(out, letter2_image)
            optimizer.zero_grad() 
            loss.backward() 
            optimizer.step() 
            # increment loss
            running_loss += loss.item() 
            # recording performance metrics every IMAGE_EVERY updates
            if i % IMG_EVERY == 0:
                sample_data = next(iter(dev_loader))
                imageProgress(IMG_SIZE, sample_data, model, IMG_EVERY, IMG_OUT_DIR, "pairs")
            # saving weights every WEIGHTS_EVERY updates
            if i % WEIGHTS_EVERY == 0 and i != 0:
                model.eval()
                torch.save(model.state_dict(), 'weights.pth')
                model.train()
            # plot loss every LOSS_EVERY
            if i % LOSS_EVERY == 0:
                # plot average training loss
                running_loss /= LOSS_EVERY
                train_loss.append(running_loss) 
                running_loss = 0
                plotLoss(data = train_loss, type = "train", step = LOSS_EVERY)
                # plot dev loss
                model.eval()
                with torch.no_grad():
                    sample_data = next(iter(dev_loader))
                    (s_letter1_image, s_letter1_label, s_letter2_label), s_letter2_image = sample_data
                    s_letter1_image = s_letter1_image.reshape(-1, IMG_SIZE*IMG_SIZE).to(GPU)
                    s_letter1_label = s_letter1_label.to(GPU)
                    s_letter2_label = s_letter2_label.to(GPU)
                    out = model(s_letter1_image, s_letter1_label, s_letter2_label)
                    s_letter2_image = s_letter2_image.reshape(-1, IMG_SIZE*IMG_SIZE).to(GPU)
                    loss = criterion(out, s_letter2_image).cpu().detach().numpy()
                    dev_loss.append(loss)
                plotLoss(data = dev_loss, type = "dev", step = LOSS_EVERY)
            # keeping track of steps
            steps += 1