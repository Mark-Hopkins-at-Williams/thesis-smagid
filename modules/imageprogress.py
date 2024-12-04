import os
import matplotlib.pyplot as plt
import torch

LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
GPU = "cuda"

def imageProgress(image_size, dev_set, model, step, out_dir, modeltype):
    """Saves an image of model progress on a sample of devset."""
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # handles both style-transfer pairs and autoencoder models
    if type == "pairs":
        (letter1_image, letter1_label, letter2_label), letter2_image = dev_set
        original = letter1_image.clone()[0].squeeze()
        letter1_image = letter1_image.to(GPU)
        letter1_label = letter1_label.to(GPU)
        letter2_label = letter2_label.to(GPU)
        model.eval()
        with torch.no_grad():
            out = model(letter1_image, letter1_label, letter2_label)
            out = out.to(GPU)
        trained = out.reshape(-1,image_size,image_size)[0]
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
    elif type == "auto": 
        sample = 5
        img = dev_set[0]
        img = img.reshape(-1, image_size*image_size)
        img = img.to(GPU)
        original = img.reshape(-1,image_size,image_size)[0]
        model.eval()
        with torch.no_grad():
            out = model(img)
            out = out.to("cuda")
        trained = out.reshape(-1,image_size,image_size)[0:sample]
        original = img.reshape(-1,image_size,image_size)[0:sample]
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
    plt.savefig(f'{out_dir}/{step}.png')