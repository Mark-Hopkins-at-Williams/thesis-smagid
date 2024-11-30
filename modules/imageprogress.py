import matplotlib.pyplot as plt
import torch

LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
TEST_IMG_DIR = 'images'
GPU = "cuda"

def image_progress(image_size, dev_set, model, step):
    """Saves an image of model progress on a sample of devset."""
    
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
    plt.tight_layout()
    plt.savefig(f'{TEST_IMG_DIR}/{step}.png')