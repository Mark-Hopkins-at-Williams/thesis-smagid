import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from training import FontDataset, DeepAutoencoder
import random
import matplotlib.pyplot as plt

IMG_SIZE = 128
IMG_DIR = 'fontdata/images'
CSV_DATA = 'fontdata/data.csv'
MODEL_FILE = 'weights/model_weights.24.pth'

model = DeepAutoencoder()
model.load_state_dict(torch.load(MODEL_FILE, weights_only=True))
model.eval()

# only the second half of the DeepAutoencoder
decoder = model.decoder

# only the first half
encoder = model.encoder

transformations = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    torchvision.transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(), 
    transforms.Normalize((0.5), (0.5)),
])

full_dataset = FontDataset(csv_file = CSV_DATA, img_dir = IMG_DIR,
                            transform = transformations)

def generate_random(n):
    # randomly take 10% of data
    sample_length = int(.1 * len(full_dataset))
    rand_dataset, _ = random_split(full_dataset, [sample_length, len(full_dataset) - sample_length])
    loader = DataLoader(rand_dataset, batch_size=10)
    batch = next(iter(loader))[0]
    if n == 1:
        return batch[0]
    else:
        return batch[:n, ...]

def encode_symbol(symbol_tensor):
    return encoder(symbol_tensor)

def decode_symbol(encoded_symbol):
    return decoder(encoded_symbol)

def experiment_1(index, percent = 50):
    """
    Experiment 1: Encode an image, alter one of the variables slightly,
    then decode it and compare it with the non-modified decoded image.
    """
    random_symbol = generate_random(n = 1)
    random_symbol = random_symbol.reshape(IMG_SIZE*IMG_SIZE)
    encoded_symbol = encode_symbol(random_symbol)
    modifier_tensor = torch.zeros(encoded_symbol.shape)
    modifier_tensor[index] = encoded_symbol[index] * (percent / 100)
    modified_encoding_1 = encoded_symbol - modifier_tensor
    modified_encoding_2 = encoded_symbol + modifier_tensor
    decoded_symbol_orig = decode_symbol(encoded_symbol)
    decoded_symbol_mod_1 = decode_symbol(modified_encoding_1)
    decoded_symbol_mod_2 = decode_symbol(modified_encoding_2)
    orig = decoded_symbol_orig.reshape(IMG_SIZE,IMG_SIZE)
    mod_1 = decoded_symbol_mod_1.reshape(IMG_SIZE,IMG_SIZE)
    mod_2 = decoded_symbol_mod_2.reshape(IMG_SIZE,IMG_SIZE)
    plt.clf()
    plt.subplot(1, 3, 1)
    plt.imshow(mod_1.cpu().detach().numpy(), cmap='gray')
    plt.title(f"minus {percent}% at index {index}")
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(orig.cpu().detach().numpy(), cmap='gray')
    plt.title(f"original")
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(mod_2.cpu().detach().numpy(), cmap='gray')
    plt.title(f"plus {percent}% at index {index}")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'experiments/experiment1.png')

def experiment_2():
    """
    Experiment 2: Encode two images, take their average, then decode it.
    """
    random_symbols = generate_random(n = 2)
    random_symbols = random_symbols.reshape(-1, IMG_SIZE*IMG_SIZE)
    encoded_symbols = encode_symbol(random_symbols)
    average_encodings = encoded_symbols[0] + encoded_symbols[1] // 2
    decoded_symbols = decode_symbol(encoded_symbols)
    average_decoded = decode_symbol(average_encodings)
    orig = decoded_symbols.reshape(-1,IMG_SIZE,IMG_SIZE)
    aver = average_decoded.reshape(IMG_SIZE,IMG_SIZE)
    plt.clf()
    plt.subplot(1, 3, 1)
    plt.imshow(orig[0].cpu().detach().numpy(), cmap='gray')
    plt.title("symbol 1")
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(aver.cpu().detach().numpy(), cmap='gray')
    plt.title("average")
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(orig[1].cpu().detach().numpy(), cmap='gray')
    plt.title("symbol 2")
    plt.axis('off')
    plt.savefig(f'experiments/experiment2.png')

def experiment_3():
    """
    Experiment 3: Generate some random encodings from zero-mean Gaussian
    with standard deviation from random tensor and decode them.
    """
    sample = 10
    random_symbol = generate_random(sample)
    random_symbol = random_symbol.reshape(-1, IMG_SIZE*IMG_SIZE)
    encoding = encode_symbol(random_symbol)
    std_dev = torch.std(encoding)
    random_encoding = torch.rand(encoding.shape) * std_dev
    random_decoded = decode_symbol(random_encoding)
    rand = random_decoded.reshape(-1, IMG_SIZE,IMG_SIZE)
    plt.clf()
    for i in range(sample):
        plt.subplot(2, sample // 2, i + 1)
        plt.imshow(rand[i].cpu().detach().numpy(), cmap='gray')
        plt.title(f"random {i + 1}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'experiments/experiment3.png')


if __name__ == "__main__":
    experiment_1(index = 2, percent = 200)
    #experiment_2()
    #experiment_3()