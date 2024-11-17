from flask import Flask, request, send_file, session
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from new_training import FontPairDataset, DeepAutoencoder
from torch.nn import Embedding, Parameter
import random
import matplotlib.pyplot as plt
import io
from PIL import Image

PORT = 18812
IMG_SIZE = 128
IMG_DIR = '/mnt/storage/smagid/thesis-smagid/fontdata/images'
CSV_DATA = '/mnt/storage/smagid/thesis-smagid/fontdata/data.csv'
MODEL_FILE = 'new_weights.pth'
NUM_MODIFIERS = 10 # bottleneck size

model = DeepAutoencoder()
model.load_state_dict(torch.load(MODEL_FILE, weights_only=True))
model.eval()
encoder = model.encoder
decoder = model.decoder

transformations = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    torchvision.transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(), 
    transforms.Normalize((0.5), (0.5)),
])

dataset = FontPairDataset(csv_file = CSV_DATA, img_dir = IMG_DIR,
                            transform = transformations)

app = Flask(__name__)
app.secret_key = 'sampassword'

@app.route('/symbolizer', methods=['GET'])
def generate_plot():
    """
    Generates a random font image passed through autoencoder.
    """
    # get parameters from URL (or default values)
    ID = int(request.args.get('id', default = random.randint(0, len(dataset) - 1)))
    modifier_list = []
    # make modifier parameters into a modifier tensor
    for i in range(NUM_MODIFIERS):
        modifier_list.append(int(request.args.get(f'mod{i}', default = 0)) / 100)
    
    modifier = torch.tensor(modifier_list)

    (letter1_image, letter1_label, letter2_label), letter2_image = dataset[ID]

    letter1_image = letter1_image.reshape(IMG_SIZE*IMG_SIZE)

    letter1_embed = model.letter1_embedding(letter1_label) 
    letter2_embed = model.letter2_embedding(letter2_label)

    encoding = encoder(torch.cat([letter1_image, letter1_embed], dim=0))

    modified = encoding + modifier * encoding

    decoded = decoder(torch.cat([modified, letter2_embed], dim=0))

    reshaped = decoded.reshape((IMG_SIZE,IMG_SIZE))
    img_pil = transforms.ToPILImage()(reshaped)
    img_io = io.BytesIO()
    img_pil.save(img_io, 'PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=PORT)