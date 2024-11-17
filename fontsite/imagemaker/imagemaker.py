from flask import Flask, request, send_file, session
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from training import FontDataset, DeepAutoencoder
import random
import matplotlib.pyplot as plt
import io
from PIL import Image

PORT = 18812
IMG_SIZE = 128
IMG_DIR = '/mnt/storage/smagid/thesis-smagid/fontdata/images'
CSV_DATA = '/mnt/storage/smagid/thesis-smagid/fontdata/data.csv'
MODEL_FILE = 'model_weights.pth'
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

dataset = FontDataset(csv_file = CSV_DATA, img_dir = IMG_DIR,
                            transform = transformations)
# loader = iter(DataLoader(dataset, batch_size=1, shuffle=True))

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
    print(modifier_list)
    modifier = torch.tensor(modifier_list)
    symbol = dataset[ID][0]
    reshaped = symbol.reshape(IMG_SIZE*IMG_SIZE)
    encoding = encoder(reshaped)
    modified = encoding + modifier * encoding
    decoded = decoder(modified)
    reshaped = decoded.reshape((IMG_SIZE,IMG_SIZE))
    img_pil = transforms.ToPILImage()(reshaped)
    img_io = io.BytesIO()
    img_pil.save(img_io, 'PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=PORT)