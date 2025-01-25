import os
from PIL import Image
from tqdm import tqdm

IMAGE_DIR = "/mnt/storage/smagid/thesis-smagid/fontdata/images"
OUTPUT_PATH = "outputs"
CAPITALS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
IMAGE_SIZE = 64

os.makedirs(OUTPUT_PATH, exist_ok=True)

for fontName in tqdm(os.listdir(IMAGE_DIR)):
    output = Image.new("L", (64 * len(CAPITALS), 64))
    x_offset = 0
    for letter in CAPITALS:
        unicodeVal = ord(letter)
        imagePath = f"{IMAGE_DIR}/{fontName}/{unicodeVal}.png"
        image = Image.open(imagePath).convert("L")
        output.paste(image, (x_offset, 0))
        x_offset += 64
    outputPath = f"{OUTPUT_PATH}/{fontName}.png"
    output.save(outputPath)
