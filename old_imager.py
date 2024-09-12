from PIL import Image, ImageDraw, ImageFont
import freetype
import os

from os import listdir
from os.path import isfile, join

from time import sleep
from tqdm import tqdm

import csv

DATABASE_FILE = 'data.csv'
TTF_DIR = 'ttf-files'
FONT_FILE_LIST = [f for f in listdir(TTF_DIR) if isfile(join(TTF_DIR, f))]
NUM_FONTS = 100

# First row is labels
data = [['font', 'character', 'path']]

for FONT_FILE_NAME in tqdm(FONT_FILE_LIST[:NUM_FONTS]):
    # Configuration
    FONT_NAME = FONT_FILE_NAME[:-3]
    FONT_PATH = 'ttf-files/' + FONT_FILE_NAME
    OUTPUT_DIR = 'images/' + FONT_NAME[:-3]
    FONT_SIZE = 48  # Adjust size as needed

    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Load the font
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

    # Generate images for each character
    for char_code in range(33, 127):  # ASCII range
        char = chr(char_code)
        # Create an image with transparent background
        img = Image.new('RGBA', (FONT_SIZE, FONT_SIZE), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), char, font=font, fill=(0, 0, 0, 255))
        
        # Save the image
        save_path = os.path.join(OUTPUT_DIR, f'{char_code}.png')
        img.save(save_path)

        data.append([FONT_NAME, char, save_path])


# write to database
with open(DATABASE_FILE, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)