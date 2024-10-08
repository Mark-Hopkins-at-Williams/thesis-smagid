"""
Takes all images in the images directory and makes them square
by pasting it onto a white square image whose width is equal
to the longest dimension of the original image.

"""

from PIL import Image
from tqdm import tqdm
import os

for font_dir in tqdm(os.listdir('images')):
    for image_name in os.listdir('images/' + font_dir):
        if image_name.endswith(".png"):
            image_path = 'images/' + font_dir + '/' + image_name
            with Image.open(image_path) as img:
                width, height = img.size
                new_size = max(width, height)
                new_img = Image.new('RGBA', (new_size, new_size), (255, 255, 255, 255))
                x = (new_size - width) // 2
                y = (new_size - height) // 2
                new_img.paste(img, (x, y))
                new_img.save(image_path)