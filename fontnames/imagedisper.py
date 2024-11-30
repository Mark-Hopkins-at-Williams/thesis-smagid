import os
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

FOLDER = '/mnt/storage/smagid/thesis-smagid/datamaker/ttf-files/'

def create_font_sample(font_path, output_path):
    # Create an image with PIL
    width, height = 600, 100  # Adjust the image size as needed
    image = Image.new('RGB', (width, height), color=(255, 255, 255))  # White background
    draw = ImageDraw.Draw(image)

    # Load the font using PIL (you can load ttf or otf directly)
    try:
        pil_font = ImageFont.truetype(font_path, size=40)  # Set the font size
    except IOError:
        raise Exception(f"Font file {font_path} could not be loaded with PIL")

    # Get the name of the font (use the font file name or any custom text)
    font_name = font_path.split('/')[-1].replace('.ttf', '').replace('.otf', '')

    # Use textbbox to calculate the size of the text
    bbox = draw.textbbox((0, 0), font_name, font=pil_font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Draw the font name text centered
    text_x = (width - text_width) // 2
    text_y = (height - text_height) // 2
    draw.text((text_x, text_y), font_name, font=pil_font, fill=(0, 0, 0))  # Black text

    # Save the image
    image.save(output_path)

with open('alphabetic_fonts.txt', 'r') as file:
    lines = file.readlines()

alpha_fonts = [line.strip() for line in lines]

for filename in tqdm(os.listdir(FOLDER)):
    if filename.endswith('.ttf') or filename.endswith('.otf'):
        font_name = filename.replace('.ttf', '').replace('.otf', '')
        if font_name in alpha_fonts:
            font_path = os.path.join(FOLDER, filename)
            output_path = f"outputs/{filename.replace('.ttf', '').replace('.otf', '')}.png"
            try:
                create_font_sample(font_path, output_path)
            except:
                print(font_name)