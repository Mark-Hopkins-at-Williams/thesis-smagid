import fontforge
import os
from os import listdir
from os.path import isfile, join
import csv

DATABASE_FILE = "data.csv"
TTF_DIRECTORY = "ttf-files"
IMG_DIR = "non-resized-images"
# NUM_FONTS = 1000 # handle the first n fonts in directory
IMG_SIZE = 512

# create a list of all the fonts
font_list = [f for f in listdir(TTF_DIRECTORY) if isfile(join(TTF_DIRECTORY, f))]

# deal with all fonts
NUM_FONTS = len(font_list)

# create data output list
data = [['font', 'character', 'path']]

# define the Unicode ranges for basic characters
basic_char_ranges = [
    (0x0021, 0x007E),  # punctuation, and basic symbols
    (0x0041, 0x005A),  # Uppercase letters A-Z
    (0x0061, 0x007A),  # Lowercase letters a-z
    (0x0030, 0x0039),  # Numbers 0-9
]

# Function to check if the Unicode value falls in the desired ranges
def is_basic_char(unicode_val):
    return any(start <= unicode_val <= end for start, end in basic_char_ranges)

# iterate through font list
for font_file in font_list[:NUM_FONTS]:
    # open font file
    font = fontforge.open(f"./{TTF_DIRECTORY}/{font_file}")
    # font_name = font_file.removesuffix(".ttf").removesuffix(".otf")
    font_name = font_file[:-4]
    # Create output directory if it doesn't exist
    output_dir = f"./{IMG_DIR}/{font_name}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # loop through each glyph in the font
    for glyph in font.glyphs():
        # get the Unicode code point of the glyph
        unicode_val = glyph.unicode
        # check if the glyph is worth exporting (non-empty)
        if glyph.isWorthOutputting() and is_basic_char(unicode_val):
            if unicode_val != -1:
                char_representation = chr(unicode_val)
            else:
                char_representation = "non-printable"
            # export the glyph to a PNG file
            export_path = f"./{IMG_DIR}/{font_name}/{glyph.glyphname}.png"
            glyph.export(export_path, IMG_SIZE)
            data.append([font_name, char_representation, export_path])

# write to database
with open(DATABASE_FILE, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)
