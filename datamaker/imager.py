"""
Takes a directory of .otf and .ttf files and generates a corresponding
directory of images of basic characters in that font.

"""

import fontforge
import os
from os import listdir
from os.path import isfile, join
import csv

DATABASE_FILE = "data.csv"
TTF_DIRECTORY = "ttf-files"
IMG_DIR = "images"
IMG_SIZE = 64
CHARACTERS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?:;@#$%&*()[]")

def generateGlyphs(font, outputDir, data):
    print(f"Generating images for {fontName}")
    for char in CHARACTERS:
        glyph = font[ord(char)]
        exportPath = f"{outputDir}/{glyph.unicode}.png"
        glyph.export(exportPath, IMG_SIZE)
        data.append([fontName, chr(glyph.unicode), exportPath])

if __name__ == "__main__":
    # get file names
    fontList = [f for f in listdir(TTF_DIRECTORY) if isfile(join(TTF_DIRECTORY, f))]
    # create data output list
    data = [['font', 'character', 'path']]
    for file in fontList:
        print(f"Trying {file}")
        font = fontforge.open(f"{TTF_DIRECTORY}/{file}")
        fontName = font.fullname
        # check that all characters in set exist
        charsInFont = [glyph for glyph in font.glyphs() if glyph.unicode != -1 and chr(glyph.unicode) in CHARACTERS]
        if len(charsInFont) == len(CHARACTERS):
            outputDir = f"{IMG_DIR}/{fontName}"
            # check if dir already exists
            if os.path.isdir(outputDir):
                # count if already populated with images
                items = os.listdir(outputDir)
                # filter out for only files
                files = [item for item in items if os.path.isfile(os.path.join(outputDir, item))]
                if len(files) < len(CHARACTERS): # if not fully populated
                    generateGlyphs(font, outputDir, data)
            # if doesn't exist, generate images
            else:
                os.makedirs(outputDir)
                generateGlyphs(font, outputDir, data)
        else:
            print(f"Skipping {fontName}")

    # write to database
    with open(DATABASE_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
