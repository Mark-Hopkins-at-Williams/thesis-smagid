from enums import Dataset, Mode
from imageio import imread
import numpy as np
import os
from set_paths import paths
import sys


def read_split(split, config):
    """For a given split, returns a list of (filename, letters) tuples.

    - filename is the font filename
    - letters is a list of length 26 for which the kth element is another list
      of length 4096 that provides the grayscale pixel intensities for the 64x64
      image depicting the kth capital letter of the alphabet
    - font is a tuple of (filename, letters)

    """
    def cap64(split):
        fonts = []
        split_path = paths['cap64'] + '/' + split
        for filename in os.listdir(split_path):
            font_image = imread(split_path + '/' + filename)
            letters = np.split(font_image, 26, axis=1)
            letters = [letter.reshape([-1]).tolist() for letter in letters]        
            fonts.append((filename, letters))
        return fonts
    
    print(f"Loading {split} from {config.dataset} ...")
    if config.dataset == Dataset.cap64:
        return cap64(split)
    else:
        sys.exit("Unrecognized dataset")

def load_corpus(config):
    if config.mode == Mode.toy:
        train = read_split('toy_train', config)
        dev = read_split('toy_val', config)
        dev_hard = read_split('toy_hard', config)
        test = read_split('toy_val', config)
        test_hard = read_split('toy_hard', config)
    else:
        train = read_split('train', config)
        dev = read_split('val', config)
        dev_hard = read_split('val_hard', config)
        if config.mode == Mode.test:
            test = read_split('test', config)
            test_hard = read_split('test_hard', config)
        else:
            test = read_split('val', config)
            test_hard = read_split('val_hard', config)
    return train, dev, dev_hard, test, test_hard
