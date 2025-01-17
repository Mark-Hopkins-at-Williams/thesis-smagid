from arguments import parse_args
from dataio import load_corpus
from enums import Mode, Model
from nearest_neighbor import NearestNeighbor
from os.path import join
import pickle
from preprocessor import Preprocessor
from set_paths import paths
import torch
from font_vae import FontVAE


def load_hard(corpus, name):
    """Loads the 10% of fonts furthest in L2 distance from their
    nearest neighbor."""
    hard_font_file = open("hardest_10_{}.txt".format(name))
    hard_fonts = set()
    for line in hard_font_file:
        hard_fonts.add(line.strip())
    return [(x,y) for (x,y) in corpus if x in hard_fonts]

def main(config):
    print(config)
    # set data types
    if config.gpu:
        config.float = torch.cuda.FloatTensor
        config.long = torch.cuda.LongTensor
    else:
        config.float = torch.FloatTensor
        config.long = torch.LongTensor
    
    # load data
    if config.mode != Mode.test:
        print("Recycling dev data for test")
    
    train, dev, dev_hard, test, test_hard = load_corpus(config)
    print("Train images:", len(train))
    print("Dev images:", len(dev))
    print("Dev (hard) images:", len(dev_hard))
    print("Test images:", len(test))
    print("Test (hard) images:", len(test_hard))
    
    config.proc = Preprocessor(config, train)

    test_dict = pickle.load(open(join(paths['cap64'], "test_dict/dev_dict.pkl"), 'rb'))
    config.test_dict = test_dict
    
    if config.model == Model.nearest_neighbor:
        model = NearestNeighbor(config)
    elif config.model == Model.font_vae:
        model = FontVAE(config)
    
    model.config.test_dict = test_dict
    model.train(train, dev, dev_hard, test, test_hard)

if __name__ == '__main__':
    config = parse_args()
    main(config)
