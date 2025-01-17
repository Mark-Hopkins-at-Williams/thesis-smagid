from abc import ABC, abstractmethod
import enums
from enums import Metric
from imageio import imwrite
import random
from set_paths import paths
from skimage.metrics import structural_similarity as ssim
import sys
import torch
from torch.autograd import Variable
from tqdm import tqdm
from utils import prep_write, smart_binarize
import os


class Model(ABC, torch.nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.initialize()
        if config.load_path is not None: # reload model from previous checkpoint, if applicable
            self.load()
            volatile = ['mode', 'blanks', 'output_binarization', 'tgt_epoch']
            for var in volatile:
                newval = getattr(config, var)
                setattr(self.config, var, newval)

    # @abstractmethod
    def initialize(self):
        """Performs any model-specific initialization."""
        pass

    # @abstractmethod
    def train_step(self, font):
        """Trains the model on the next available training font.
        
        Arguments:
        - font is a pair (filename, letters) in which:
          - letters is a list of 26 lists. The kth list is a length-4096 that
            contains the pixel intensities of a 64x64 image of the kth capital
            letter of the English alphabet.            
        """
        pass

    # @abstractmethod
    def reconstruct(self, letter_matrix, mask):
        """Reconstructs a partially specified letter matrix.
        
        Arguments:
        - letter_matrix is a 26x4096 matrix where the kth row corresponds to the pixel
          intensities of the image representing the kth letter of the English alphabet
        - mask is a 26x1 binary vector, where 0 represents a letter that must be reconstructed

        Returns:
        - a 26x4096 matrix where the missing letters have been reconstructed
        """
        pass

    # @abstractmethod
    def get_mask(self, filename):
        """Determines which letters from a font should be masked.
        
        Returns:
        - a 26x1 binary vector, where 0 represents a letter that must be reconstructed
        """
        pass

    # @abstractmethod
    def sample_fonts(self):
        pass

    def train(self, train, dev, dev_hard, test, test_hard):
        """Runs a general-purpose training loop.
        
        Model-specific training code is provided by self.train_step.
        """
        self.training = False
        for e in range(self.curr_epoch + 1, self.config.tgt_epoch + 1):
            self.curr_epoch = e
            self.training = True
            random.shuffle(train)
            for font in tqdm(train):                
                self.train_step(font)
            # save and evaluate
            self.training = False
            if e % self.config.eval_every == 0 or e == self.config.tgt_epoch:
                self.print_metrics([train, dev, dev_hard], ["tr", "d", "dh"])
                self.sample_fonts()
                self.visualize_reconstructions(dev, "dev", max_fonts=100)
                self.visualize_reconstructions(dev_hard, "dev_hard", max_fonts=100)
            if self.config.save_every > 0 and e % self.config.save_every == 0 or e == self.config.tgt_epoch:
                self.save()
        self.print_metrics([train, dev, dev_hard, test, test_hard], ["tr", "d", "dh", "ts", "tsh"])
        self.sample_fonts()
        self.visualize_reconstructions(dev, "dev", max_fonts=100)
        self.visualize_reconstructions(dev_hard, "dev_hard", max_fonts=1000)
        self.visualize_reconstructions(test, "test", max_fonts=100)
        self.visualize_reconstructions(test_hard, "test_hard", max_fonts=1000)

    def visualize_reconstructions(self, corpus, prefix, max_fonts=None):
        """Reconstructs missing letters for each font in the provided corpus.
        
        The reconstructed images are written to files.
        
        """
        if max_fonts is None:
            max_fonts = len(corpus)
        for i, font in enumerate(corpus[:max_fonts]):
            filename, letters = font
            letter_matrix = torch.tensor(letters, dtype=float) # letter_matrix is 26x4096            
            mask = self.get_mask(filename)
            letter_matrix_hat = self.reconstruct(letter_matrix, mask)
            if self.standardize_pixel_intensities():
                letter_matrix /= 255.
                letter_matrix_hat /= 255.
            # write reconstructions to file
            if self.config.output_binarization:
                letter_matrix_hat = self.config.float(smart_binarize(letter_matrix_hat))
            inter = letter_matrix * mask + letter_matrix_hat * (1 - mask)
            if not self.standardize_pixel_intensities():
                inter /= 255.
            epoch_affix = f'_e{self.curr_epoch}' if self.curr_epoch >= 0 else ''
            path = "{}/{}/{}/recon_{}{}_b{}_{}_{}".format(
                paths['images'], 
                self.config.mode, 
                self.config.model, 
                prefix, 
                epoch_affix,
                self.config.blanks, 
                i, 
                filename
            )
            print("SAVING")
            imwrite(path, prep_write(inter))

    def evaluate(self, corpus, metric, max_fonts=None):
        """Evaluates model reconstruction ability based on several metrics."""
        if max_fonts is None:
            max_fonts = len(corpus)
        if metric == Metric.nll_mf:
            if self.config.model == enums.Model.font_vae:
                total_nll = 0.0
                for font in corpus[:max_fonts]:
                    datum = font[1]
                    datum = Variable(self.config.float(datum))
                    total_nll -= float(self.forward(datum))
                num_letters = 26. * max_fonts
                return total_nll / num_letters
            else:
                return 0.0
        elif metric == Metric.mse:
            total_mse = 0.0
            for i, font in enumerate(corpus[:max_fonts]):
                filename, datum = font
                datum = Variable(self.config.float(datum))
                mask = self.get_mask(filename)
                datum_hat = self.reconstruct(datum, mask)
                # only compute the loss over the unobserved characters
                if self.config.blanks == 0:
                    mask = torch.zeros_like(mask)
                if filename in self.config.test_dict:
                    print(filename, float(torch.nn.functional.mse_loss((datum_hat / 255.) * (1-mask), (datum / 255.) * (1-mask), reduction='sum')) / self.config.blanks)
                font_mse = float(torch.nn.functional.mse_loss((datum_hat / 255.) * (1-mask), (datum / 255.) * (1-mask), reduction='sum'))
                # dir = f"{paths['images']}/{self.config.mode}/{self.config.model}"
                # os.makedirs(dir, exist_ok=True)
                # outfile = open(f"{dir}/mse_e{self.curr_epoch}_b{self.config.blanks}_{filename}.txt", 'w+')
                # outfile.write(str(font_mse / self.config.blanks))
                # outfile.close()
                total_mse += font_mse
            if self.config.blanks != 0:
                return total_mse / (max_fonts * self.config.blanks)
            else:
                return total_mse / (max_fonts * 26)
        elif metric == Metric.ssim:
            total_ssim = 0.0
            for i, font in enumerate(corpus[:max_fonts]):
                filename, datum = font
                datum = Variable(self.config.float(datum))
                mask = self.get_mask(filename)
                datum_hat = self.reconstruct(datum, mask)
                if self.config.blanks == 0:
                    mask = torch.zeros_like(mask)
                for j in range(26):
                    if mask[j,0].item() == 0:
                        total_ssim += float(ssim(datum_hat[j].detach().cpu().numpy() / 255., datum[j].detach().cpu().numpy() / 255., data_range=(datum[j].max()-datum[j].min()).item() / 255.))
            if self.config.blanks != 0:
                return total_ssim / (max_fonts * self.config.blanks)
            else:
                return total_ssim / (max_fonts * 26)
        else:
            sys.exit("Unsupported metric")

    def print_metrics(self, corpora, names, max_fonts=None):
        """Prints current model statistics to terminal."""
        outstring = "E:{:4d}".format(self.curr_epoch)
        values = []
        for metric in list(Metric):
            for corpus, name in zip(corpora, names):
                outstring += "    " + name
                outstring += " {}:".format(metric)
                outstring += "  {:10.4f}"
                values.append(self.evaluate(corpus, metric, max_fonts=max_fonts))
        print(outstring.format(*values))

    def save(self):
        """Saves model weights and parameters to .pt file."""
        save_path = self.get_save_path()
        save_dict = {'state_dict' : self.state_dict(),
                     'config' : self.config,
                     'curr_epoch' : self.curr_epoch,
                     'hyperparams' : self.hyperparams}
        torch.save(save_dict, save_path)

    def load(self):
        """Loads a checkpoint model weights .pt file."""
        checkpoint_dict = torch.load(self.config.load_path)
        self.load_state_dict(checkpoint_dict['state_dict'], strict=False)
        self.config = checkpoint_dict['config']
        self.curr_epoch = checkpoint_dict['curr_epoch']
        self.hyperparams = checkpoint_dict['hyperparams']

    def get_save_path(self):
        """Generates save path directory for checkpoints."""
        filename = "{}_E-{}.pt".format(self, self.curr_epoch)
        os.makedirs(f"{paths['checkpoints']}/{self.config.mode}", exist_ok=True)
        return "{}/{}/{}".format(paths['checkpoints'], self.config.mode, filename)

    def __str__(self):
        string = ""
        for k,v in self.hyperparams.items():
            string += "{}-{}_".format(k, v)
        return string
