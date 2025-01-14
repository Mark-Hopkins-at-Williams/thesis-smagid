from enums import Metric
from imageio import imwrite
from model import Model
import random
from set_paths import paths
from skimage.metrics import structural_similarity as ssim
import torch
from torch.autograd import Variable
from utils import prep_write


class NearestNeighbor(Model):

    def initialize(self):
        self.curr_epoch = -1
        self.hyperparams = {'metric' : Metric.mse}
        self.fonts = {}

    def train_step(self, font):
        """Simply stores all training fonts in a dictionary."""
        filename, letters = font
        self.fonts[filename] = torch.tensor(letters)

    def standardize_pixel_intensities(self):
        return False

    def reconstruct(self, letter_matrix, mask):   
        letter_matrix_hat = None
        if self.hyperparams['metric'] == Metric.mse:
            best_mse = float("inf")
            for other_letter_matrix in self.fonts.values():                             
                # only consider the observed characters to find neighbor
                candidate_mse = torch.nn.functional.mse_loss(
                    (other_letter_matrix / 255.) * mask, 
                    (letter_matrix / 255.) * mask
                )
                if candidate_mse < best_mse:
                    best_mse = candidate_mse
                    letter_matrix_hat = other_letter_matrix
        elif self.hyperparams['metric'] == Metric.ssim: # is this ever used?
            best_ssim = float("-inf")
            for f in self.fonts.values():
                candidate_ssim = 0.0
                for j in range(26):
                    if mask[j,0] == 1:
                        candidate_ssim += float(ssim(f[j].detach().cpu().numpy() / 255., letter_matrix[j].detach().cpu().numpy() / 255., data_range=(letter_matrix[j].max()-letter_matrix[j].min()).item() / 255.))
                candidate_ssim /= mask.sum().item()
                if candidate_ssim > best_ssim:
                    best_ssim = candidate_ssim
                    letter_matrix_hat = f
        return letter_matrix_hat

    def get_mask(self, filename):       
        mask = torch.ones(26, 1, dtype=float)
        if filename in self.config.test_dict:
            blank_ids = self.config.test_dict[filename][:self.config.blanks]
        else:
            blank_ids = list(range(26))
            random.shuffle(blank_ids)
            blank_ids = blank_ids[:self.config.blanks]
        mask[blank_ids] = 0
        return mask

    def sample_fonts(self):
        num_samples = 100
        for i in range(num_samples):
            sample = prep_write(random.choice(self.fonts.values()) / 255.)
            imwrite("{}/{}/{}/sample_{}.png".format(paths['images'], self.config.mode, self.config.model, i), sample)
