import torch
from tqdm import tqdm
from utils import binarize


class Preprocessor:
    def __init__(self, config, corpus):
        self.config = config
        self.eps = 0.1
        self.mu = None
        self.ZCA = None
    
        # Make a large matrix where each row corresponds to one letter of one font.
        # Binarize the grayscale pixel values, if desired.
        X = torch.empty(len(corpus) * 26, 64 * 64, dtype=float)
        for i, font in tqdm(enumerate(corpus)):
            _, letters = font
            letters = torch.tensor(letters)
            if self.config.binarize:
                letters = binarize(letters)
            for j, letter in enumerate(letters):
                X[i * 26 + j] = letter.view(-1)
        
        # Compute the mean pixel intensity for each letter (i.e., row of matrix X),
        # then subtract it from each pixel intensity, so that the pixel intensities
        # for each letter center around zero.
        self.mu = torch.mean(X, dim=0)          
        X -= self.mu.view(1, -1)

        # Build the ZCA transformation matrix (TODO: find out what this is).
        cov = 1. / (X.shape[0] - 1) * torch.matmul(X.t(), X)
        U, S, V = torch.svd(cov)
        self.ZCA = torch.matmul(torch.matmul(U, torch.diag(1./((S + self.eps) ** 0.5))), U.t())

    def preprocess(self, x):
        x_shape = x.shape
        x_flat = x.view(26, -1)
        x_cent = x_flat - self.mu.view(1, -1)
        zca_x = torch.matmul(self.ZCA.view(1, 64 * 64, 64 * 64), x_cent.view(26, -1, 1))
        return zca_x.view(x_shape)
