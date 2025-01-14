import argparse

from enums import Dataset, Mode, Optimizer, Model, Arch, Loss

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type = Mode, choices = list(Mode),
                        help = "Mode to run program")
    parser.add_argument('dataset', type = Dataset, choices = list(Dataset),
                        help = "Dataset to use")
    parser.add_argument('--binarize', dest = 'binarize', action = 'store_true',
                        help = "Binarize the input images")
    parser.add_argument('--no-binarize', dest = 'binarize', action = 'store_false',
                        help = "Leave input images as greyscale")
    parser.set_defaults(binarize=True)
    parser.add_argument('-b', '--beta', type = float,
                        default = 1.0, help = "Scaling of KL loss at train")
    parser.add_argument('--blanks', type = int,
                        default = 12, help = "Number of letters in font unobserved at test time")
    parser.add_argument('--decoder-type', type = Arch, choices = list(Arch),
                        default = Arch.conv, help = "Type of decoder architecture")
    parser.add_argument('--dynamic-params', dest = 'dynamic_params', action = 'store_true',
                        help = "Dynamically set decoder parameters based on latent embedding")
    parser.add_argument('--no-dynamic-params', dest = 'dynamic_params', action = 'store_false',
                        help = "Decoder is explicitly parameterized and learned")
    parser.set_defaults(dynamic_params=True)
    parser.add_argument('-e', '--tgt-epoch', type = int,
                        default = 100, help = "Target epoch to reach")
    parser.add_argument('--encoder-type', type = Arch, choices = list(Arch),
                        default = Arch.conv, help = "Type of encoder architecture")
    parser.add_argument('--eval-every', type = int,
                        default = 1, help = "How often to evaluate in epochs")
    parser.add_argument('--gpu', dest = 'gpu', action = 'store_true',
                        help = "Run on GPU")
    parser.add_argument('--no-gpu', dest = 'gpu', action = 'store_false',
                        help = "Run on CPU")
    parser.set_defaults(gpu=True)
    parser.add_argument('--hidden-size', type = int,
                        default = 128, help = "Hidden embedding size")
    parser.add_argument('--kl-threshold', type = float,
                        default = float('-inf'), help = "Threshold for KL of any dimension")
    parser.add_argument('-l', '--lamb', type = float,
                        default = 0.0, help = "Scaling of L1 loss at train")
    parser.add_argument('--loss', type = Loss, choices = list(Loss),
                        default = Loss.dct_cauchy, help = "Type of loss to use for reconstruction")
    parser.add_argument('-lr', '--learn-rate', type = float,
                        default = 1e-5, help = "Learning rate for training")
    parser.add_argument('--load-path', type = str,
                        default = None, help = "Path to checkpoint to resume")
    parser.add_argument('--model', type = Model, choices = list(Model),
                        default = Model.font_vae, help = "Type of model to use")
    parser.add_argument('-o', '--optim', type = Optimizer, choices = list(Optimizer),
                        default = Optimizer.adam, help = "Optimizer to use")
    parser.add_argument('--output-binarization', dest = 'output_binarization', action = 'store_true',
                        help = "Binarize the output images")
    parser.add_argument('--no-output-binarization', dest = 'output_binarization', action = 'store_false',
                        help = "Leave output images as greyscale")
    parser.set_defaults(output_binarization=False)
    parser.add_argument('-r', '--regularization', type = float,
                        default = 0, help = "L2 regularization constant")
    parser.add_argument('--sample-test', type = int,
                        default = 0, help = "Number of samples z to reconstruct with at test time")
    parser.add_argument('--save-every', type = int,
                        default = 0, help = "How often to save in epochs")
    parser.add_argument('--zca', dest = 'zca', action = 'store_true',
                        help = "Whiten data with ZCA")
    parser.add_argument('--no-zca', dest = 'zca', action = 'store_false',
                        help = "Don't whiten data")
    parser.set_defaults(zca=False)
    parser.add_argument('--z-size', type = int,
                        default = 32, help = "Latent z size")
    return parser.parse_args()
