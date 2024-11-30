from torchvision.models import resnet18
import torch
import torch.nn as nn
from torch.nn import Embedding, Parameter

class FontResNet(nn.Module):
    """
    A modified resnet18 model to output glyph images.
    """
    def __init__(self, image_size):
        super(FontResNet, self).__init__()
        self.resnet = resnet18()
        self.image_size = image_size
        # first convolutional layer has one channel for B/W (21 to match embeddings??)
        self.resnet.conv1 = nn.Conv2d(21, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # output image_size * image_size tensor
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, image_size * image_size)
        self.letter1_embedding = Embedding(num_embeddings=52, embedding_dim=10)
        self.letter2_embedding = Embedding(num_embeddings=52, embedding_dim=10)

    def forward(self, letter1_img, letter1_label, letter2_label): 
        letter1_embed = self.letter1_embedding(letter1_label)
        letter2_embed = self.letter2_embedding(letter2_label)
        embeddings = torch.cat([letter1_embed, letter2_embed], dim=1)
        embeddings = embeddings.unsqueeze(2).unsqueeze(3)
        embeddings = embeddings.expand(-1, -1, self.image_size, self.image_size)
        full_tensor = torch.cat([letter1_img, embeddings], dim=1)
        # forces pixel values to binary B/W
        return torch.sigmoid(self.resnet(full_tensor))