import torch
from torch.nn import Embedding, Parameter

class FontAutoencoder(torch.nn.Module): 
    def __init__(self): 
        super().__init__()         
        self.letter_embedding_dim = 10
        self.encoder = torch.nn.Sequential( 
            torch.nn.Linear(4096, 2048), 
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 1024), 
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512), 
            torch.nn.ReLU(), 
            torch.nn.Linear(512, 256), 
            torch.nn.ReLU(), 
            torch.nn.Linear(256, 128), 
            torch.nn.ReLU(), 
            torch.nn.Linear(128, 64), 
            torch.nn.ReLU(), 
            torch.nn.Linear(64, 32),
        ) 
          
        self.decoder = torch.nn.Sequential( 
            torch.nn.Linear(32, 64), 
            torch.nn.ReLU(), 
            torch.nn.Linear(64, 128), 
            torch.nn.ReLU(), 
            torch.nn.Linear(128, 256), 
            torch.nn.ReLU(), 
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(), 
            torch.nn.Linear(512, 1024),
            torch.nn.ReLU(), 
            torch.nn.Linear(1024, 2048),
            torch.nn.ReLU(), 
            torch.nn.Linear(2048, 4096),
            torch.nn.Sigmoid() 
        ) 
  
    def forward(self, img): 
        encoded = self.encoder(img)        
        decoded = self.decoder(encoded)
        return decoded