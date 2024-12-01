import torch
from torch.nn import Embedding

class StyleTransfer(torch.nn.Module): 
    def __init__(self, img_size): 
        super().__init__()         
        self.letter_embedding_dim = 10
        self.encoder = torch.nn.Sequential( 
            torch.nn.Linear(img_size*img_size + self.letter_embedding_dim, 6272), 
            torch.nn.ReLU(),
            torch.nn.Linear(6272, 3136), 
            torch.nn.ReLU(),
            torch.nn.Linear(3136, 1568), 
            torch.nn.ReLU(),
            torch.nn.Linear(1568, 784), 
            torch.nn.ReLU(), 
            torch.nn.Linear(784, 256), 
            torch.nn.ReLU(), 
            torch.nn.Linear(256, 128), 
            torch.nn.ReLU(), 
            torch.nn.Linear(128, 64), 
            torch.nn.ReLU(), 
            torch.nn.Linear(64, 10) 
        ) 
          
        self.decoder = torch.nn.Sequential( 
            torch.nn.Linear(10+self.letter_embedding_dim, 64), 
            torch.nn.ReLU(), 
            torch.nn.Linear(64, 128), 
            torch.nn.ReLU(), 
            torch.nn.Linear(128, 256), 
            torch.nn.ReLU(), 
            torch.nn.Linear(256, 784),
            torch.nn.ReLU(), 
            torch.nn.Linear(784, 1568),
            torch.nn.ReLU(), 
            torch.nn.Linear(1568, 3136),
            torch.nn.ReLU(), 
            torch.nn.Linear(3136, 6272),
            torch.nn.ReLU(), 
            torch.nn.Linear(6272, img_size*img_size), 
            torch.nn.Sigmoid() 
        ) 
        self.letter1_embedding = Embedding(num_embeddings=52, embedding_dim=10)
        self.letter2_embedding = Embedding(num_embeddings=52, embedding_dim=10)
        
  
    def forward(self, letter1_img, letter1_label, letter2_label): 
        try:
            letter1_embed = self.letter1_embedding(letter1_label) 
        except Exception:
            print(letter1_label.device)
            print(self.letter1_embedding.weight.device)
            
        letter2_embed = self.letter2_embedding(letter2_label)       
        encoded = self.encoder(torch.cat([letter1_img, letter1_embed], dim=1))        
        decoded = self.decoder(torch.cat([encoded, letter2_embed], dim=1)) 
        return decoded 