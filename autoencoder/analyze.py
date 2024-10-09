import torch
from training import DeepAutoencoder

model = DeepAutoencoder()
model.load_state_dict(torch.load('model_weights.0.pth', weights_only=True))
model.eval()  
print(model.state_dict())

# only the second half of the DeepAutoencoder
decoder = model.decoder

# only the first half
encoder = model.encoder


# EXPERIMENT SET 1
# take some image from our dev or train set
# encode it using the encoder, that will output a vector of size 10 (or 32, depending on your version)
# decode it and visualize
# decode some tweaked version (change one element of the vector) and visualize
# repeat for other vector elements
# repeat for other images

# EXPERIMENT SET 2
# take two images from our dev or train set
# encode both
# take the average of the two vectors 
# decode all


# EXPERIMENT SET 3
# do random vector of samples from zero-mean Gaussian
# decode
# enjoy the show







# ('decoder.14.bias', tensor([ 0.0178, -0.0042,  0.0018,  ..., -0.0037, -0.0032,  0.0046],
 #      device='cuda:0'))]