import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from src.variational_autoencoder import get_model
import matplotlib.pyplot as plt

model_path = './models/VAE_epoch_100.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading the model for sampling random faces...")
VAE = get_model(latent_dims = 20, hidden_dims = [32, 64, 64], image_shape = [1, 48, 48])
VAE.load_state_dict(torch.load("./models/VAE_epoch_100.pth", map_location = device))
VAE = VAE.to(device)
print("No issues detected, model loaded successfully.\n\n")
VAE.eval()

samples = VAE.sample(num_samples = 15, device = device)
samples = np.squeeze(samples)

fig, axes = plt.subplots(nrows = 3, ncols = 5, figsize = (10, 5))
k = 0
for i in range(3):
    for j in range(5):
        axes[i][j].imshow(samples[k], cmap = 'gray')
        axes[i][j].axis('off')
        k = k + 1
        
plt.tight_layout(pad = 0.)  
plt.show()