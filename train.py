import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.data import process
from src.dataloader import get_loaders
from src.variational_autoencoder import VAE_v1
from src.utils import plot_reconstruction
import matplotlib.pyplot as plt

model_path = './models/VAE_epoch_100.pth'
data_root = './data/age_gender.csv'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading the data for training...")
train_df, eval_df = process(data_root)
train_dataset, eval_dataset, train_dataloader, eval_dataloader = get_loaders(train_df, eval_df, batch_size = 100, num_workers = 0)
print("Dataframes and Dataloaders are loaded successfully!\n\n")

# Initialize the model
VAE = VAE_v1(latent_dims = 20, hidden_dims = [32, 64, 64], image_shape = [1, 48, 48]).to(device)

# Number of epochs
epochs = 100

# Standard Adam optimizer
optimizer = torch.optim.Adam(VAE.parameters(), lr = 1e-3)

if os.path.exists(model_path):
    VAE.load_state_dict(torch.load(model_path, map_location = device))
    VAE.eval()
    print(f"Loaded existing model from {model_path}. Skipping training.")
else:
    print("No existing model found. Training the model from scratch...")
    VAE.train()
    for epoch in range(epochs):
        for batch in tqdm(train_dataloader, desc = f"Epoch {epoch + 1}/{epochs}", leave = True):
            img = batch['image'].to(device)
            optimizer.zero_grad()
            recons, input, mu, log_var, _ = VAE.forward(img)
            loss = VAE.loss_function(recons, input, mu, log_var)
            loss.backward()
            optimizer.step()
            
        print(f"Epoch: {epoch + 1} \tLoss: {loss:.4f}")
        
        torch.save(VAE.state_dict(), f"./models/VAE_epoch_{epoch + 1}.pth")
        print(f"-> Saved VAE model at epoch {epoch + 1}")
        
        if (epoch + 1) % 10 == 0:
            plot_reconstruction(img, recons)

print("Model training finished, starting interpolation!")
total_z = []

for train_sample in train_dataset:
    img = train_sample['image'].to(device)
    _, _, _, _, z = VAE.forward(img.unsqueeze(0))
    z = z.squeeze().detach().cpu().numpy()
    total_z.append(z)
    
total_z = np.array(total_z)

max_z = total_z.max(axis = 0)
min_z = total_z.min(axis = 0)

# Number of steps required for the interpolation
granularity = 10
latent_space_dims = VAE.get_latent_dims()

fig, axes = plt.subplots(nrows = latent_space_dims, ncols = granularity, figsize = (20, 40))

for i in range(latent_space_dims):
    t = torch.linspace(min_z[i], max_z[i], granularity).to(device)
    z = torch.zeros([1, latent_space_dims]).to(device)
    for j, y in enumerate(t):
        z[0][i] = y
        imgs_decoded = VAE.decode(z).squeeze()
        axes[i][j].imshow(imgs_decoded.detach().cpu().numpy(), cmap = 'gray')
        axes[i][j].axis('Off')

plt.tight_layout(pad = 0.) 
plt.show()
