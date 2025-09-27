# Variational Autoencoder GAN for Faces

A PyTorch project combining a **Variational Autoencoder (VAE)** and **GAN** to generate realistic human faces.  
This hybrid approach leverages the reconstruction ability of VAEs and the adversarial training of GANs.

## Project Structure
<pre>
project/
│
├─ data/ # CSV dataset goes here
├─ models/ # Saved checkpoints (VAE_epoch_*.pth)
├─ src/
│ ├─ init.py
│ ├─ data.py
│ ├─ dataloader.py
│ ├─ variational_autoencoder.py
│ └─ utils.py
├─ train.py # Main training & interpolation script
├─ README.md
└─ requirements.txt # Optional for dependencies
</pre>
## Features
- Hybrid VAE-GAN architecture
- Realistic face generation
- Latent space exploration for image variations
- Easy experimentation with network architectures and hyperparameters

## Variational Autoencoder Math
- Encoder outputs mean `μ` and variance `σ²` for the latent space
- Latent vector `z` is sampled as: `z = μ + σ * ε,  ε ~ N(0,1)`
- Loss combines reconstruction loss and KL divergence: `L = E_q(z|x)[log p(x|z)] - KL(q(z|x) || p(z))`

## Usage

1. **Create a virtual environment**

 - **On Linux/Mac:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```
 - **On Windows (Command Prompt):**
  ```bash
  python -m venv venv
  venv\Scripts\activate
```

2. **Install dependencies**

Since I don’t have a `requirements.txt` yet, manually install the main packages:
```bash
pip install torch torchvision tqdm matplotlib pandas numpy
```


- If a checkpoint exists (`./models/VAE_epoch_100.pth`), it will **skip training** and load the model automatically.  
- You should see **interpolations and generated faces** plotted at the end.

3. **Dataset**
The dataset for this implementation can be found at [Kaggle](https://www.kaggle.com/datasets/nipunarora8/age-gender-and-ethnicity-face-data-csv)
-  After downloading the dataset, make a new folder '**data**' and paste the **dataset (.csv)** there.
-  then run python train.py and you shall see interpolations.

## Examples
- Some interpolation examples from the model's output:

<img width="896" height="717" alt="image" src="https://github.com/user-attachments/assets/1c12e178-2fd5-4dbc-b1cd-234fa845988e" />
<img width="898" height="535" alt="image" src="https://github.com/user-attachments/assets/4632ac7d-2c2e-4a07-b5d0-b6f8d546e385" />

- Some constructed faces (random samples):

<img width="634" height="336" alt="image" src="https://github.com/user-attachments/assets/eadf77f3-96e3-4c93-ac3d-3a629595128d" />

## Research Papers
- Kingma, D.P. & Welling, M. (2013). [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)  
- Goodfellow, I. et al. (2014). [Generative Adversarial Nets](https://arxiv.org/abs/1406.2661)  
- Larsen, A. B. L. et al. (2015). [Autoencoding beyond pixels](https://arxiv.org/abs/1512.09300)
