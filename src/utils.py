import matplotlib.pyplot as plt
import numpy as np

def plot_reconstruction(img, recons):
    fig, axes = plt.subplots(nrows = 2, ncols = 5, figsize = (10, 5))
    for j in range(5):
        axes[0][j].imshow(np.squeeze(img[j].detach().cpu().numpy()), cmap = 'gray')
        axes[0][j].axis('off')

    for j in range(5):
        axes[1][j].imshow(np.squeeze(recons[j].detach().cpu().numpy()), cmap = 'gray')
        axes[1][j].axis('off')

    plt.tight_layout(pad = 0.)
    plt.show()

