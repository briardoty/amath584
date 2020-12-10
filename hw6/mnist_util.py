import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from PIL import Image
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_mapping(X, cmap="bwr"):
    
    vabs = np.max(np.abs(X))
    vmin, vmax = -vabs, vabs
    
    plt.figure(figsize=(8,4))
    im = plt.imshow(X.T, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax)

    cbar.ax.set_ylabel("Coeff")
    cbar.ax.tick_params(labelsize=12)

    ax.set_ylabel("Digit class")
    ax.set_xlabel("Pixel number")

    plt.tight_layout()


