import os
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg
from PIL import Image

def load_faces(cropped=True):
    
    faces_arr = []

    cropped_state = "cropped" if cropped else "uncropped"
    img_dir = f"/home/briardoty/Source/amath584/hw2/data/yalefaces_{cropped_state}"
    for root, _, files in os.walk(img_dir):

        # consider all files...
        for filename in files:
            
            filepath = os.path.join(root, filename)

            try:
                img = Image.open(filepath)
                x, y = img.size
                img = img.resize((int(x/2), int(y/2)), Image.ANTIALIAS)
            except:
                continue

            im_arr = np.asarray(img)
            im_col = np.reshape(im_arr, -1)
            faces_arr.append(im_col)

    x, y = im_arr.shape
    return np.transpose(faces_arr), x, y

def display_top_modes(u, x, y, few=5):
    # plot first few reshaped columns of u
    fig, axes = plt.subplots(nrows=1, ncols=few, figsize=(15,5))
    for i in range(few):
        u_col = u[:,i]
        u_im = np.reshape(u_col, (x,y))
        axes[i].imshow(u_im, cmap="gray")

def summarize_svd_spectrum(s, bins=12, modes=101):
    # plot singular value spectrum
    s = s**2
    s = s * 100 / sum(s)
    fig, ax = plt.subplots()
    ax.bar(range(bins), s[:bins], color="g", alpha=0.7)
    ax.set_xlabel("Singular value mode")
    ax.set_ylabel("Relative variance (%)")

    # interpret sv spectrum
    print(f"The first 1 mode accounts for {s[0]}% of variance.")
    for i in np.arange(2, modes, 2):
        print(f"The first {i} modes account for {sum(s[:i])}% of variance.")

def reconstruct(u, s, vh, faces_arr, x, y, img_idxs, modes_arr):

    s = s**2
    s = s * 100 / sum(s)
    for img_idx in img_idxs:

        # reconstruct
        recos = []
        for modes in modes_arr:
            reco = np.zeros((x,y))
            for i in range(modes):
                mode = np.reshape(u[:,i], (x,y))
                cntr = vh[i, img_idx]

                reco += cntr * mode
            recos.append(reco)

        # visualize
        fig, axes = plt.subplots(nrows=1, ncols=1 + len(recos), figsize=(20,5))
        axes[0].imshow(np.reshape(faces_arr[:,img_idx], (x,y)), cmap="gray")
        axes[0].set_title("Original")
        for i in range(len(recos)):
            reco = recos[i]
            axes[i+1].imshow(reco, cmap="gray")
            axes[i+1].set_title("{:.2f}% reconstruction".format(sum(s[:modes_arr[i]])))

def main():

    # load cropped images
    faces_arr, x, y = load_faces(cropped=False)

if __name__ == "__main__":
    main()
