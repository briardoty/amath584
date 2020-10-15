import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from PIL import Image

def load_cropped():
    
    faces_arr = []

    cropped_dir = "/home/briardoty/Source/amath584/hw2/data/yalefaces_cropped"
    for root, _, files in os.walk(cropped_dir):

        # consider all files...
        for filename in files:

            # ...as long as they are .pgm images
            if not filename.endswith(".pgm"):
                continue
            
            filepath = os.path.join(root, filename)
            img = Image.open(filepath)
            x, y = img.size
            img = img.resize((int(x/2), int(y/2)), Image.ANTIALIAS)

            im_arr = np.asarray(img)
            im_col = np.reshape(im_arr, -1)
            faces_arr.append(im_col)

    x, y = im_arr.shape
    return np.transpose(faces_arr), x, y

def main():

    # load cropped images
    faces_arr, x, y = load_cropped()

    # svd
    u, s, vh = linalg.svd(faces_arr)

    # plot first few reshaped columns of u

    x = 1

if __name__ == "__main__":
    main()