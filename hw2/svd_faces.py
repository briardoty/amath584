import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg

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
            im = plt.imread(filepath)
            im_col = np.reshape(im, -1)
            faces_arr.append(im_col)

    x, y = im.shape
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