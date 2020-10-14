import os
import matplotlib.pyplot as plt
import numpy as np

def load_cropped():
    
    faces_arr = []

    cropped_dir = "./hw2/data/yalefaces_cropped"
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

    return np.transpose(faces_arr)

def main():

    # load cropped images
    faces_arr = load_cropped()

    # 

if __name__ == "__main__":
    main()