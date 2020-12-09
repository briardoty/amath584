import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from PIL import Image

def load_mnist(train=True):
    
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