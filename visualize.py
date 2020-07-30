"""
visualize.py

Written by Sindre Stenen Blakseth, 2020.

Visualizing pickled images and saving them as regular images.
"""

#----------------------------------------------------------------------------
# Package imports.

import imageio
import numpy as np
import pickle
import torch

from os import listdir
from os.path import isfile, join

#----------------------------------------------------------------------------
# File imports.

import config
from evaluate import calculate_metrics

#----------------------------------------------------------------------------
# Load images.

def load_image_dict_from_pickle(file):
    with open(file, 'rb') as f:
        image = pickle.load(f)
    return image

#----------------------------------------------------------------------------
# Save images.

def save_image(image, folderpath, filename):
    file = join(folderpath, filename + '.png')
    imageio.imwrite(file, image)

#----------------------------------------------------------------------------

def visualize(src_path, dest_path):
    # Get all files in folder.
    files = [f for f in listdir(src_path) if isfile(join(src_path, f))]

    # If file is pickle file, load content into image list.
    images  = []
    names   = []
    metrics = {}
    for metric in config.test_metrics:
        metrics[metric] = []
    for file in files:
        if file[-4:] == '.pkl':
            image_dict = load_image_dict_from_pickle(join(src_path, file))
            #print(image_dict)
            torch_hr = torch.from_numpy(np.asarray(image_dict['HR']))
            torch_sr = torch.from_numpy(np.asarray(image_dict['SR']))
            im_metrics = calculate_metrics(torch_hr, torch_sr, config.test_metrics)
            for key in im_metrics.keys():
                print(f"File {file}, metric {key}: {im_metrics[key]}")
            for key in image_dict.keys():
                images.append(image_dict[key])
                names.append(file[:-4] + '_' + key)

    # Process and save all images in image list
    for i, image in enumerate(images):
        image = np.moveaxis(image, 0, -1)
        image *= 255
        image = image.astype(np.uint8)
        save_image(image, dest_path, names[i])

#----------------------------------------------------------------------------

if __name__ == '__main__':
    visualize(config.eval_im_dir, config.eval_im_dir)

#----------------------------------------------------------------------------