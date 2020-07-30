"""
datasets.py

Written by Duy Tan Huynh Tran and Thomas Nakken Larsen.
Modified by Sindre Stenen Blakseth, 2020.

Creating training, validation and test datasets for ESRGAN using HARMONIE-SIMRA simulation data.
"""

#----------------------------------------------------------------------------
# Package imports

import numpy as np
import os
import pickle
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data

from PIL import Image
from torchvision import transforms

#----------------------------------------------------------------------------
# File imports

import config

#----------------------------------------------------------------------------
# Data configurations.

raw_data_filename = config.data_tag + '.pkl'
raw_data_location = config.raw_data_dir
datasets_location = config.datasets_dir

#----------------------------------------------------------------------------
# Normalization/standardization helper functions.

def get_normalization_factors(dataset):
    """
    Purpose: Calculate channel-wise maximum and minimum values across dataset.
    Args:
        dataset: Dataset for which channel-wise maximum and minimum values should be computed.
    Returns:
        d_min: Channel-wise minimum across dataset.
        d_max: Channel-wise maximum across dataset.
    """

    dataset = np.array(dataset)

    d_min = np.zeros(dataset.shape[1])
    d_max = np.zeros(dataset.shape[1])
    for ch in range(dataset.shape[1]):
        d_min[ch] = np.amin(dataset[:, ch, :, :])  # Global minimum
        d_max[ch] = np.amax(dataset[:, ch, :, :])  # Global maximum

    #print("Calculated denorm. factors: (d_min=", d_min, " & d_max=", d_max, ")", sep="")
    return d_min, d_max

def normalize(dataset, d_min, d_max, low=0, high=1):
    """
    Purpose: # Normalize (center and scale) data to specified interval, channel-wise.
    Args:
        dataset: Dataset to normalize.
        d_min: Minimum value of each channel of dataset.
        d_max: Maximum value of each channel of dataset.
        low: Low bound of specified interval.
        high: High bound of specified interval.
    Returns:
        Normalized version of dataset.
    """
    dataset = np.array(dataset)  # Can't use torch.Tensor

    # Channel-wise normalization
    for ch in range(dataset.shape[1]):
        dataset[:, ch, :, :] = (high - low) * (dataset[:,ch,:,:] - d_min[ch]) / (d_max[ch] - d_min[ch]) + low

    return torch.from_numpy(dataset)


# NOT USED
def standardize(dataset):
    """
    Purpose: Subtracts mean and divides by standard deviation, channel-wise
    Args:
        dataset: Dataset to standardize.
    Returns:
        Standardized version of dataset.
    """
    mean = dataset.mean(dim=[0,2,3])  # returns list of means for each channel
    std  = dataset.std(dim=[0,2,3])   # returns list of stds  for each channel

    standardization = transforms.Normalize(mean=mean, std=std, inplace=True)
    for i in range(len(dataset)):
        standardization(dataset[i])

    return dataset

#----------------------------------------------------------------------------
# Create training, validation and test datasets from HARMONIE-SIMRA data.

def create_datasets():
    #with open('/lustre1/work/sindresb/code_base/sommerjobb2020/ESRGAN/data_raw/april2018_iz39.pkl', 'rb') as f:
    #    u, v, w = pickle.load(f)
    with open(os.path.join(raw_data_location, raw_data_filename), 'rb') as f:
        u, v, w = pickle.load(f)

    # Replace masked data with NaN-values.
    u_nan = np.ma.filled(u.astype(float), np.nan)
    v_nan = np.ma.filled(v.astype(float), np.nan)
    w_nan = np.ma.filled(w.astype(float), np.nan)

    # Remove some rows and cols to make the dimensions powers of 2, e.g. 128 x 128.
    u_nomask = u_nan[:, 4:-4, 4:-3]
    v_nomask = v_nan[:, 4:-4, 4:-3]
    w_nomask = w_nan[:, 4:-4, 4:-3]

    assert type(u_nomask) is np.ndarray, "input u is not np.ndarray: type is %r" % type(u_nomask)
    assert type(v_nomask) is np.ndarray, "input v is not np.ndarray: type is %r" % type(v_nomask)
    assert type(w_nomask) is np.ndarray, "input w is not np.ndarray: type is %r" % type(w_nomask)

    # Transform the HR data into tensor form.
    u_tensor = torch.from_numpy(u_nomask[:, np.newaxis, :, :])
    v_tensor = torch.from_numpy(v_nomask[:, np.newaxis, :, :])
    w_tensor = torch.from_numpy(w_nomask[:, np.newaxis, :, :])

    # Concatenate the tensors together like the channels of an RGB-image.
    HR_data = torch.cat((u_tensor, v_tensor, w_tensor), dim=1)
    assert HR_data.shape[2] == config.HR_size, "HR-dimensions are incorrect."
    assert HR_data.shape[3] == config.HR_size, "HR-dimensions are incorrect."

    # Create LR data from HR data.
    LR_data = F.interpolate(HR_data, size = config.LR_size, mode = 'nearest')
    assert LR_data.shape[2] == config.LR_size, "LR-dimensions are incorrect."
    assert LR_data.shape[3] == config.LR_size, "LR-dimensions are incorrect."
    print("HR_data.shape:", HR_data.shape)  # output = dim ( it, 3, 128, 128 )
    print("LR_data.shape:", LR_data.shape)  # output = dim ( it, 3,  32,  32 )

    # Creating training, validation, test split.
    train_frac = 0.8
    num_train_samples = int(HR_data.size(0) * train_frac)
    num_val_samples   = int(HR_data.size(0) * (1 - train_frac) / 2)

    # Calculate normalization factors from the training dataset.
    d_min, d_max = get_normalization_factors(HR_data[:num_train_samples])
    print("normalization factors are: \nd_max=", d_max, "\nd_min=", d_min)

    # Separate HR and LR training data and normalize.
    HR_data_train = normalize(HR_data[:num_train_samples], d_min=d_min, d_max=d_max)
    LR_data_train = normalize(LR_data[:num_train_samples], d_min=d_min, d_max=d_max)

    # Separate HR and LR validation data and normalize.
    HR_data_val = normalize(HR_data[num_train_samples:num_val_samples + num_train_samples], d_min=d_min, d_max=d_max)
    LR_data_val = normalize(LR_data[num_train_samples:num_val_samples + num_train_samples], d_min=d_min, d_max=d_max)

    # Separate HR and LR test data and normalize.
    HR_data_test = normalize(HR_data[num_val_samples + num_train_samples:], d_min=d_min, d_max=d_max)
    LR_data_test = normalize(LR_data[num_val_samples + num_train_samples:], d_min=d_min, d_max=d_max)

    # Create datasets.
    dataset_train = torch.utils.data.TensorDataset(LR_data_train, HR_data_train)
    dataset_val   = torch.utils.data.TensorDataset(LR_data_val,   HR_data_val)
    dataset_test  = torch.utils.data.TensorDataset(LR_data_test,  HR_data_test)

    print("Datasets complete:")
    print("dataset_train.shape:", dataset_train.__len__())
    print("dataset_val.shape:", dataset_val.__len__())
    print("dataset_test.shape:", dataset_test.__len__())

    return dataset_train, dataset_val, dataset_test

#----------------------------------------------------------------------------
# Create a code verification dataset with 100 all-black images.

def create_colour_datasets(colour: str = 'black'):
    # Create 3 same-coloured numpy arrays.
    scale = 0
    if colour == 'black':
        pass
    elif colour == 'white':
        scale = 1.0
    elif colour == 'grey':
        scale = 0.5

    u = np.ones((100, 128, 128))*scale
    v = np.ones((100, 128, 128))*scale
    w = np.ones((100, 128, 128))*scale

    # Transform the HR data into tensor form.
    u_tensor = torch.from_numpy(u[:, np.newaxis, :, :])
    v_tensor = torch.from_numpy(v[:, np.newaxis, :, :])
    w_tensor = torch.from_numpy(w[:, np.newaxis, :, :])

    # Concatenate the tensors together like the channels of an RGB-image.
    HR_data = torch.cat((u_tensor, v_tensor, w_tensor), dim=1)
    assert HR_data.shape[2] == config.HR_size, "HR-dimensions are incorrect."
    assert HR_data.shape[3] == config.HR_size, "HR-dimensions are incorrect."

    # Create LR data from HR data.
    LR_data = F.interpolate(HR_data, size = config.LR_size, mode = 'nearest')
    assert LR_data.shape[2] == config.LR_size, "LR-dimensions are incorrect."
    assert LR_data.shape[3] == config.LR_size, "LR-dimensions are incorrect."
    print("HR_data.shape:", HR_data.shape)  # output = dim ( it, 3, 128, 128 )
    print("LR_data.shape:", LR_data.shape)  # output = dim ( it, 3,  32,  32 )

    # Creating training, validation, test split.
    train_frac = 0.8
    num_train_samples = int(HR_data.size(0) * train_frac)
    num_val_samples   = int(HR_data.size(0) * (1 - train_frac) / 2)

    # Separate HR and LR training data.
    HR_data_train = HR_data[:num_train_samples]
    LR_data_train = LR_data[:num_train_samples]

    # Separate HR and LR validation data.
    HR_data_val = HR_data[num_train_samples:num_val_samples + num_train_samples]
    LR_data_val = LR_data[num_train_samples:num_val_samples + num_train_samples]

    # Separate HR and LR test data.
    HR_data_test = HR_data[num_val_samples + num_train_samples:]
    LR_data_test = LR_data[num_val_samples + num_train_samples:]

    # Create datasets.
    dataset_train = torch.utils.data.TensorDataset(LR_data_train, HR_data_train)
    dataset_val   = torch.utils.data.TensorDataset(LR_data_val,   HR_data_val)
    dataset_test  = torch.utils.data.TensorDataset(LR_data_test,  HR_data_test)

    print("Datasets complete:")
    print("dataset_train.shape:", dataset_train.__len__())
    print("dataset_val.shape:", dataset_val.__len__())
    print("dataset_test.shape:", dataset_test.__len__())

    return dataset_train, dataset_val, dataset_test

#----------------------------------------------------------------------------
# Create gradient dataset.

def create_gradient_dataset():
    u = np.ones((100, 128, 128)) * np.linspace(0, 1, 128)
    v = np.ones((100, 128, 128)) * np.linspace(0, 1, 128)
    w = np.ones((100, 128, 128)) * np.linspace(0, 1, 128)

    print(u)

    # Transform the HR data into tensor form.
    u_tensor = torch.from_numpy(u[:, np.newaxis, :, :])
    v_tensor = torch.from_numpy(v[:, np.newaxis, :, :])
    w_tensor = torch.from_numpy(w[:, np.newaxis, :, :])

    # Concatenate the tensors together like the channels of an RGB-image.
    HR_data = torch.cat((u_tensor, v_tensor, w_tensor), dim=1)
    assert HR_data.shape[2] == config.HR_size, "HR-dimensions are incorrect."
    assert HR_data.shape[3] == config.HR_size, "HR-dimensions are incorrect."

    # Create LR data from HR data.
    LR_data = F.interpolate(HR_data, size=config.LR_size, mode='nearest')
    assert LR_data.shape[2] == config.LR_size, "LR-dimensions are incorrect."
    assert LR_data.shape[3] == config.LR_size, "LR-dimensions are incorrect."
    print("HR_data.shape:", HR_data.shape)  # output = dim ( it, 3, 128, 128 )
    print("LR_data.shape:", LR_data.shape)  # output = dim ( it, 3,  32,  32 )

    # Creating training, validation, test split.
    train_frac = 0.8
    num_train_samples = int(HR_data.size(0) * train_frac)
    num_val_samples = int(HR_data.size(0) * (1 - train_frac) / 2)

    # Separate HR and LR training data.
    HR_data_train = HR_data[:num_train_samples]
    LR_data_train = LR_data[:num_train_samples]

    # Separate HR and LR validation data.
    HR_data_val = HR_data[num_train_samples:num_val_samples + num_train_samples]
    LR_data_val = LR_data[num_train_samples:num_val_samples + num_train_samples]

    # Separate HR and LR test data.
    HR_data_test = HR_data[num_val_samples + num_train_samples:]
    LR_data_test = LR_data[num_val_samples + num_train_samples:]

    # Create datasets.
    dataset_train = torch.utils.data.TensorDataset(LR_data_train, HR_data_train)
    dataset_val = torch.utils.data.TensorDataset(LR_data_val, HR_data_val)
    dataset_test = torch.utils.data.TensorDataset(LR_data_test, HR_data_test)

    print("Datasets complete:")
    print("dataset_train.shape:", dataset_train.__len__())
    print("dataset_val.shape:", dataset_val.__len__())
    print("dataset_test.shape:", dataset_test.__len__())

    return dataset_train, dataset_val, dataset_test

#----------------------------------------------------------------------------
# Create dataset from flickr30.

def create_flickr_dataset():
    num_train_im = 15500
    num_val_im =   1900
    num_test_im =  1900
    n = num_train_im + num_val_im + num_test_im

    src_path = os.path.join(config.raw_data_dir, 'flickr_30k', '31296_39911_bundle_archive', 'flickr30k_images', 'flickr30k_images')
    files = [f for f in os.listdir(src_path) if os.path.isfile(os.path.join(src_path, f)) and f[-4:] == '.jpg']

    images = []
    for i, file in enumerate(files):
        if i == n:
            break
        # Get file path.
        img_path = os.path.join(src_path, file)
        # Open image.
        image = Image.open(img_path)
        # Crop image.
        image_cropped = image.crop((0, 0, config.HR_size, config.HR_size))
        # Convert image to tensor.
        image_tensor = transforms.ToTensor()(image_cropped) # unsqueeze to add artificial first dimension
        images.append(image_tensor)
    # Convert list of tensors to tensor.
    HR_data = torch.stack(images)
    # Debug prints.
    print("Shape:", HR_data.size())
    print("Max:", torch.max(HR_data[0]))

    LR_data = F.interpolate(HR_data, size=config.LR_size, mode='nearest')

    # Separate HR and LR training data.
    HR_data_train = HR_data[:num_train_im]
    LR_data_train = LR_data[:num_train_im]

    # Separate HR and LR validation data.
    HR_data_val = HR_data[num_train_im:num_train_im + num_val_im]
    LR_data_val = LR_data[num_train_im:num_train_im + num_val_im]

    # Separate HR and LR test data.
    HR_data_test = HR_data[num_train_im + num_val_im:]
    LR_data_test = LR_data[num_train_im + num_val_im:]

    # Create datasets.
    dataset_train = torch.utils.data.TensorDataset(LR_data_train, HR_data_train)
    dataset_val = torch.utils.data.TensorDataset(LR_data_val, HR_data_val)
    dataset_test = torch.utils.data.TensorDataset(LR_data_test, HR_data_test)

    print("Datasets complete:")
    print("dataset_train.shape:", dataset_train.__len__())
    print("dataset_val.shape:", dataset_val.__len__())
    print("dataset_test.shape:", dataset_test.__len__())

    return dataset_train, dataset_val, dataset_test

#----------------------------------------------------------------------------
# Writing and loading datasets to/from disk.

def save_datasets(dataset_train, dataset_val, dataset_test):
    torch.save(dataset_train, os.path.join(datasets_location, config.data_tag + '_train.pt'))
    print("Saved training set to:", os.path.join(datasets_location, config.data_tag + '_train.pt'))
    torch.save(dataset_val,   os.path.join(datasets_location, config.data_tag + '_val.pt'  ))
    print("Saved validation set to:", os.path.join(datasets_location, config.data_tag + '_val.pt'))
    torch.save(dataset_test,  os.path.join(datasets_location, config.data_tag + '_test.pt' ))
    print("Saved test set to:", os.path.join(datasets_location, config.data_tag + '_test.pt'))

def load_datasets():
    dataset_train = torch.load(os.path.join(datasets_location, config.data_tag + '_train.pt'))
    dataset_val   = torch.load(os.path.join(datasets_location, config.data_tag + '_val.pt'  ))
    dataset_test  = torch.load(os.path.join(datasets_location, config.data_tag + '_test.pt' ))
    return dataset_train, dataset_val, dataset_test

#----------------------------------------------------------------------------

def main():
    if config.data_tag == 'all_black':
        dataset_train, dataset_val, dataset_test = create_colour_datasets(colour = 'black')
    elif config.data_tag == 'all_grey':
        dataset_train, dataset_val, dataset_test = create_colour_datasets(colour = 'grey')
    elif config.data_tag == 'gradient':
        dataset_train, dataset_val, dataset_test = create_gradient_dataset()
    elif config.data_tag == 'flickr_15k':
        dataset_train, dataset_val, dataset_test = create_flickr_dataset()
    else:
        dataset_train, dataset_val, dataset_test = create_datasets()
    save_datasets(dataset_train, dataset_val, dataset_test)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------