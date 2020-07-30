# ESRGAN Implementation i PyTorch
Written by Sindre Stenen Blakseth, June/July 2020.  
Based on the work of:
- Thomas Nakken Larsen
- Duy Tan Huynh Tran
- Eirik Vesterkjær

With guidance from:
- Adil Rasheed
- Trond Kvamsdal

## Disclaimer
This implementation does not produce satisfactory results at the moment. This could be due to any or all of the discrepancies between this work and the original ESRGAN (see below), or some other factor unknown to the author.

## Using the ESRGAN
1. All configuration should be done in config.py. All output files are saved to the specified run_dir or one of its subdirectories.
2. Use main.py for training, testing and dataset management. main.py support the following command line options:
    - --download            (Download raw data.)
    - --dataset             (Create new datasets from raw data.)
    - --train               (Train ESRGAN.)
    - --test                (Test pre-trained ESRGAN.)
3. Use plot_results.py and visualize.py 

## Outputs
By default, the model prints its training losses every 100 iterations to a dedicated .txt-file. Furthermore, every 2000 iterations, it prints the following to two separate .txt-files:

1. Validation losses (file 1).
2. LPIPS and PSNR scores (file 1).
3. Discriminator prediction accuracy on the validation set (file 1).
4. Mean, mean of absolute value, and variance of gradients at select layers in both networks (file 2).
5. Mean of absolute value, and variance of weights at select layers in both networks (file 2).

Every 10 000 iterations, the model stores select LR, HR and SR examples in pickle-files, and also stores the discriminator predictions on these SR and HR examples in a dedicated .txt-file.

The outputing intervals can be changed in the configuration file.

Tensorboard is not used in the current implementation.

## Contents
This code base contains the following files and folders:
- data_raw/                 - Folder for storing raw data.
- datasets/                 - Folder for storing assembled datasets.
- metrics/                  - Folder containing implementations of performance evaluation metrics.
    - metrics.py            - Implements metrics for evaluating the performance of ESRGAN during testing.
    - LPIPS/                - Folder containing the official LPIPS implementation: https://github.com/richzhang/PerceptualSimilarity
- models/                   - Folder containing the model implementation.
    - esrgan.py             - Implements the basic structure of ESRGAN.
    - initialization.py     - Implements initialization helper functions.
    - losses.py             - Implements ESRGAN's loss functions.
    - networks.py           - Implements ESRGAN's generator and discriminator.
- results/                  - Folder for storing results.
- config.py                 - Global configuration file.
- datasets.py               - Implements functions for creating datasets from raw simulation data.
- download_data.py          - (not public) Implements functions for downloading and processing HARMONIE-SIMRA simulation data.
- evaluate.py               - Implements functions for validating and testing ESRGAN.
- main.py                   - Main entry point for training, testing and using ESRGAN, and for managing data.
- plot_results.py           - Script for plotting results written to .txt-files during training.
- README.md                 - General documentation of the code base.
- train.py                  - Implements the training loop of ESRGAN.
- visualize.py              - Script for converting images stored as pickle-files during validation to regular .png-images.

### metrics/metrics.py
Implements functions for retrieving metric scores which can be used for evaluating network performance when testing.
Currently, the following metrics are supported:
- PSNR.
- LPIPS (relies on code in the directory metrics/LPIPS/).

### models/esrgan.py
- Builds the model's generator     according to the configurations specified in config.py.
- Builds the model's discriminator according to the configurations specified in config.py.
- Combines the generator and the discriminator in one model.
- Defines dictionaries which can be used for storing training results.
- Defines the networks' optimizers.
- Defines the networks' learning rate schedulers.
- Implements functionality for saving and loading models.

### models/initialization.py
- Implements Kaiming initialization (whatever that is ...) of model weights. Is it a good idea to use Kaiming init?

### models/losses.py
- Implements functions for calculating the losses of the model's generator and discriminator.
The losses are calculated as linear combinations of different loss terms.
The scaling parameter of each loss term is defined in the configuration file.
- Currently, the following loss terms can be used for the generator loss:
    - Adversarial losses:
        - Standard DCGAN loss.
        - Relativistic GAN loss.
        - Relativistic average GAN loss.
    - Feature losses (The features are extracted using a pre-trained VGG19 network.
      They can be extracted at one (high) level or two (low and high) levels.
      They number of the layer(s) from which features are gathered are specified in config.py.
      config.py also contains variables for toggling the use of input normalization and
      batch normalization in the feature extraction network.):
        - L1 distance between HR image features and SR image features.
        - L2 distance between HR image features and SR image features.
    - Pixel losses:
        - L1 distance between HR image and SR image.
        - L2 distance between HR image and SR image.
- Currently, the following loss terms can be used for the discriminator loss:
    - Adversarial losses:
        - Standard DCGAN loss.
        - Relativistic GAN loss.
        - Relativistic average GAN loss.
        
### models/networks.py
- Implements the architecture of the model's generator.
- Implements the architecture of the model's discriminator.

### config.py
Global configuration file offering options for:
- Environment
    - Path to directory containing this file and the rest of the ESRGAN implementation.
    - Path to directory for storing datasets.
    - Path to directory for storing raw data.
    - Path to directory for storing results.
    - Path to directory for storing the current run's results.
    - Path to directory for storing tensorboard information.
    - Path to directory for storing checkpoints.
    - Path to directory for storing images resulting from evaluation of model.
    - Path to directory for storing metric values resulting from evaluation of model. UNUSED
- Device
    - Current device.
    - ID of GPU to use.
- Run
    - Toggle training.
    - Toggle testing.
    - Loading pre-trained model (toggle variables and paths).
- Training
    - Number of training iterations.
    - Number of discriminator training iterations per generator training iteration.
    - Loss printing period.
    - Model saving period.
    - Configuration of generator's     Adam optimizer.
    - Configuration of discriminator's Adam optimizer.
    - Learning rate schedule configuration. (Currently, the same scheduling is used for both generator and discriminator.)
    - Toggle label flips.
    - Toggle one-sided label smoothing.
    - Training batch size.
- Validation
    - Validation period.
    - Period of validations with visualization of validation images.
    - The number of validation images to visualize at each validation visualization.
    - A list containing the names of interpolation techniques to compare with when doing validation visualization.
    - Validation batch size.
- Testing
    - A list containing the names of interpolation techniques to compare with when doing testing.
    - A list containing the names of metrics to use for evaluating network performance quantitatively.
    - Testing batch size.
- Data
    - Code for accessing raw data on external server.
    - First date for which raw data is retrieved.
    - Last  date for which raw data is retrieved.
    - Data tag used in result file names.
    - HR image dimensions. (Cannot be changed without altering the discriminator architecture and the dataset creation procedure.)
    - LR image dimensions.
    - Upscaling factor.
- Discriminator
    - Architecture.
    - Activation function.
    - Weight initialization.
    - Loss function (given as a dictionary where the keys are the names of the terms to include
    and the values are their corresponding scaling factors).
- Generator
    - Architecture (including the number of RRDBs to use).
    - Activation function.
    - Weight initialization.
    - Loss function (given as a dictionary where the keys are the names of the terms to include
    and the values are their corresponding scaling factors).
- Feature extractor (VGG19)
    - Layer from which to extract low  level features (can be disabled by setting None or a negative value).
    - Layer from which to extract high level features.
    - Toggle batch normalization in feature extractor.
    - Toggle (standard RGB) input normalization in feature extractor.
    
The configuration file also ensures that all configurations are saved to a text file located in the current run directory.

### datasets.py
- Creates a dataset from raw data by
    - Reading u,v and w from a pickle file.
    - Resizing to 128x128. (This is hard-coded to work for the Bessaker data only.)
    - Replacing masked values with Nan.
    - Converting from numpy-array to pytorch-tensor.
    - Splitting data into training set, validation set and testing set.
    - Create datasets from the split data.
- Can save a dataset to   file.
- Can load a dataset from file.

### download_data.py
- Download Bessaker data.
- Extract u,v and w from a single vertical plane in the computational domain (not necessarily a plane in the real domain).

### evaluate.py
- Evaluates the performance of a model on a given dataset by
    - ... calculating losses for the images in the dataset.
    - ... possibly (if testing or visualizing validation) saving the images
    (in LR and HR) in the dataset along with
    SR images created by the model and SR images created using
    a set of given interpolation techniques (specified in config.py).
    - ... possibly (if testing) calculate a set of metrics specified in config.py.

### main.py
Main entry point for this ESRGAN implementation. By calling functions in elsewhere in the code base, this script handles:
- Data download.
- Dataset creation.
- Training.
- Testing.

### plot_results.py
Plots the following results stored to .txt-files during training/validation:
- Generator training loss.
- Generator validation loss.
- Discriminator training loss.
- Discriminator validation loss.
- LPIPS score on validation set.
- PSNR score on validation set.
- Discriminator accuracy on real and fake data of the validation set.
- Mean of gradients in the first, the last and an intermediate layer of the generator.
- Mean of absolute value of gradients in the first, the last and an intermediate layer of the generator.
- Variance of gradients in the first, the last and an intermediate layer of the generator.
- Mean of gradients in the first and the last layer of the discriminator.
- Mean of absolute value of gradients in the first and the last layer of the discriminator.
- Variance of gradients in the first and the last layer of the discriminator.
- Mean of absolute value of weights in the first, the last and an intermediate layer of the generator.
- Variance of weights in the first, the last and an intermediate layer of the generator.
- Mean of absolute value of weights in the first and the last layer of the discriminator.
- Variance of weights in the first and the last layer of the discriminator.

### train.py
Implements the ESRGAN training loop. This includes:
- Handling label augmentation (e.g. label smoothing).
- Creating data loaders.
- Performing forward pass.
- Calling the loss calculations functions defined in models/losses.py.
- Performing backward pass.
- Updating learning rate.
- Saving model at given intervals.
- Performing validation by calling the function evaluate() in evaluate.py at given intervals.

### visualize.py
Stores pickled images from validation as .png-images.

## Packages used
- argparse
- collections  (LPIPS)
- datetime     (SIMRA data download)
- fractions    (LPIPS)
- functools    (LPIPS)
- imageio
- IPython      (LPIPS)
- itertools    (LPIPS)
- math
- matplotlib
- netcdf4      (SIMRA data download)
- numpy
- os
- pdb          (LPIPS)
- pickle
- PIL
- random
- scipy        (LPIPS)
- skimage      (LPIPS)
- sys          (LPIPS)
- tensorboardX (currently unused)
- torch
- torchvision
- tqdm         (LPIPS)
- urllib       (SIMRA data download)

## Default configuration

Most of the configurations are lifted directly from the ESRDGAN implementation of Eirik Vesterkjær.
These differ from those of the original ESRGAN in a number of ways, though no justification of these changes is know to the author of this implementation.
The discrepancies between this repo and the original ESRGAN are noted below.
Anyone interested in using this model for research should first check if the original ESRGAN parameter choices yield better results than the current configuration when training on normal images.

By default, the following configurations apply:
- Model uses GPU if available.
- Model is in training mode, not in testing mode.
- Model is trained from scratch.
- Training runs for 200 000 iterations (**Original ESRGAN uses more than 300 000**), and the generator is trained every 2nd iteration.
- Losses are printed every 100 iterations.
- The model is saved every 50 000 iterations.
- Validation losses are computed every 2000 iterations.
- Visualization of validation data is performed every 10 000 iterations,
at which point 8 validation images are visualized.
For comparison, bicubic interpolation is also applied to create SR images from the LR validation images.
- Labels are **not** flipped.
- Noisy labels are **not** used.
- Label smoothing is **not** used.
- Instance noise is **not** used.
- All batch sizes are 8. **Original ESRGAN uses 16.**
- Both optimizers use learning rate 1e-4, 0 weight decay and betas 0.9 and 0.999.
- Both learning rate schedulers use gamma 0.5 and steps every 50 000 iterations.
- The metrics LPIPS and PSNR are used during validation and testing.
- For comparison, bicubic interpolation is used to create SR images from all images in the test set.
- Data tag is the name I used for a custom dataset containing roughly 15k images from the Flickr_30k-dataset.
- LR images are 32x32 and HR images are 128x128.
- Generator and discriminator both use leaky relu activation with slope 0.2.
- Generator uses 16 RRDBs with 32 growth channels. **Original ESRGAN uses 32 RRDBs.**
- Discriminator uses the following loss term(s):
    - RaGAN adversarial loss, scaling factor 1.0.
- Generator uses the following loss term(s):
    - RaGAN adversarial loss, scaling factor 5e-3
    - L1 pixel loss, scaling factor 1e-2
    - L1 feature loss, scaling factor 1.0
- Features for the feature loss are extracted from layer 34 of a pre-trained VGG19 network.
- The feature extractor uses standard RGB input normalization.
- The feature extractor does not use batch normalization.
- Initialization scaling parameter for the generator is 0.5. **Original ESRGAN uses 0.1.**
- Initialization scaling parameter for the discriminator is 0.1. **Original ESRGAN value is unknown.**
- Generator uses 128 internal feature channels. **Original ESRGAN uses 64.**
- HR-conv layers in G use a kernel size of 5. **Original ESRGAN uses 3.**
- LeakyReLU with a negative slope of 0.2 is used throughout both the generator and the discriminator.
- The discriminator uses BatchNorm2d with keep_running_stats set to False.