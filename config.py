"""
config.py

Written by Sindre Stenen Blakseth, 2020

Main configuration file for ESRGAN implementation.
"""

#----------------------------------------------------------------------------
# Package imports.

import os
import torch

#----------------------------------------------------------------------------
# Environment configuration.

base_dir     = '/lustre1/work/sindresb/code_base/sommerjobb2020/ESRGAN'
#base_dir     = '/home/sindre/sommerjobb/git/ESRGAN'
datasets_dir = os.path.join(base_dir,    'datasets')
raw_data_dir = os.path.join(base_dir,    'data_raw')
results_dir  = os.path.join(base_dir,    'results')
run_dir      = os.path.join(results_dir, 'esrgan_exp8_run4')
tb_dir       = os.path.join(results_dir, 'tensorboard')
tb_run_dir   = os.path.join(tb_dir,       os.path.basename(os.path.normpath(run_dir)))
cp_load_dir  = os.path.join(results_dir, 'exp_loss_equal/checkpoints')
cp_save_dir  = os.path.join(run_dir,     'checkpoints')
eval_im_dir  = os.path.join(run_dir,     'images')
metrics_dir  = os.path.join(run_dir,     'metric_data')

training_data_file    = os.path.join(run_dir, 'training_data.txt')
val_loss_metrics_file = os.path.join(run_dir, 'val_loss_metrics.txt')
val_pred_file         = os.path.join(run_dir, 'val_D_predicts.txt')
val_w_and_grad_file   = os.path.join(run_dir, 'val_w_and_grads.txt')

#----------------------------------------------------------------------------
# Device configuration.

gpu_id = 0
device = torch.device("cuda" if (torch.cuda.is_available() and gpu_id is not None) else "cpu")

#----------------------------------------------------------------------------
# Run configuration.

is_train = True
is_test  = False

load_model_from_save      = False
resume_training_from_save = False
generator_load_path       = None #os.path.join(cp_load_dir, 'G_150000.pth')
discriminator_load_path   = None #os.path.join(cp_load_dir, 'D_150000.pth')
state_load_path           = None #os.path.join(cp_load_dir, 'state_975.pth')

#----------------------------------------------------------------------------
# Training configuration.

# TODO: Add learning rate schedule configurations?

num_train_it    = 200000         # Number of training iterations.
d_g_train_ratio = 2             # Number of training iterations for discriminator per training iteration of generator.

print_train_loss_period = 100    # Number of training iterations per print of training losses.
save_model_period       = 50000  # Number of training iterations per model save.

opt_G_lr      = 1e-4            # Learning rate of generator.
opt_G_w_decay = 0               # Weight decay of generator's Adam optimizer.
opt_G_betas   = (0.9, 0.999)    # Beta values of generator's Adam optimizer.

opt_D_lr      = 1e-4            # Learning rate of discriminator.
opt_D_w_decay = 0               # Weight decay of discriminator's Adam optimizer.
opt_D_betas   = (0.9, 0.999)    # Beta values of discriminator's Adam optimizer.

lr_gamma = 0.5                  # Multiplicative factor of learning rate decay.
lr_steps = [50000, 100000,      # Iterations at which current learning rates are multiplied by lr_gamma.
            150000]

flip_labels    = False          # Toggle label flips.
smooth_labels  = False          # Toggle one-sided label smoothing.
noisy_labels   = False          # Toggle noisy labels.
use_inst_noise = False          # Toggle use of instance noise.

train_batch_size          = 8   # Batch size for training.
dataset_train_num_workers = 0   # TODO: Don't know what this is for.

#----------------------------------------------------------------------------
# Validation configuration.

val_period             = 2000   # Number of training iterations per validation.
val_visual_period      = 10000  # Number of training iterations per visualization of validation data.
num_val_visualizations = 8      # Number of images to visualize when visualizing validation data.

val_comparisons = ['bicubic']   # Interpolation techniques to compare with when visualizing validation data.
val_metrics     = ['psnr',
                   'lpips']     # Metrics to use for evaluating model performance on validation data.

val_batch_size          = 8     # Batch size for validation.
dataset_val_num_workers = 0     # TODO: Don't know what this is for.

#----------------------------------------------------------------------------
# Testing configuration.

test_comparisons = ['bicubic']  # Interpolation techniques to compare with when visualizing test data.
test_metrics     = ['psnr',
                    'lpips']    # Metrics to use for evaluating model performance on test data.

test_batch_size  = 8            # Batch size for testing.
test_num_workers = 0            # TODO: Don't know what this is for.

#----------------------------------------------------------------------------
# Data configuration.

data_code   = 'simra_BESSAKER_' # Data code for accessing raw data on external server.
start_year  = 2017              # Year  of first date for which raw data is retrieved.
start_month = 8                 # Month of first date for which raw data is retrieved.
start_day   = 4                 # Day   of first date for which raw data is retrieved.
end_year    = 2019              # Year  of last  date for which raw data is retrieved.
end_month   = 10                # Month of last  date for which raw data is retrieved.
end_day     = 30                # Day   of last  date for which raw data is retrieved.
altitude    = 39                # Altitude level at which raw data is retrieved.
data_tag    = 'flickr_15k'       # Tag describing the retrieved raw data
HR_size     = 128               # High resolution (HR) image height and width.
LR_size     = 32                # Low resolution  (LR) image height and width.
upscale     = 4                 # Scaling factor needed for making HR data from LR data.

#----------------------------------------------------------------------------
# Discriminator configuration.

D_in_num_ch         = 3         # Number of input  channels of discriminator's first layer.
D_base_num_f        = 128       # Number of output channels of discriminator's first layer.
D_feat_kern_size    = 3         # Kernel size of discriminator's convolution layers.
D_norm_type         = 'batch'   # Type of normalization used in discriminator's normalization layers.
D_act_type          = 'lrelu'   # Type of activation function used in discriminator.
D_weight_init_scale = 0.1       # Scale Kaiming initialization weights by this factor during init of D.
D_loss_scales = {               # Factors for scaling different contributions to the total discriminator loss.
    'rel_avg': 1.0
} # Losses not included as keys in this dictionary will not be used for training discriminator.

#----------------------------------------------------------------------------
# Generator configuration.

G_in_num_ch         = 3         # Number of input  channels of generator's first layer.
G_out_num_ch        = 3         # Number of output channels of generator's last  layer.
G_num_feat          = 128       # Number of feature channels used in the interior of the generator.
G_num_rrdb          = 16        # Number of Residual in Residual Dense Blocks (RRDBs) used in the generator.
G_lr_kern_size      = 3         # Kernel size of generator's convolution layers in the LR domain.
G_lff_kern_size     = 1         # Kernel size of generator's local feature fusion convolution layers.
G_hr_kern_size      = 5         # kernel size of generator's convolution layers in the HR domain.
G_num_gc            = 32        # Number of growth channels used in the generator's Residual Dense Blocks (RDBs).
G_res_scaling       = 0.2       # Residual scaling factor used in generators RRDBs and RDBs.
G_act_type          = 'lrelu'   # Type of activation function used in the discriminator.
G_weight_init_scale = 0.5       # Scale Kaiming initialization weights by this factor during init of G.
G_loss_scales = {               # Factors for scaling different contributions to the total generator loss.
    'rel_avg':  0.005,
    'pix_l1':   0.01,
    'vgg19_l1': 1.0
}  # Losses not included as keys in this dictionary will not be used for training generator

#----------------------------------------------------------------------------
# Feature extractor (VGG19) configuration.

low_level_feat_layer  = -1      # Layer of VGG19 from which low  level features are extracted. Setting this value to None or a negative value disables low lever feature extraction.
high_level_feat_layer = 34      # Layer of VGG19 from which high level features are extracted.
use_feat_extractor_bn = False   # Toggle use of batch normalization in feature extractor.
use_feat_extractor_in = True    # Toggle use of input normalization in feature extractor. This normalizes data as standard

#----------------------------------------------------------------------------
# Write configurations to file.

os.makedirs(run_dir, exist_ok = True)
config_save_file = open(os.path.join(run_dir, "config_save.txt"), "w")
config_save_file.write(
    f"Environment configuration--------------------------------------\n"
    f"base_dir     = {base_dir}\n"
    f"datasets_dir = {datasets_dir}\n"
    f"raw_data_dir = {raw_data_dir}\n"
    f"results_dir  = {results_dir}\n"
    f"run_dir      = {run_dir}\n"
    f"tb_dir       = {tb_dir}\n"
    f"tb_run_dir   = {tb_run_dir}\n"
    f"cp_load_dir  = {cp_load_dir}\n"
    f"cp_save_dir  = {cp_save_dir}\n"
    f"eval_im_dir  = {eval_im_dir}\n"
    f"metrics_dir  = {metrics_dir}\n"
    f"training_data_file    = {training_data_file}\n"
    f"val_loss_metrics_file = {val_loss_metrics_file}\n"
    f"val_pred_file         = {val_pred_file}\n"
    f"val_w_and_grad_file   = {val_w_and_grad_file}\n"
    f"\n"
    f"Device configuration-------------------------------------------\n"
    f"device = {device}\n"
    f"gpu_id = {gpu_id}\n"
    f"\n"
    f"Run configuration----------------------------------------------\n"
    f"is_train                  = {is_train}\n"
    f"is_test                   = {is_test}\n"
    f"load_model_from_save      = {load_model_from_save}\n"
    f"resume_training_from_save = {resume_training_from_save}\n"
    f"generator_load_path       = {generator_load_path}\n"
    f"discriminator_load_path   = {discriminator_load_path}\n"
    f"state_load_path           = {state_load_path}\n"
    f"\n"
    f"Training configuration-----------------------------------------\n"
    f"num_train_it            = {num_train_it}\n"
    f"d_g_train_ratio         = {d_g_train_ratio}\n"
    f"print_train_loss_period = {print_train_loss_period}\n"
    f"save_model_period       = {save_model_period}\n"
    f"opt_G_lr                = {opt_G_lr}\n"
    f"opt_G_w_decay           = {opt_G_w_decay}\n"
    f"opt_G_betas             = {opt_G_betas}\n"
    f"opt_D_lr                = {opt_D_lr}\n"
    f"opt_D_w_decay           = {opt_D_w_decay}\n"
    f"opt_D_betas             = {opt_D_betas}\n"
    f"lr_gamma                = {lr_gamma}\n"
    f"lr_steps                = {lr_steps}\n"
    f"flip_labels             = {flip_labels}\n"
    f"smooth_labels           = {smooth_labels}\n"
    f"noisy_labels            = {noisy_labels}\n"
    f"use_inst_noise          = {use_inst_noise}\n"
    f"train_batch_size        = {train_batch_size}\n"
    f"dataset_train_#_workers = {dataset_train_num_workers}\n"
    f"\n"
    f"Validation configuration---------------------------------------\n"
    f"val_period              = {val_period}\n"
    f"val_visual_period       = {val_visual_period}\n"
    f"num_val_visualizations  = {num_val_visualizations}\n"
    f"val_comparisons         = {val_comparisons}\n"
    f"val_metrics             = {val_metrics}\n"
    f"val_batch_size          = {val_batch_size}\n"
    f"dataset_val_num_workers = {dataset_val_num_workers}\n"
    f"\n"
    f"Testing configuration------------------------------------------\n"
    f"test_comparisons = {test_comparisons}\n"
    f"test_metrics     = {test_metrics}\n"
    f"test_batch_size  = {test_batch_size}\n"
    f"test_num_workers = {test_num_workers}\n"
    f"\n"
    f"Data configuration---------------------------------------------\n"
    f"data_code   = {data_code}\n"
    f"start_year  = {start_year}\n"
    f"start_month = {start_month}\n"
    f"start_day   = {start_day}\n"
    f"end_year    = {end_year}\n"
    f"end_month   = {end_month}\n"
    f"end_day     = {end_day}\n"
    f"altitude    = {altitude}\n"
    f"data_tag    = {data_tag}\n"
    f"HR_size     = {HR_size}\n"
    f"LR_size     = {LR_size}\n"
    f"upscale     = {upscale}\n"
    f"\n"
    f"Discriminator configuration------------------------------------\n"
    f"D_in_num_ch         = {D_in_num_ch}\n"
    f"D_base_num_f        = {D_base_num_f}\n"
    f"D_feat_kern_size    = {D_feat_kern_size}\n"
    f"D_norm_type         = {D_norm_type}\n"
    f"D_act_type          = {D_act_type}\n"
    f"D_weight_init_scale = {D_weight_init_scale}\n"
    f"D_loss_scales       = {D_loss_scales}\n"
    f"\n"
    f"Generator configuration----------------------------------------\n"
    f"G_in_num_ch         = {G_in_num_ch}\n"
    f"G_out_num_ch        = {G_out_num_ch}\n"
    f"G_num_feat          = {G_num_feat}\n"
    f"G_num_rrdb          = {G_num_rrdb}\n"
    f"G_lr_kern_size      = {G_lr_kern_size}\n"
    f"G_lff_kern_size     = {G_lff_kern_size}\n"
    f"G_hr_kern_size      = {G_hr_kern_size}\n"
    f"G_num_gc            = {G_num_gc}\n"
    f"G_res_scaling       = {G_res_scaling}\n"
    f"G_act_type          = {G_act_type}\n"
    f"G_weight_init_scale = {G_weight_init_scale}\n"
    f"G_loss_scales       = {G_loss_scales}\n"
    f"\n"
    f"Feature extractor configuration--------------------------------\n"
    f"low_level_feat_layer  = {low_level_feat_layer}\n"
    f"high_level_feat_layer = {high_level_feat_layer}\n"
    f"use_feat_extractor_bn = {use_feat_extractor_bn}\n"
    f"use_feat_extractor_in = {use_feat_extractor_in}\n"
)
config_save_file.close()