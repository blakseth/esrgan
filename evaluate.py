"""
evaluate.py

Written by Sindre Stenen Blakseth, 2020.
Based on work by Eirik Vesterkjær, 2019.

Functions for performing validation and testing of ESRGAN.
"""

#----------------------------------------------------------------------------
# Package imports.

import numpy as np
import os
import pickle as pkl
import torch
import torch.nn as nn

#----------------------------------------------------------------------------
# File imports.

import config
import datasets
import metrics.metrics as metrics

from models.losses import get_G_loss, get_D_loss
from train import get_labels

#----------------------------------------------------------------------------
# Convert torch.Tensor to np.ndarray.

def tensor_to_numpy(im: torch.Tensor):
    """
    Args:
        im (torch.Tensor): Tensor to convert.
    Returns:
        im_final (np.ndarray): Converted tensor.
    """
    im_np = im.squeeze().detach().cpu().numpy() # ch w h -> w, h, ch
    #print(im_np.shape)
    u_im = im_np[0, :, :]
    v_im = im_np[1, :, :]
    w_im = im_np[2, :, :]
    # TODO: Do unnormalization here?
    im_final = np.asarray([u_im, v_im, w_im])
    return im_final

#----------------------------------------------------------------------------
# Create images from LR-, HR-, SR- and interpolation-data.

def make_and_save_images(lr:          torch.Tensor,
                         hr:          torch.Tensor,
                         sr:          torch.Tensor,
                         folder_path: str,
                         filename:    str,
                         comparisons: list):
    """
    Args:
        lr:          Low-resolution data.
        hr:          High-resolution data.
        sr:          Super-resolution data.
        folder_path: Path to folder in which to save images.
        filename:    Name of file in which to save images.
        comparisons: Interpolation techniques we want to compare with.
    Returns:
        imgs: LR-, HR- and SR-images, and images generated using the comparison interpolation techniques.
    """

    # Check if high resolution data is given.
    hr_available = True
    if hr is None:
        hr_available = False

    # Convert from tensor to numpy array.
    lr_img = tensor_to_numpy(lr)
    sr_img = tensor_to_numpy(sr)
    if hr_available:
        hr_img = tensor_to_numpy(hr)

    # Collect LR-, HR- and SR-images in dictionary.
    imgs = dict()
    imgs['LR'] = lr_img
    imgs['SR'] = sr_img
    if hr_available:
        imgs['HR'] = hr_img

    # Interpolations require first axis to be batch size (which is 1).
    lr.unsqueeze_(0)

    # Perform comparison interpolations and store resulting images in dictionary.
    for interpolation in comparisons:
        interpolated = nn.functional.interpolate(
            input         = lr,
            scale_factor  = 4,
            mode          = interpolation,
            align_corners = True
        )
        interpolated_img = tensor_to_numpy(interpolated)
        imgs[interpolation] = interpolated_img[0] # Remove first axis since batch size is 1.

    # Write images to file
    dest_file = os.path.join(folder_path, filename + ".pkl")
    with open(dest_file, 'wb') as f:
        pkl.dump(imgs, f)

#----------------------------------------------------------------------------
# Calculate metrics for evaluating network performance.

def calculate_metrics(hr:           torch.Tensor,
                      sr:           torch.Tensor,
                      metrics_list: list):
    hr_np = tensor_to_numpy(hr)
    sr_np = tensor_to_numpy(sr)
    hr_expanded = hr.unsqueeze(0)
    sr_expanded = sr.unsqueeze(0)

    metrics_dict = {}

    if 'psnr' in metrics_list:
        psnr = metrics.calculate_psnr(hr = hr_np, sr = sr_np)
        metrics_dict['psnr'] = psnr

    if 'lpips' in metrics_list:
        hr_rescaled = hr_expanded * 2 - 1
        sr_rescaled = sr_expanded * 2 - 1
        lpips = metrics.calculate_lpips(hr = hr_rescaled, sr = sr_rescaled)
        metrics_dict['lpips'] = lpips

    return metrics_dict

#----------------------------------------------------------------------------
# Testing/validation loop.

def evaluate(model:       nn.Module,
             do_val:      bool,
             do_test:     bool,
             num_visuals: int,
             comparisons: list,
             metrics:     list,
             call_tag:    str,
             train_it     = None,
             tb_writer    = None):
    with torch.no_grad():
        if do_val == do_test:
            raise ValueError("Invalid evaluation configuration.")

        metric_values = {}

        # Create data loader.
        dataset = None
        dataloader = None
        if do_val:
            _, dataset, _ = datasets.load_datasets()
            dataloader = torch.utils.data.DataLoader(
                dataset     = dataset,
                batch_size  = config.val_batch_size,
                shuffle     = False,
                num_workers = config.dataset_val_num_workers,
                pin_memory  = True
            )
        elif do_test:
            _, _, dataset = datasets.load_datasets()
            dataloader = torch.utils.data.DataLoader(
                dataset     = dataset,
                batch_size  = config.val_batch_size,
                shuffle     = False,
                num_workers = config.dataset_val_num_workers,
                pin_memory  = True
            )

        dataset_length = dataset.__len__()
        if num_visuals == -1:
            num_visuals = dataset_length

        summed_G_losses = {"total": 0.0}
        summed_D_losses = {"total": 0.0}
        for key in config.G_loss_scales.keys():
            summed_G_losses[key] = 0.0
        for key in config.D_loss_scales.keys():
            summed_D_losses[key] = 0.0

        summed_metrics = dict()
        for metric in metrics:
            summed_metrics[metric] = 0.0

        HR_num_correct = 0
        SR_num_correct = 0

        it = 0
        for epoch, data in enumerate(dataloader):
            real_lr = data[0].to(config.device, dtype = torch.float)
            real_hr = data[1].to(config.device, dtype = torch.float)

            current_batch_size = real_hr.size(0)

            real_labels, fake_labels = get_labels(current_batch_size)
            real_labels = real_labels.to(config.device).squeeze()
            fake_labels = fake_labels.to(config.device).squeeze()

            # Feed forward.
            fake_hr = model.G(real_lr)
            real_pred = model.D(real_hr).squeeze()
            fake_pred = model.D(fake_hr).squeeze()  # Squeeze to go from shape [batch_sz, 1] to [batch_sz].

            # Compute losses.
            loss_G, loss_dict_G = get_G_loss(real_hr, fake_hr, real_pred, fake_pred, real_labels, fake_labels)
            loss_D, loss_dict_D = get_D_loss(real_pred, fake_pred, real_labels, fake_labels)
            #with open(os.path.join(config.run_dir, "eval_debug.txt"), 'a') as f:
            #    print("Real pred:", real_pred.cpu().detach().numpy(), file=f)
            #    print("Fake pred:", fake_pred.cpu().detach().numpy(), file=f)
            #    print("Real labels:", real_labels.cpu().detach().numpy(), file=f)
            #    print("Fake labels:", fake_labels.cpu().detach().numpy(), file=f)
            #    print("D_loss:", loss_D.item(), file=f)
            #    print("Batch size:", current_batch_size, file=f)

            # Unreduce losses.
            if do_val:
                summed_G_losses["total"] += (loss_G.item() * current_batch_size)
                for key in config.G_loss_scales.keys():
                    summed_G_losses[key] += (loss_dict_G[key].item() * current_batch_size)
                summed_D_losses["total"] += (loss_D.item() * current_batch_size)
                #with open(os.path.join(config.run_dir, "eval_debug.txt"), 'a') as f:
                #    print("Accumulated D_loss:", summed_D_losses, file=f)
                for key in config.D_loss_scales.keys():
                    summed_D_losses[key] += (loss_dict_D[key].item() * current_batch_size)

            # Calculate metrics.
            for j in range(current_batch_size):
                single_im_metrics = calculate_metrics(
                    hr           = real_hr[j],
                    sr           = fake_hr[j],
                    metrics_list = metrics
                )
                for metric in metrics:
                    summed_metrics[metric] += single_im_metrics[metric]

            # Calculate number of correct discriminator predictions.
            HR_num_correct += torch.sum(real_pred > 0).item()
            SR_num_correct += torch.sum(fake_pred < 0).item()

            # Visualize and save the first num_visuals images in dataset, and the corresponding predictions.
            os.makedirs(config.eval_im_dir, exist_ok=True)
            for j in range(current_batch_size):
                if it >= num_visuals:
                    break
                #print("it:", it)
                #print("num_visuals:", num_visuals)
                filename = call_tag + '_' + 'im_' + str(it)
                make_and_save_images(
                    lr          = real_lr[j],
                    hr          = real_hr[j],
                    sr          = fake_hr[j],
                    folder_path = config.eval_im_dir,
                    filename    = filename,
                    comparisons = comparisons
                )

                with open(config.val_pred_file, "a") as data_file:
                    data_file.write("iteration " + str(train_it) + ", image " + str(j) + "\n")
                    data_file.write("HR prediction:" + str(torch.sigmoid(real_pred[j]).item()) + "\n")
                    data_file.write("SR prediction:" + str(torch.sigmoid(fake_pred[j]).item()) + "\n")
                it += 1

        val_data_str = ""

        val_data_str += "," + str(summed_G_losses['total'] / dataset_length)
        for key in config.G_loss_scales.keys():
            val_data_str += "," + str(summed_G_losses[key] / dataset_length)

        val_data_str += "," + str(summed_D_losses['total'] / dataset_length)
        for key in config.D_loss_scales.keys():
            val_data_str += "," + str(summed_D_losses[key] / dataset_length)

        #with open(os.path.join(config.run_dir, "eval_debug.txt"), 'a') as f:
        #    print("Averaged D_loss:", summed_D_losses['total'] / dataset_length, file=f)
        #    print("", file=f)

        for metric in config.val_metrics:
            val_data_str += "," + str(summed_metrics[metric] / dataset_length)

        val_data_str += "," + str(HR_num_correct / dataset_length)
        val_data_str += "," + str(SR_num_correct / dataset_length)

        with open(config.val_loss_metrics_file, "a") as data_file:
            data_file.write(str(train_it) + val_data_str + "\n")

        """
        if tb_writer is not None:
            val_it = int(call_tag.split("_")[-1])
            tb_writer.add_scalars("G_losses_val", averaged_G_losses, val_it)
            tb_writer.add_scalars("D_losses_val", averaged_D_losses, val_it)

        print("Might write metrics to file now.")
        if do_test:
            os.makedirs(config.metrics_dir, exist_ok=True)
            metrics_file = os.path.join(config.metrics_dir, call_tag + '.csv')
            print("Metrics file:", metrics_file)
            with open(metrics_file, "w") as f:
                for key in metric_values.keys():
                    f.write(f"Values for metric {key}:\n")
                    for i in range(len(metric_values[key])):
                        f.write(f"Image number {i}: {metric_values[key][i]}\n")
                    f.write("\n")

        # Print average value of metrics
        for key in metric_values.keys():
            metric_avg = 0.0
            n = len(metric_values[key])
            for i in range(n):
                metric_avg += metric_values[key][i]
            metric_avg /= n
            print(f"Average for metric {key}: {metric_avg}")
        """

#----------------------------------------------------------------------------

def main():
    x_torch = torch.rand(2, 100, 200)
    x_numpy = tensor_to_numpy(x_torch)
    print(x_torch.size())
    print(x_numpy.shape)

#----------------------------------------------------------------------------

if __name__ == '__name__':
    main()

#----------------------------------------------------------------------------