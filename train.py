"""
train.py

Written by Sindre Stenen Blakseth, 2020.

Implements the training loop of ESRGAN.
"""

#----------------------------------------------------------------------------
# Package imports
import math
import os
import torch
import tensorboardX
import random

#----------------------------------------------------------------------------
# File imports

import config
import evaluate

from datasets import load_datasets
from models.esrgan import ESRGAN
from models.losses import get_G_loss, get_D_loss

#----------------------------------------------------------------------------
# Helper function for making labels noisy.

def make_noisy_labels(
        label_type:      bool,
        batch_size:      int,
        noise_stddev:    float = 0.05,
        false_label_val: float = 0.0,
        true_label_val:  float = 1.0,
        val_lower_lim:   float = 0.0,
        val_upper_lim:   float = 1.0) -> torch.Tensor:
    """
    Purpose: make_noisy_labels adds gaussian noise to True/False GAN label values,
             but keeps the resulting value within a specified range,
             and returns a tensor of size batch_size filled with that value.
    Args:
        label_type: True if representing images perceived as real (not generated), else False.
        batch_size: Size of tensor filled with noisy label value.
        noise_stddev: gaussian noise stddev.
        [false|true]_label_val: label values without noise.
        val_[lower|upper]_lim: thresholds for label value cutoff.
    Return:
        See purpose description above.
    """

    label_val: float = random.gauss(mu=0.0, sigma=noise_stddev)

    if label_type == True:
        label_val += true_label_val
    else:
        label_val += false_label_val

    if label_val > val_upper_lim:
        label_val = val_upper_lim
    elif label_val < val_lower_lim:
        label_val = val_lower_lim

    return torch.FloatTensor(batch_size).fill_(label_val)

#----------------------------------------------------------------------------
# Helper function for retrieving labels.

def get_labels(batch_size):
    pred_real = True
    pred_fake = False

    if config.flip_labels:
        pred_real = False
        pred_fake = True

    real_label_value = 1.0
    fake_label_value = 0.0
    if config.smooth_labels and config.flip_labels:
        real_label_value = 1.0
        fake_label_value = 0.1
    elif config.smooth_labels:
        real_label_value = 0.9
        fake_label_value = 0.0

    if config.noisy_labels:
        real_labels = make_noisy_labels(
            label_type      = pred_real,
            batch_size      = batch_size,
            true_label_val  = real_label_value,
            false_label_val = fake_label_value
        )
        fake_labels = make_noisy_labels(
            label_type      = pred_fake,
            batch_size      = batch_size,
            true_label_val  = real_label_value,
            false_label_val = fake_label_value
        )
    else:  # no noise std dev -> no noise
        real_labels = make_noisy_labels(
            label_type      = pred_real,
            batch_size      = batch_size,
            noise_stddev    = 0.0,
            true_label_val  = real_label_value,
            false_label_val = fake_label_value
        )
        fake_labels = make_noisy_labels(
            label_type      = pred_fake,
            batch_size      = batch_size,
            noise_stddev    = 0.0,
            true_label_val  = real_label_value,
            false_label_val = fake_label_value
        )
    return real_labels, fake_labels

def instance_noise(
        sigma_base: float,
        shape:      torch.Size,
        it:         int,
        niter:      int) -> torch.Tensor:
    noise = torch.rand( shape ) # N(0,1)
    var_desired = sigma_base * (1 - (it / niter))
    return noise * math.sqrt(var_desired)

#----------------------------------------------------------------------------
# Training ESRGAN.

def train():
    print("CUDA availability:", torch.cuda.is_available())
    print("Current device:", torch.cuda.current_device(),
          "- num devices:", torch.cuda.device_count(),
          "- device name:", torch.cuda.get_device_name(0))

    torch.backends.cudnn.benchmark = True
    tb_train_writer = tensorboardX.SummaryWriter(log_dir = os.path.join(config.tb_run_dir, 'train'))
    tb_eval_writer  = tensorboardX.SummaryWriter(log_dir = os.path.join(config.tb_run_dir, 'eval'))
    # TODO: Add support for tensorboard.

    # Load datasets for training and validation.
    dataset_train, dataset_val, _ = load_datasets()

    # Create data loader.
    dataloader_train = torch.utils.data.DataLoader(
        dataset     = dataset_train,
        batch_size  = config.train_batch_size,
        shuffle     = True,
        num_workers = config.dataset_train_num_workers,
        pin_memory  = True
    )

    start_epoch = 0
    it = 0
    it_per_epoch = len(dataloader_train)
    num_epochs = config.num_train_it // it_per_epoch

    # Build networks.
    model = ESRGAN()

    with open(os.path.join(config.run_dir, "networks.txt"), "w") as data_file:
        data_file.write(str(model.G))
        data_file.write("\n\n")
        data_file.write(str(model.D))

    # Load from save.
    if config.load_model_from_save:
        print(f"Loading model from from saves. G: {config.generator_load_path}, D: {config.discriminator_load_path}")
        _, __ = model.load_model(
            generator_load_path     = config.generator_load_path,
            discriminator_load_path = config.discriminator_load_path,
            state_load_path         = None
        )

        if config.resume_training_from_save:
            print(f"Resuming training from save. State: {config.state_load_path}")
            loaded_epoch, loaded_it = model.load_model(
                generator_load_path     = None,
                discriminator_load_path = None,
                state_load_path         = config.state_load_path
            )
            print(f"Loaded epoch {loaded_epoch}, it {loaded_it}.")
            if loaded_it:
                start_epoch = loaded_epoch
                it = loaded_it

    # Create saving files.
    with open(config.training_data_file, "w") as data_file:
        num_labels = 3 + len(config.G_loss_scales.keys()) + len(config.D_loss_scales.keys())
        data_file.write(str(num_labels) + "\n")

        label_str = "it"
        label_str += ",G_total_loss"
        for key in config.G_loss_scales.keys():
            label_str += ",G_" + key + "_loss"
        label_str += ",D_total_loss"
        for key in config.D_loss_scales.keys():
            label_str += ",D_" + key + "_loss"
        data_file.write(label_str + "\n")

    with open(config.val_loss_metrics_file, "w") as data_file:
        num_labels = 3 + len(config.G_loss_scales.keys()) + len(config.D_loss_scales.keys()) + len(config.val_metrics) + 2
        data_file.write(str(num_labels) + "\n")

        label_str = "it"
        label_str += ",G_total_loss"
        for key in config.G_loss_scales.keys():
            label_str += ",G_" + key + "_loss"
        label_str += ",D_total_loss"
        for key in config.D_loss_scales.keys():
            label_str += ",D_" + key + "_loss"
        for metric in config.val_metrics:
            label_str += "," + metric
        label_str += ",D_HR_acc"
        label_str += ",D_SR_acc"
        data_file.write(label_str + "\n")

    with open(config.val_pred_file, "w") as data_file:
        data_file.write("")

    #with open(os.path.join(config.run_dir, "eval_debug.txt"), "w") as data_file:
    #    data_file.write("")

    with open(config.val_w_and_grad_file, "w") as data_file:
        data_file.write("26\n")
        data_file.write(
            "it"
            + ",G_grad_start_mean"
            + ",G_grad_start_abs_mean"
            + ",G_grad_start_variance"
            + ",G_grad_mid_mean"
            + ",G_grad_mid_abs_mean"
            + ",G_grad_mid_variance"
            + ",G_grad_end_mean"
            + ",G_grad_end_abs_mean"
            + ",G_grad_end_variance"
            + ",G_weight_start_abs_mean"
            + ",G_weight_start_variance"
            + ",G_weight_mid_abs_mean"
            + ",G_weight_mid_variance"
            + ",G_weight_end_abs_mean"
            + ",G_weight_end_variance"
            + ",D_grad_start_mean"
            + ",D_grad_start_abs_mean"
            + ",D_grad_start_variance"
            + ",D_grad_end_mean"
            + ",D_grad_end_abs_mean"
            + ",D_grad_end_variance"
            + ",D_weight_start_abs_mean"
            + ",D_weight_start_variance"
            + ",D_weight_end_abs_mean"
            + ",D_weight_end_variance"
            + "\n"
        )

    # Training loop.
    for epoch in range(start_epoch, num_epochs + 1):
        print("--------------------------------")
        print(f"Beginning epoch number {epoch}.")
        for i, data in enumerate(dataloader_train):
            if it >= config.num_train_it:
                break
            it += 1

            model.G.train()
            model.D.train()

            #---------------------
            # Load training data.
            real_lr = data[0].to(config.device, dtype = torch.float)
            real_hr = data[1].to(config.device, dtype = torch.float)

            # Define labels for real and fake data
            current_batch_size = real_hr.size(0)
            real_labels, fake_labels = get_labels(current_batch_size)
            real_labels = real_labels.to(config.device).squeeze()
            fake_labels = fake_labels.to(config.device).squeeze()

            # Forward pass through generator.
            fake_hr = model.G(real_lr)

            # For storing training data.
            training_data_str = ""

            #---------------------
            # Train generator.
            if it % config.d_g_train_ratio == 0:
                for param in model.D.parameters():
                    param.required_grad = False
                #for param in model.G.parameters():
                #    param.required_grad = True
                model.G.zero_grad()

                # Forward pass through discriminator.
                if config.use_inst_noise:
                    real_pred = model.D(
                                    real_hr + instance_noise(0.1, real_hr.size(), it, config.num_train_it).to(config.device)
                                ).squeeze().detach()  # Detach to avoid backprop through D.
                    fake_pred = model.D(
                                    fake_hr + instance_noise(0.1, fake_hr.size(), it, config.num_train_it).to(config.device)
                                ).squeeze()
                else:
                    real_pred = model.D(real_hr).squeeze().detach() # Detach to avoid backprop through D.
                    fake_pred = model.D(fake_hr).squeeze()          # Squeeze to go from shape [batch_sz, 1] to [batch_sz].

                # Compute generator loss.
                loss_G, loss_dict_G = get_G_loss(real_hr, fake_hr, real_pred, fake_pred, real_labels, fake_labels)

                # Do backpropagation using generator loss.
                loss_G.backward()

                # Make optimization step for generator.
                model.optimizer_G.step()

                # TODO: Maybe save some loss information and other stuff here.
                if it % config.print_train_loss_period == 0:
                    training_data_str += ("," + str(loss_G.item()))
                    for key in config.G_loss_scales.keys(): # Use the keys of this dict to ensure same order of access across iterations.
                        training_data_str += ("," + str(loss_dict_G[key].item()))

                    #print(f"Generator losses for iteration {it}.")
                    #print("Total generator loss:", loss_G)
                    #print("I am trying to write to tensorboard now.")
                    #loss_dict_G["total"] = loss_G
                    #tb_train_writer.add_scalars('G_losses_train', loss_dict_G, it)
                    #for key in loss_dict_G.keys():
                    #    print("Generator loss ", key, ": ", loss_dict_G[key], sep="")
                    # for hist_name, val in hist_vals.items():
                    #    tb_writer.add_histogram(f"data/hist/{hist_name}", val, it)

            #---------------------
            # Train discriminator.
            for param in model.D.parameters():
                param.requires_grad = True

            #for param in model.G.parameters():
            #    param.requires_grad = False
                # TODO: Why not set requires_grad to False for generator parameters? Because then it cannot be printed during validation.

            model.D.zero_grad()

            # Forward pass through discriminator.
            if config.use_inst_noise:
                real_pred = model.D(
                                real_hr + instance_noise(0.1, real_hr.size(), it, config.num_train_it).to(config.device)
                            ).squeeze()  # Squeeze to go from shape [batch_sz, 1] to [batch_sz].
                fake_pred = model.D(
                                fake_hr.detach() + instance_noise(0.1, real_hr.size(), it, config.num_train_it).to(config.device)
                            ).squeeze()  # Detach to avoid backprop through G.
            else:
                real_pred = model.D(real_hr).squeeze()          # Squeeze to go from shape [batch_sz, 1] to [batch_sz].
                fake_pred = model.D(fake_hr.detach()).squeeze() # Detach to avoid backprop through G.

            # Compute discriminator loss.
            loss_D, loss_dict_D = get_D_loss(real_pred, fake_pred, real_labels, fake_labels)

            # Do backpropagation using discriminator loss.
            loss_D.backward()

            # Make optimization step for discriminator.
            model.optimizer_D.step()

            # TODO: Maybe save some loss information and other stuff here.
            if it % config.print_train_loss_period == 0:
                training_data_str += ("," + str(loss_D.item()))
                for key in config.D_loss_scales.keys():  # Use the keys of this dict to ensure same order of access across iterations.
                    training_data_str += ("," + str(loss_dict_D[key].item()))
                #print(f"Discriminator losses for iteration {it}.")
                #print("Total discriminator loss:", loss_D)
                #loss_dict_D["total"] = loss_D
                #tb_train_writer.add_scalars('D_losses_train', loss_dict_D, it)
                #for key in loss_dict_D.keys():
                #    print("Discriminator loss ", key, ": ", loss_dict_D[key], sep="")

                #print("Real pred:", real_pred.cpu().detach().numpy())
                #print("Fake pred:", fake_pred.cpu().detach().numpy())
                #print("Real labels:", real_labels.cpu().detach().numpy())
                #print("Fake labels:", fake_labels.cpu().detach().numpy())
                #print("D_loss:", loss_D.item())

            # Update learning rate schedulers.
            if i > 0:
                for s in model.schedulers:
                    s.step()

            #---------------------
            # Save model.
            if it % config.save_model_period == 0:
                print(f"saving model (it {it})")
                model.save_model(epoch, it)

            #---------------------
            # Store training data.
            if it % config.print_train_loss_period == 0:
                with open(config.training_data_file, "a") as data_file:
                    data_file.write(str(it) + training_data_str + "\n")

            #---------------------
            # Validation.
            if it % config.val_period == 0:

                G_grad_start   = model.G.network[0].weight.grad.detach()
                G_grad_mid     = model.G.network[1].module[7].RDB2.conv1.weight.grad.detach() # First index chooses the skip_block, second index chooses 8th RRDB
                G_grad_end     = model.G.network[-1].weight.grad.detach()
                G_weight_start = model.G.network[0].weight.detach()
                G_weight_mid   = model.G.network[1].module[7].RDB2.conv1.weight.detach()
                G_weight_end   = model.G.network[-1].weight.detach()

                G_grad_start_mean = G_grad_start.mean().item()
                G_grad_start_abs_mean = (torch.abs(G_grad_start)).mean().item()
                G_grad_start_variance = G_grad_start.var(unbiased=False).item()

                G_grad_mid_mean = G_grad_mid.mean().item()
                G_grad_mid_abs_mean = (torch.abs(G_grad_mid)).mean().item()
                G_grad_mid_variance = G_grad_mid.var(unbiased=False).item()

                G_grad_end_mean = G_grad_end.mean().item()
                G_grad_end_abs_mean = (torch.abs(G_grad_end)).mean().item()
                G_grad_end_variance = G_grad_end.var(unbiased=False).item()

                G_weight_start_abs_mean = (torch.abs(G_weight_start)).mean().item()
                G_weight_start_variance = G_weight_start.var(unbiased=False).item()

                G_weight_mid_abs_mean = (torch.abs(G_weight_mid)).mean().item()
                G_weight_mid_variance = G_weight_mid.var(unbiased=False).item()

                G_weight_end_abs_mean = (torch.abs(G_weight_end)).mean().item()
                G_weight_end_variance = G_weight_end.var(unbiased=False).item()

                D_grad_start   = model.D.features[0].block[0].weight.grad.detach()
                D_grad_end     = model.D.classifier[-1].weight.grad.detach()
                D_weight_start = model.D.features[0].block[0].weight.detach()
                D_weight_end   = model.D.classifier[-1].weight.detach()

                D_grad_start_mean = D_grad_start.mean().item()
                D_grad_start_abs_mean = (torch.abs(D_grad_start)).mean().item()
                D_grad_start_variance = D_grad_start.var(unbiased=False).item()

                D_grad_end_mean = D_grad_end.mean().item()
                D_grad_end_abs_mean = (torch.abs(D_grad_end)).mean().item()
                D_grad_end_variance = D_grad_end.var(unbiased=False).item()

                D_weight_start_abs_mean = (torch.abs(D_weight_start)).mean().item()
                D_weight_start_variance = D_weight_start.var(unbiased=False).item()

                D_weight_end_abs_mean = (torch.abs(D_weight_end)).mean().item()
                D_weight_end_variance = D_weight_end.var(unbiased=False).item()

                val_data_str = "," + str(G_grad_start_mean) \
                             + "," + str(G_grad_start_abs_mean) \
                             + "," + str(G_grad_start_variance) \
                             + "," + str(G_grad_mid_mean) \
                             + "," + str(G_grad_mid_abs_mean) \
                             + "," + str(G_grad_mid_variance) \
                             + "," + str(G_grad_end_mean) \
                             + "," + str(G_grad_end_abs_mean) \
                             + "," + str(G_grad_end_variance) \
                             + "," + str(G_weight_start_abs_mean) \
                             + "," + str(G_weight_start_variance) \
                             + "," + str(G_weight_mid_abs_mean) \
                             + "," + str(G_weight_mid_variance) \
                             + "," + str(G_weight_end_abs_mean) \
                             + "," + str(G_weight_end_variance) \
                             + "," + str(D_grad_start_mean) \
                             + "," + str(D_grad_start_abs_mean) \
                             + "," + str(D_grad_start_variance) \
                             + "," + str(D_grad_end_mean) \
                             + "," + str(D_grad_end_abs_mean) \
                             + "," + str(D_grad_end_variance) \
                             + "," + str(D_weight_start_abs_mean) \
                             + "," + str(D_weight_start_variance) \
                             + "," + str(D_weight_end_abs_mean) \
                             + "," + str(D_weight_end_variance)

                with open(config.val_w_and_grad_file, "a") as data_file:
                    data_file.write(str(it) + val_data_str + "\n")

                model.G.eval()
                model.D.eval()
                num_visuals = 0
                if it % config.val_visual_period == 0:
                    num_visuals = config.num_val_visualizations
                evaluate.evaluate(
                    model       = model,
                    do_val      = True,
                    do_test     = False,
                    num_visuals = num_visuals,
                    comparisons = config.val_comparisons,
                    metrics     = config.val_metrics,
                    call_tag    = 'val_it_' + str(it),
                    train_it    = it,
                    tb_writer   = tb_eval_writer
                )

    tb_train_writer.close()
    tb_eval_writer.close()

#----------------------------------------------------------------------------