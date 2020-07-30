"""
plot.results.py

Written by Sindre Stenen Blakseth, 2020.

Script for plotting ESRGAN training and validation data recorded in .txt-files.
"""

import matplotlib.pyplot as plt
import numpy as np
from os.path import exists, join
import pickle

from config import run_dir as data_dir

#----------------------------------------------------------------------------
# Loading data to plot.

def load_data(filename):
    data_dict = {}

    # Pickle data if not already pickled.
    if not exists(join(data_dir, filename + ".pkl")):
        # Open text file.
        with open(join(data_dir, filename + ".txt"), "r") as data_file:
            train_data_strings = data_file.readlines()

        # Format loaded strings.
        n_lines = len(train_data_strings)
        for i in range(n_lines):
            train_data_strings[i] = train_data_strings[i].strip()  # Remove last newline character.

        # Load first line, which contains the number of variables stored per line.
        n_var = train_data_strings[0]
        print("Number of variables to load is " + n_var)

        # Load second line, which contains labels.
        data_labels = train_data_strings[1].split(",")
        for label in data_labels:
            data_dict[label] = []

        # Load all other lines, which contain recorded training data.
        for i in range(2, n_lines):
            data = train_data_strings[i].split(",")
            for j, label in enumerate(data_labels):
                data_dict[label].append(float(data[j]))

        # Convert lists to numpy arrays.
        for label in data_labels:
            data_dict[label] = np.asarray(data_dict[label])

        # Save dict as pickle for easy access later.
        with open(join(data_dir, filename + ".pkl"), 'wb') as f:
            pickle.dump(data_dict, f)
    else:
        with open(join(data_dir, filename + ".pkl"), 'rb') as f:
            data_dict = pickle.load(f)

    return data_dict

#----------------------------------------------------------------------------
# Plotting data.

def main():
    train_data_filename = "training_data"
    val_loss_filename   = "val_loss_metrics"
    val_grad_filename   = "val_w_and_grads"
    val_pred_filename   = "val_D_predicts"

    train_data_dict = load_data(train_data_filename)
    val_loss_dict = load_data(val_loss_filename)
    val_grad_dict = load_data(val_grad_filename)
    val_data_dict = {**val_loss_dict, **val_grad_dict}

    #------------------------------------------------------------------------
    print("Begin plotting generator training loss.")
    plt.figure()
    plt.title("Generator Training Loss")
    plt.xlabel("Training iterations")
    plt.ylabel("Loss")
    plt.plot(train_data_dict['it'], train_data_dict['G_total_loss'],    label='G total')
    plt.plot(train_data_dict['it'], train_data_dict['G_rel_avg_loss'],  label='G RaGAN')
    plt.plot(train_data_dict['it'], train_data_dict['G_pix_l1_loss'],   label='G L1-pix')
    plt.plot(train_data_dict['it'], train_data_dict['G_vgg19_l1_loss'], label='G L1-feat')
    plt.legend()
    plt.grid()
    plt.savefig(join(data_dir, "generator_training_loss.pdf"))
    plt.close()

    #------------------------------------------------------------------------
    print("Begin plotting discriminator training loss.")
    plt.figure()
    plt.title("Discriminator Training Loss")
    plt.xlabel("Training iterations")
    plt.ylabel("Loss")
    plt.plot(train_data_dict['it'], train_data_dict['D_total_loss'],   label='D total')
    plt.plot(train_data_dict['it'], train_data_dict['D_rel_avg_loss'], label='D RaGAN')
    plt.legend()
    plt.grid()
    plt.savefig(join(data_dir, "discriminator_training_loss.pdf"))
    plt.close()

    #------------------------------------------------------------------------
    print("Begin plotting generator validation loss.")
    plt.figure()
    plt.title("Generator Validation Loss")
    plt.xlabel("Training iterations")
    plt.ylabel("Loss")
    plt.plot(val_data_dict['it'], val_data_dict['G_total_loss'],    label='G total')
    plt.plot(val_data_dict['it'], val_data_dict['G_rel_avg_loss'],  label='G RaGAN')
    plt.plot(val_data_dict['it'], val_data_dict['G_pix_l1_loss'],   label='G L1-pix')
    plt.plot(val_data_dict['it'], val_data_dict['G_vgg19_l1_loss'], label='G L1-feat')
    plt.legend()
    plt.grid()
    plt.savefig(join(data_dir, "generator_validation_loss.pdf"))
    plt.close()

    #------------------------------------------------------------------------
    print("Begin plotting discriminator validation loss.")
    plt.figure()
    plt.title("Discriminator Validation Loss")
    plt.xlabel("Training iterations")
    plt.ylabel("Loss")
    plt.plot(val_data_dict['it'], val_data_dict['D_total_loss'],   label='D total')
    plt.plot(val_data_dict['it'], val_data_dict['D_rel_avg_loss'], label='D RaGAN')
    plt.legend()
    plt.grid()
    plt.savefig(join(data_dir, "discriminator_validation_loss.pdf"))
    plt.close()

    #------------------------------------------------------------------------
    print("Begin plotting PSNR on validation set.")
    plt.figure()
    plt.title("Average PSNR Score over the Validation Set")
    plt.xlabel("Training iterations")
    plt.ylabel("PSNR")
    plt.plot(val_data_dict['it'], val_data_dict['psnr'], label='PSNR')
    plt.legend()
    plt.grid()
    plt.savefig(join(data_dir, "validation_psnr.pdf"))
    plt.close()

    #------------------------------------------------------------------------
    print("Begin plotting LPIPS on validation set.")
    plt.figure()
    plt.title("Average LPIPS Score over the Validation Set")
    plt.xlabel("Training iterations")
    plt.ylabel("LPIPS")
    plt.plot(val_data_dict['it'], val_data_dict['lpips'], label='LPIPS')
    plt.legend()
    plt.grid()
    plt.savefig(join(data_dir, "validation_lpips.pdf"))
    plt.close()

    #------------------------------------------------------------------------
    print("Begin plotting accuracies on validation set.")
    plt.figure()
    plt.title("Average Discriminator accuracies over the Validation Set")
    plt.xlabel("Training iterations")
    plt.ylabel("Accuracy")
    plt.plot(val_data_dict['it'], val_data_dict['D_HR_acc'], label='HR')
    plt.plot(val_data_dict['it'], val_data_dict['D_SR_acc'], label='SR')
    plt.legend()
    plt.grid()
    plt.savefig(join(data_dir, "validation_accuracies.pdf"))
    plt.close()

    #------------------------------------------------------------------------
    print("Begin plotting mean gradients in generator.")
    plt.figure()
    plt.title("Mean Gradients of Generator Layers")
    plt.xlabel("Training iterations")
    plt.ylabel(r"mean($\partial w / \partial L$)")
    plt.plot(val_data_dict['it'], val_data_dict['G_grad_start_mean'], label='first layer')
    plt.plot(val_data_dict['it'], val_data_dict['G_grad_mid_mean'],   label='mid layer')
    plt.plot(val_data_dict['it'], val_data_dict['G_grad_end_mean'],   label='last layer')
    plt.legend()
    plt.grid()
    plt.savefig(join(data_dir, "generator_mean_gradients.pdf"))
    plt.close()

    #------------------------------------------------------------------------
    print("Begin plotting mean absolute gradients in generator.")
    plt.figure()
    plt.title("Mean Absolute Value of Gradients of Generator Layers")
    plt.xlabel("Training iterations")
    plt.ylabel(r"mean($|\partial w / \partial L|$)")
    plt.plot(val_data_dict['it'], val_data_dict['G_grad_start_abs_mean'], label='first layer')
    plt.plot(val_data_dict['it'], val_data_dict['G_grad_mid_abs_mean'],   label='mid layer')
    plt.plot(val_data_dict['it'], val_data_dict['G_grad_end_abs_mean'],   label='last layer')
    plt.legend()
    plt.grid()
    plt.savefig(join(data_dir, "generator_mean_abs_gradients.pdf"))
    plt.close()

    #------------------------------------------------------------------------
    print("Begin plotting variance of gradients in generator.")
    plt.figure()
    plt.title("Variance of Gradients of Generator Layers")
    plt.xlabel("Training iterations")
    plt.ylabel(r"var($\partial w / \partial L$)")
    plt.plot(val_data_dict['it'], val_data_dict['G_grad_start_variance'], label='first layer')
    plt.plot(val_data_dict['it'], val_data_dict['G_grad_mid_variance'],   label='mid layer')
    plt.plot(val_data_dict['it'], val_data_dict['G_grad_end_variance'],   label='last layer')
    plt.legend()
    plt.grid()
    plt.savefig(join(data_dir, "generator_var_gradients.pdf"))
    plt.close()

    #------------------------------------------------------------------------
    print("Begin plotting mean gradients in discriminator.")
    plt.figure()
    plt.title("Mean Gradients of Discriminator Layers")
    plt.xlabel("Training iterations")
    plt.ylabel(r"mean($\partial w / \partial L$)")
    plt.plot(val_data_dict['it'], val_data_dict['D_grad_start_mean'], label='first layer')
    plt.plot(val_data_dict['it'], val_data_dict['D_grad_end_mean'],   label='last layer')
    plt.legend()
    plt.grid()
    plt.savefig(join(data_dir, "discriminator_mean_gradients.pdf"))
    plt.close()

    #------------------------------------------------------------------------
    print("Begin plotting mean absolute gradients in discriminator.")
    plt.figure()
    plt.title("Mean Absolute Value of Gradients of Discriminator Layers")
    plt.xlabel("Training iterations")
    plt.ylabel(r"mean($|\partial w / \partial L|$)")
    plt.plot(val_data_dict['it'], val_data_dict['D_grad_start_abs_mean'], label='first layer')
    plt.plot(val_data_dict['it'], val_data_dict['D_grad_end_abs_mean'],   label='last layer')
    plt.legend()
    plt.grid()
    plt.savefig(join(data_dir, "discriminator_mean_abs_gradients.pdf"))
    plt.close()

    #------------------------------------------------------------------------
    print("Begin plotting variance of gradients in discriminator.")
    plt.figure()
    plt.title("Variance of Gradients of Discriminator Layers")
    plt.xlabel("Training iterations")
    plt.ylabel(r"var($\partial w / \partial L$)")
    plt.plot(val_data_dict['it'], val_data_dict['D_grad_start_variance'], label='first layer')
    plt.plot(val_data_dict['it'], val_data_dict['D_grad_end_variance'],   label='last layer')
    plt.legend()
    plt.grid()
    plt.savefig(join(data_dir, "discriminator_var_gradients.pdf"))
    plt.close()

    #------------------------------------------------------------------------
    print("Begin plotting mean absolute weights in generator.")
    plt.figure()
    plt.title("Mean Absolute Value of Weights of Generator Layers")
    plt.xlabel("Training iterations")
    plt.ylabel(r"mean($|w|$)")
    plt.plot(val_data_dict['it'], val_data_dict['G_weight_start_abs_mean'], label='first layer')
    plt.plot(val_data_dict['it'], val_data_dict['G_weight_mid_abs_mean'],   label='mid layer')
    plt.plot(val_data_dict['it'], val_data_dict['G_weight_end_abs_mean'],   label='last layer')
    plt.legend()
    plt.grid()
    plt.savefig(join(data_dir, "generator_mean_abs_weights.pdf"))
    plt.close()

    # ------------------------------------------------------------------------
    print("Begin plotting variance of weights in generator.")
    plt.figure()
    plt.title("Variance of Weights of Generator Layers")
    plt.xlabel("Training iterations")
    plt.ylabel(r"var(w)")
    plt.plot(val_data_dict['it'], val_data_dict['G_weight_start_variance'], label='first layer')
    plt.plot(val_data_dict['it'], val_data_dict['G_weight_mid_variance'],   label='mid layer')
    plt.plot(val_data_dict['it'], val_data_dict['G_weight_end_variance'],   label='last layer')
    plt.legend()
    plt.grid()
    plt.savefig(join(data_dir, "generator_var_weights.pdf"))
    plt.close()

    # ------------------------------------------------------------------------
    print("Begin plotting mean absolute weights in discriminator.")
    plt.figure()
    plt.title("Mean Absolute Value of Weights of Discriminator Layers")
    plt.xlabel("Training iterations")
    plt.ylabel(r"mean($|w|$)")
    plt.plot(val_data_dict['it'], val_data_dict['D_weight_start_abs_mean'], label='first layer')
    plt.plot(val_data_dict['it'], val_data_dict['D_weight_end_abs_mean'],   label='last layer')
    plt.legend()
    plt.grid()
    plt.savefig(join(data_dir, "discriminator_mean_abs_weights.pdf"))
    plt.close()

    # ------------------------------------------------------------------------
    print("Begin plotting variance of weights in discriminator.")
    plt.figure()
    plt.title("Variance of Weights of Discriminator Layers")
    plt.xlabel("Training iterations")
    plt.ylabel(r"var($w$)")
    plt.plot(val_data_dict['it'], val_data_dict['D_weight_start_variance'], label='first layer')
    plt.plot(val_data_dict['it'], val_data_dict['D_weight_end_variance'],   label='last layer')
    plt.legend()
    plt.grid()
    plt.savefig(join(data_dir, "discriminator_var_weights.pdf"))
    plt.close()

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------