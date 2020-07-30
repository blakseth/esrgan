"""
esrgan.py

Written by Sindre Stenen Blakseth, 2020.

Based on work by Eirik Vesterkjær, 2019.

Implements ESRGAN model.
"""

#----------------------------------------------------------------------------
# Package imports.

import os
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

#----------------------------------------------------------------------------
# File imports.

import config
import models.networks as nets
import models.initialization as init

#----------------------------------------------------------------------------
# ESRGAN implementation.

class ESRGAN:
    device = None

    G: nn.Module = None
    D: nn.Module = None

    schedulers = None
    optimizers = None

    is_train = None

    # TODO: These dicts are currently used. Maybe make use of them?
    loss_dict = None
    hist_dict = None
    metrics_dict = None

    def __init__(self):
        self.device = config.device
        self.schedulers = []
        self.optimizers = []
        self.is_train = config.is_train

        self.loss_dict = {
            "train_loss_D": 0.0,
            "train_loss_G": 0.0,
            "train_loss_G_GAN": 0.0,
            "train_loss_G_feat": 0.0,
            "train_loss_G_pix": 0.0,
            "val_loss_D": 0.0,
            "val_loss_G": 0.0,
            "val_loss_G_GAN": 0.0,
            "val_loss_G_feat": 0.0,
            "val_loss_G_pix": 0.0,
        }
        self.hist_dict = {
            "val_grad_G_first_layer": 0.0,
            "val_grad_G_last_layer": 0.0,
            "val_grad_D_first_layer": -1.0,
            "val_grad_D_last_layer": -1.0,
            "val_weight_G_first_layer": 0.0,
            "val_weight_G_last_layer": 0.0,
            "val_weight_D_first_layer": -1.0,
            "val_weight_D_last_layer": -1.0,
            "SR_pix_distribution": 0.0,
            "D_pred_HR": 0.0,
            "D_pred_SR": 0.0,
        }
        self.metrics_dict = {
            "val_PSNR": 0.0,
        }

        # Define and initialize generator.
        self.G = nets.ESRGAN128Generator(
            in_num_ch     = config.G_in_num_ch,
            out_num_ch    = config.G_out_num_ch,
            num_feat      = config.G_num_feat,
            num_rrdb      = config.G_num_rrdb,
            upscale       = config.upscale,
            lr_kern_size  = config.G_lr_kern_size,
            lff_kern_size = config.G_lff_kern_size,
            hr_kern_size  = config.G_hr_kern_size,
            num_gc        = config.G_num_gc,
            res_scaling   = config.G_res_scaling,
            act_type      = config.G_act_type
        )
        self.G = self.G.to(self.device)
        init.init_weights(self.G, scale = config.G_weight_init_scale)

        # Define and initialize discriminator.
        self.D = nets.VGG128Discriminator(
            in_num_ch      = config.D_in_num_ch,
            base_num_f     = config.D_base_num_f,
            feat_kern_size = config.D_feat_kern_size,
            norm_type      = config.D_norm_type,
            act_type       = config.D_act_type
        )
        self.D = self.D.to(self.device)
        init.init_weights(self.D, scale = config.D_weight_init_scale)

        # Define optimizers.
        if self.is_train:
            self.optimizer_G = torch.optim.Adam(
                params       = self.G.parameters(),
                lr           = config.opt_G_lr,
                weight_decay = config.opt_G_w_decay,
                betas        = config.opt_G_betas
            )
            self.optimizer_D = torch.optim.Adam(
                params       = self.D.parameters(),
                lr           = config.opt_D_lr,
                weight_decay = config.opt_D_w_decay,
                betas        = config.opt_D_betas
            )
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        # Define learning rate schedulers.
        if self.is_train:
            self.scheduler_G = lr_scheduler.MultiStepLR(self.optimizer_G, config.lr_steps, gamma = config.lr_gamma)
            self.scheduler_D = lr_scheduler.MultiStepLR(self.optimizer_D, config.lr_steps, gamma = config.lr_gamma)
            self.schedulers.append(self.scheduler_G)
            self.schedulers.append(self.scheduler_D)

    # Function for loading model from file.
    def load_model(self,
                   generator_load_path:     str = None,
                   discriminator_load_path: str = None,
                   state_load_path:         str = None):
        """
        Args:
            generator_load_path:     Path to file from which to load generator.
            discriminator_load_path: Path to file from which to load discriminator.
            state_load_path:         Path to file from which to load state.
        Returns:
            epoch, it: epoch and iteration at loaded state (None, None if noe state was loaded).
        """
        # Load generator.
        if not generator_load_path is None:
            self.G.load_state_dict(torch.load(generator_load_path, map_location="cpu"))

        # Load discriminator.
        if not discriminator_load_path is None:
            print("Discriminator:", self.D)
            self.D.load_state_dict(torch.load(discriminator_load_path, map_location="cpu"))

        # Load state.
        if not state_load_path is None:
            # Load.
            state = torch.load(state_load_path)
            loaded_optimizers = state["optimizers"]
            loaded_schedulers = state["schedulers"]

            # Assert valid loads.
            assert len(loaded_optimizers) == len(
                self.optimizers), f"Loaded {len(loaded_optimizers)} optimizers but expected {len(self.optimizers)}"
            assert len(loaded_schedulers) == len(
                self.schedulers), f"Loaded {len(loaded_schedulers)} schedulers but expected {len(self.schedulers)}"

            # Set optimizers using loaded data.
            for i, o in enumerate(loaded_optimizers):
                self.optimizers[i].load_state_dict(o)

            # Set scheduler using loaded data.
            for i, s in enumerate(loaded_schedulers):
                self.schedulers[i].load_state_dict(s)

            return state["epoch"], state["it"]
        return None, None

    # Function for saving model to file.
    def save_model(self,
                   epoch:      int  = 0,
                   it:         int  = 0,
                   save_G:     bool = True,
                   save_D:     bool = True,
                   save_state: bool = True):
        """
        Args:
            save_G:     Toggle for saving generator.
            save_D:     Toggle for saving discriminator.
            save_state: Toggle for saving state.
            epoch:      Current epoch     when saving state.
            it:         Current iteration when saving state.
        """

        # Set up save paths.
        save_basepath = config.cp_save_dir
        generator_save_path     = os.path.join(save_basepath, f"G_{it}.pth")
        discriminator_save_path = os.path.join(save_basepath, f"D_{it}.pth")
        state_save_path         = os.path.join(save_basepath, f"state_{it}.pth")

        # Save generator.
        if save_G:
            print("Saving generator to path:", generator_save_path)
            torch.save(self.G.state_dict(), generator_save_path)

        # Save discriminator.
        if save_D:
            torch.save(self.D.state_dict(), discriminator_save_path)

        # Save state.
        if save_state:
            state = {"it": it, "epoch": epoch, "schedulers": [], "optimizers": []}
            for s in self.schedulers:
                state["schedulers"].append(s.state_dict())
            for o in self.optimizers:
                state["optimizers"].append(o.state_dict())
            torch.save(state, state_save_path)

#----------------------------------------------------------------------------