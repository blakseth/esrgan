"""
losses.py

Written by Sindre Stenen Blakseth, 2020.
Based on work by Eirik Vesterkjær, 2019.

Implements loss functions for generator and discriminator of ESRGAN.
"""

#----------------------------------------------------------------------------
# Package imports.

import torch
import torch.nn as nn
import torchvision

#----------------------------------------------------------------------------
# File imports.

import config

#----------------------------------------------------------------------------
# VGG19-based feature extractor used for computing feature loss.

class VGG19FeatureExtractor(nn.Module):
    def __init__(self,
                 low_level_feature_layer:  int = 1,
                 high_level_feature_layer: int = 34,
                 use_batch_norm:           bool = False,
                 use_input_norm:           bool = True,
                 device                         = torch.device('cpu')):

        # Initialize pre-trained VGG19 model with or without batch normalization.
        super(VGG19FeatureExtractor, self).__init__()
        if use_batch_norm:
            model = torchvision.models.vgg19_bn(pretrained=True)
        else:
            model = torchvision.models.vgg19(pretrained=True)

        # Perform input normalization if use_input_norm is True.
        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)

        # Default to using just high level features.
        self.dual_output = False

        # If a low level feature extraction layer is specified, define the low level feature extractor.
        if low_level_feature_layer is not None and low_level_feature_layer >= 0:
            self.dual_output = True

            self.features_low = nn.Sequential(
                *list(model.features.children())[:(low_level_feature_layer + 1)]
            )

            # No gradients should be propagated through the low level feature extractor.
            for _, v in self.features_low.named_parameters():
                v.requires_grad = False

        # Define the high level feature extractor.
        self.features_high = nn.Sequential(
            *list(model.features.children())[max(low_level_feature_layer + 1, 0): (high_level_feature_layer + 1)]
        )

        # No gradients should be propagated through the high level feature extractor.
        for _, v in self.features_high.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        if self.dual_output:
            x1 = self.features_low(x)
            x2 = self.features_high(x1)
            x1 = x1.reshape(x1.shape[0], -1)
            x2 = x2.reshape(x2.shape[0], -1)
            return torch.cat((x1, x2), 1)
        return self.features_high(x)

# Instantiate the VGG19 feature extractor if it will be used in loss calculations.
if ('vgg19_l1' in config.G_loss_scales.keys() or
    'vgg19_l2' in config.G_loss_scales.keys()):
    feature_extractor = VGG19FeatureExtractor(
        low_level_feature_layer  = config.low_level_feat_layer,
        high_level_feature_layer = config.high_level_feat_layer,
        use_batch_norm = False,
        use_input_norm = True,
        device         = config.device
    )
    feature_extractor = feature_extractor.to(device = config.device)

#----------------------------------------------------------------------------
# Discriminator loss.

def get_D_loss(real_pred, fake_pred, real_label, fake_label):
    loss_dict_D = {}

    # Adversarial losses.
    criterion = nn.BCEWithLogitsLoss().to(config.device)

    if 'dcgan' in config.D_loss_scales.keys():
        dcgan_loss = criterion(real_pred, real_label) + criterion(fake_pred, fake_label)
        loss_dict_D['dcgan'] = config.D_loss_scales['dcgan'] * dcgan_loss

    if 'rel' in config.D_loss_scales.keys():
        rel_loss = criterion(real_pred - fake_pred, real_label)
        loss_dict_D['rel'] = config.D_loss_scales['rel'] * rel_loss

    if 'rel_avg' in config.D_loss_scales.keys():
        rel_avg_loss = (criterion(real_pred - torch.mean(fake_pred), real_label) +
                        criterion(fake_pred - torch.mean(real_pred), fake_label)) / 2.0
        loss_dict_D['rel_avg'] = config.D_loss_scales['rel_avg'] * rel_avg_loss

    # Sum all computed losses.
    total_loss = 0.0
    for loss in loss_dict_D.values():
        total_loss += loss

    return total_loss, loss_dict_D

#----------------------------------------------------------------------------
# Generator loss.

def get_G_loss(real_hr, fake_hr, real_pred, fake_pred, real_label, fake_label):
    loss_dict_G = {}

    # Adversarial losses.
    criterion = nn.BCEWithLogitsLoss().to(config.device)

    if 'dcgan' in config.G_loss_scales.keys():
        dcgan_loss = criterion(fake_pred, real_label) # + criterion(real_pred, fake_label)
        loss_dict_G['dcgan'] = config.G_loss_scales['dcgan'] * dcgan_loss

    if 'rel' in config.G_loss_scales.keys():
        rel_loss = criterion(fake_pred - real_pred, real_label)
        loss_dict_G['rel'] = config.G_loss_scales['rel'] * rel_loss

    if 'rel_avg' in config.G_loss_scales.keys():
        rel_avg_loss = (criterion(fake_pred - torch.mean(real_pred), real_label) +
                        criterion(real_pred - torch.mean(fake_pred), fake_label)) / 2.0
        loss_dict_G['rel_avg'] = config.G_loss_scales['rel_avg'] * rel_avg_loss

    # TODO: Maybe feature and pixel losses should be sent to device?
    # Feature losses.
    if 'vgg19_l1' in config.G_loss_scales.keys():
        real_features = feature_extractor(real_hr).detach()
        fake_features = feature_extractor(fake_hr)
        feature_criterion = nn.L1Loss()
        feature_l1_loss = feature_criterion(real_features, fake_features)
        loss_dict_G['vgg19_l1'] = config.G_loss_scales['vgg19_l1'] * feature_l1_loss

    if 'vgg19_l2' in config.G_loss_scales.keys():
        real_features = feature_extractor(real_hr).detach()
        fake_features = feature_extractor(fake_hr)
        feature_criterion = nn.MSELoss()
        feature_l2_loss = feature_criterion(real_features, fake_features)
        loss_dict_G['vgg19_l2'] = config.G_loss_scales['vgg19_l2'] * feature_l2_loss

    # Pixel losses.
    if 'pix_l1' in config.G_loss_scales.keys():
        pixel_criterion = nn.L1Loss()
        pixel_l1_loss = pixel_criterion(real_hr, fake_hr)
        loss_dict_G['pix_l1'] = config.G_loss_scales['pix_l1'] * pixel_l1_loss

    if 'pix_l2' in config.G_loss_scales.keys():
        pixel_criterion = nn.MSELoss()
        pixel_l2_loss = pixel_criterion(real_hr, fake_hr)
        loss_dict_G['pix_l2'] = config.G_loss_scales['pix_l2'] * pixel_l2_loss

    # Sum all computed losses.
    total_loss = 0.0
    for loss in loss_dict_G.values():
        total_loss += loss

    return total_loss, loss_dict_G

#----------------------------------------------------------------------------