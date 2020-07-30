"""
networks.py

Compiled and modified by Sindre Stenen Blakseth, 2020.

Based on work by Eirik Vesterkjær, 2019.

Network architectures for ESRGAN implementation.
"""

#----------------------------------------------------------------------------
# Package imports.

import math
import torch
import torch.nn as nn

#----------------------------------------------------------------------------
# Skip connection block used by ESRGAN128Generator.

class SkipConnectionBlock(nn.Module):
    def __init__(self, submodule):
        """
        Args:
            submodule: Module to insert skip connection across.
        """
        super(SkipConnectionBlock, self).__init__()
        self.module = submodule
    def forward(self, x):
        return x + self.module(x)

#----------------------------------------------------------------------------
# Block containing upsampling, conv2d and activation.

class UpConv(nn.Module):
    def __init__(self,
                 in_num_ch:       int,
                 out_num_ch:      int,
                 scale:           int,
                 lrelu_neg_slope: float = 0.2):
        """
        Args:
            in_num_ch:       Number of input channels.
            out_num_ch:      Number of output channels.
            scale:           Upscaling factor.
            lrelu_neg_slope: lrelu_neg_slope: Negative slope of leaky relu activation function.
        """
        super(UpConv, self).__init__()

        self.upconv = nn.Sequential(
            nn.Upsample(
                scale_factor = scale,
                mode         = 'nearest'
            ),
            nn.Conv2d(
                in_channels  = in_num_ch,
                out_channels = out_num_ch,
                kernel_size  = 3,
                padding      = 1,
                stride       = 1
            ),
            nn.LeakyReLU(negative_slope = lrelu_neg_slope)
        )
    def forward(self, x):
        return self.upconv(x)

# ---------------------------------------------------------------------------
# A 2d convolution layer followed by a LReLU activation layer.

# class Conv2dAndLReLU(nn.module):
#     def __init__(self,
#                  in_num_ch:       int,
#                  out_num_ch:      int,
#                  feat_kern_size:  int   = 3,
#                  lrelu_neg_slope: float = 0.2,
#                  norm_type:       str   = 'batch',
#                  drop_first_norm: bool  = False):
#
#         super(Conv2dAndLReLU, self).__init__()
#
#         if not feat_kern_size % 2 == 1:
#             raise NotImplementedError(f"Even kernel size {str(feat_kern_size)} not implemented.")
#
#         # Defining conv2d layer
#         stride     = 1
#         padding    = (feat_kern_size - stride) / 2
#         conv_layer = nn.Conv2d(in_channels  = in_num_ch,
#                                out_channels = out_num_ch,
#                                kernel_size  = feat_kern_size,
#                                padding      = padding,
#                                stride       = stride)
#
#         # Defining activation layer.
#         act_layer = nn.LeakyReLU(negative_slope=lrelu_neg_slope)
#
#         # Defining block.
#         self.block = nn.Sequential(conv_layer, act_layer)
#
#     def forward(self, x):
#         return self.block(x)

# ---------------------------------------------------------------------------
# Residual Dense Block with 5 convolution layers.

class RDB(nn.Module):
    """
    Based on: https://github.com/open-mmlab/mmsr/blob/dfdaaef492107f448ea49b9edebcb4f81e7893e8/codes/models/archs/RRDBNet_arch.py
    The implementation from the authors of ESRGAN has kernel size 3 in the local feature fusion convolution,
    whereas the authors of the residual dense block use kernel size 1 for the same layer. The authors of ESRGAN
    make no mention of this alteration in their paper or in their implementation.
    """
    def __init__(self,
                 in_num_ch:       int,
                 num_gc:          int   = 32,
                 feat_kern_size:  int   = 3,
                 lff_kern_size:   int   = 1,
                 lrelu_neg_slope: float = 0.2,
                 residual_scale:  float = 0.2):
        """
        Args:
            in_num_ch:       Number of input channels.
            num_gc:          Number of intermediate growth channels.
            feat_kern_size:  Kernel size of feature extraction convolution layers.
            lff_kern_size:   Kernel size of local feature fusion convolution layer.
            lrelu_neg_slope: Negative slope of leaky relu activation functions.
            residual_scale:  Residual scaling parameter.
        """
        super(RDB, self).__init__()

        self.residual_scale = residual_scale

        # Validate initialization parameters.
        if feat_kern_size <= 0 or (feat_kern_size % 2) == 0:
            raise ValueError("Feature kernel size (feat_kern_size) must be an odd number > 0")
        if lff_kern_size <= 0 or (lff_kern_size % 2) == 0:
            raise ValueError("LFF kernel size (lff_kern_size) must be an odd number > 0")

        # Calculate padding required for preserving dimensions of feature maps.
        feat_stride  = 1
        feat_padding = int((feat_kern_size - feat_stride) / 2)
        lff_stride   = 1
        lff_padding  = int((lff_kern_size - lff_stride) / 2)

        # feature extraction convolution layers.
        self.conv1 = nn.Conv2d(
            in_channels  = in_num_ch,
            out_channels = num_gc,
            kernel_size  = feat_kern_size,
            padding      = feat_padding,
            stride       = feat_stride
        )
        self.conv2 = nn.Conv2d(
            in_channels  = in_num_ch + num_gc,
            out_channels = num_gc,
            kernel_size  = feat_kern_size,
            padding      = feat_padding,
            stride       = feat_stride
        )
        self.conv3 = nn.Conv2d(
            in_channels  = in_num_ch + 2 * num_gc,
            out_channels = num_gc,
            kernel_size  = feat_kern_size,
            padding      = feat_padding,
            stride       = feat_stride
        )
        self.conv4 = nn.Conv2d(
            in_channels  = in_num_ch + 3 * num_gc,
            out_channels = num_gc,
            kernel_size  = feat_kern_size,
            padding      = feat_padding,
            stride       = feat_stride
        )

        # Local feature fusion convolution layer.
        self.lff = nn.Conv2d(
            in_channels  = in_num_ch + 4 * num_gc,
            out_channels = in_num_ch,
            kernel_size  = lff_kern_size,
            padding      = lff_padding,
            stride       = lff_stride
        )

        # Activation function.
        self.lrelu = nn.LeakyReLU(negative_slope = lrelu_neg_slope, inplace = True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.lff(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * self.residual_scale + x

# ---------------------------------------------------------------------------
# Residual in Residual Dense Block.

class RRDB(nn.Module):
    """
    Based on: https://github.com/open-mmlab/mmsr/blob/dfdaaef492107f448ea49b9edebcb4f81e7893e8/codes/models/archs/RRDBNet_arch.py
    """
    def __init__(self,
                 in_num_ch:       int,
                 num_gc:          int   = 32,
                 feat_kern_size:  int   = 3,
                 lff_kern_size:   int   = 1,
                 lrelu_neg_slope: float = 0.2,
                 residual_scale:  float = 0.2):
        """
        Args:
            in_num_ch:       Number of input channels.
            num_gc:          Number of intermediate growth channels.
            feat_kern_size:  Kernel size of feature extraction convolution layers.
            lff_kern_size:   Kernel size of local feature fusion convolution layer.
            lrelu_neg_slope: Negative slope of leaky relu activation functions.
            residual_scale:  Residual scaling parameter.
        """

        super(RRDB, self).__init__()

        self.residual_scale = residual_scale

        if feat_kern_size <= 0 or (feat_kern_size % 2) == 0:
            raise ValueError("Feature kernel size (feat_kern_size) must be an odd number > 0")
        if lff_kern_size <= 0 or (lff_kern_size % 2) == 0:
            raise ValueError("LFF kernel size (lff_kern_size) must be an odd number > 0")

        self.RDB1 = RDB(
            in_num_ch       = in_num_ch,
            num_gc          = num_gc,
            feat_kern_size  = feat_kern_size,
            lff_kern_size   = lff_kern_size,
            lrelu_neg_slope = lrelu_neg_slope,
            residual_scale  = residual_scale
        )
        self.RDB2 = RDB(
            in_num_ch       = in_num_ch,
            num_gc          = num_gc,
            feat_kern_size  = feat_kern_size,
            lff_kern_size   = lff_kern_size,
            lrelu_neg_slope = lrelu_neg_slope,
            residual_scale  = residual_scale
        )
        self.RDB3 = RDB(
            in_num_ch       = in_num_ch,
            num_gc          = num_gc,
            feat_kern_size  = feat_kern_size,
            lff_kern_size   = lff_kern_size,
            lrelu_neg_slope = lrelu_neg_slope,
            residual_scale  = residual_scale
        )

    def forward(self, x):
        x1 = self.RDB1(x)  * self.residual_scale + x
        x2 = self.RDB2(x1) * self.residual_scale + x1
        x3 = self.RDB3(x2) * self.residual_scale + x2
        return x3 * self.residual_scale + x


#----------------------------------------------------------------------------
# 2d convolution block used in VGG128Discriminator.

class Conv2dBlock(nn.Module):
    """
    Structure:
        - conv2d with stride 1 and padding chosen to preserve dimensions of feature map.
        - normalization layer; the type of normalization is user-defined.
        - leaky relu activation with user-defined negative slope.
        - conv2d with stride 2 and padding chosen to halve dimensions of feature map.
        - normalization layer; the type of normalization is user-defined.
        - leaky relu activation with user-defined negative slope.
    """
    def __init__(self,
                 in_num_ch:       int,
                 out_num_ch:      int,
                 feat_kern_size1: int   = 3,
                 lrelu_neg_slope: float = 0.2,
                 norm_type:       str   = 'batch',
                 drop_first_norm: bool  = False):
        """
        Args:
            in_num_ch:       Number of input channels.
            out_num_ch:      Number of output channels.
            feat_kern_size1: Kernel size of 1st conv2d layer.
            lrelu_neg_slope: Negative slope of leaky relu activation functions.
            norm_type:       Normalization type (valid values: 'batch', 'instance', 'none', None).
            drop_first_norm: Bool to determine if dropping 1st normalization layer.
        """
        super(Conv2dBlock, self).__init__()

        if not feat_kern_size1 % 2 == 1:
            raise NotImplementedError(f"Even kernel size {str(feat_kern_size1)} not implemented.")

        # Helper for defining normalization layers.
        def norm(n_ch: int, t: str):
            if t is None or t == 'none':
                return None
            elif t == 'batch':
                return nn.BatchNorm2d(n_ch, track_running_stats=False)
            elif t == 'instance':
                return nn.InstanceNorm2d(n_ch)
            else:
                raise NotImplementedError(f"Unknown norm type {t}.")

        # Defining first conv2d layer.
        stride1     = 1
        padding1    = int((feat_kern_size1 - stride1) / 2)
        conv_layer1 = nn.Conv2d(
            in_channels  = in_num_ch,
            out_channels = out_num_ch,
            kernel_size  = feat_kern_size1,
            padding      = padding1,
            stride       = stride1
        )

        # Defining first normalization layer.
        norm_layer1 = norm(out_num_ch, norm_type)

        # Defining first activation layer.
        act_layer1 = nn.LeakyReLU(negative_slope = lrelu_neg_slope)

        # Defining second conv2d layer.
        feat_kern_size2 = 4
        stride2     = 2
        padding2    = int((feat_kern_size2 - stride2) / 2)
        conv_layer2 = nn.Conv2d(
            in_channels  = out_num_ch,
            out_channels = out_num_ch,
            kernel_size  = feat_kern_size2,
            padding      = padding2,
            stride       = stride2
        )

        # Defining second normalization layer.
        norm_layer2 = norm(out_num_ch, norm_type)

        # Defining second activation layer.
        act_layer2 = nn.LeakyReLU(negative_slope = lrelu_neg_slope)

        # Defining block
        block = []
        block.append(conv_layer1)
        if norm_layer1 is not None and not drop_first_norm:
            block.append(norm_layer1)
        block.append(act_layer1)
        block.append(conv_layer2)
        if norm_layer2 is not None:
            block.append(norm_layer2)
        block.append(act_layer2)
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)

#----------------------------------------------------------------------------
# ESRGAN generator.

class ESRGAN128Generator(nn.Module):
    def __init__(self,
                 in_num_ch:     int,
                 out_num_ch:    int,
                 num_feat:      int,
                 num_rrdb:      int = 23,
                 upscale:       int = 4,
                 lr_kern_size:  int = 3,
                 lff_kern_size: int = 1,
                 hr_kern_size:  int = 3,
                 num_gc:        int = 32,
                 res_scaling:   float = 0.2,
                 act_type:      str = 'leakyrelu'):
        """
        Args:
            in_num_ch:     Number of input channels.
            out_num_ch:    Number of output channels.
            num_feat:      Number of intermediate feature channels.
            num_rrdb:      Number of RRDBs.
            upscale:       Upscaling factor for making HR data from LR data.
            lr_kern_size:  Kernel size for convolution layers in the LR domain.
            lff_kern_size: Kernel size for local feature fusion convolution layers.
            hr_kern_size:  Kernel size for convolution layers in the HR domain.
            num_gc:        Number of growth channels in the RDBs.
            res_scaling:   Residual scaling parameter used in RDBs and RRDBs.
            act_type:      Type of activation function (valid values: 'relu', 'leakyrelu').
                           Choosing act_type='relu' sets negative slope = 0 for all
                           leaky relu activation functions used in discriminator.
        """

        super(ESRGAN128Generator, self).__init__()

        # Choose slope for leaky relu activation functions.
        slope = 0
        if act_type == "lrelu":
            slope = 0.2
        elif act_type == "relu":
            slope = 0.0
        else:
            print(f"WARNING (generator): activation type {act_type} has not been implemented - defaulting to leaky ReLU (0.2)")
            slope = 0.2

        # LR feature extraction 1.
        lr_stride  = 1
        lr_padding = int((lr_kern_size - lr_stride) / 2)
        lr_conv1   = nn.Conv2d(
            in_channels  = in_num_ch,
            out_channels = num_feat,
            kernel_size  = lr_kern_size,
            padding      = lr_padding,
            stride       = lr_stride,
        )

        # Residual in residual dense blocks
        rrdbs = [
            RRDB(
                in_num_ch       = num_feat,
                num_gc          = num_gc,
                feat_kern_size  = lr_kern_size,
                lff_kern_size   = lff_kern_size,
                lrelu_neg_slope = slope,
                residual_scale  = res_scaling
            ) for block in range(num_rrdb)
        ]

        # LR feature extraction 2.
        lr_conv2 = nn.Conv2d(
            in_channels  = num_feat,
            out_channels = num_feat,
            kernel_size  = lr_kern_size,
            padding      = lr_padding,
            stride       = lr_stride,
        )

        # Skip connection across RRDBs and lr_conv2.
        skip_rrdbs_and_conv2 = SkipConnectionBlock(
            nn.Sequential(*rrdbs, lr_conv2)
        )

        # Upsampling.
        n_upsample = math.floor(math.log2(upscale))
        if 2 ** n_upsample != upscale:
            print(f"WARNING (generator): upsampling only supported for factors 2^n. Defaulting {upscale} to {2 ** n_upsample}.")
        upsampler = [
            UpConv(
                in_num_ch       = num_feat,
                out_num_ch      = num_feat,
                scale           = 2,
                lrelu_neg_slope = slope
            ) for upsample in range(n_upsample)
        ]

        # HR feature extraction.
        hr_stride  = 1
        hr_padding = int((hr_kern_size - hr_stride) / 2)
        hr_conv1   = nn.Conv2d(
            in_channels  = num_feat,
            out_channels = num_feat,
            kernel_size  = hr_kern_size,
            padding      = hr_padding,
            stride       = hr_stride,
        )
        hr_act   = nn.LeakyReLU(negative_slope = slope)
        hr_conv2 = nn.Conv2d(
            in_channels  = num_feat,
            out_channels = out_num_ch,
            kernel_size  = hr_kern_size,
            padding      = hr_padding,
            stride       = hr_stride,
        )

        # Assemble the full generator network.
        self.network = nn.Sequential(
            lr_conv1,
            skip_rrdbs_and_conv2,
            *upsampler,
            hr_conv1,
            hr_act,
            hr_conv2
        )
        print("Finished building ESRGAN128Generator.")

    def forward(self, x):
        return self.network(x)


#----------------------------------------------------------------------------
# VGG-style discriminator for 128x128-images.

class VGG128Discriminator(nn.Module):
    """
    Structure:
        - Conv2dBlock, [in_num_ch, 128, 128]  -> [base_num_f, 64, 64]
        - Conv2dBlock, [base_num_f, 64, 64]   -> [2*base_num_f, 32, 32]
        - Conv2dBlock, [2*base_num_f, 32, 32] -> [4*base_num_f, 16, 16]
        - Conv2dBlock, [4*base_num_f, 16, 16] -> [8*base_num_f, 8, 8]
        - Conv2dBlock, [8*base_num_f, 8, 8]   -> [8*base_num_f, 4, 4]
        - nn.Linear,   [8*base_num_f * 4 * 4] -> [100]
        - nn.LeakyReLU
        - nn.Linear,   [100] -> [1]
    """
    def __init__(self,
                 in_num_ch: int,
                 base_num_f: int,
                 feat_kern_size: int = 3,
                 norm_type: str = "batch",
                 act_type: str = "leakyrelu"):
        """
        Args:
            in_num_ch:      Number of input channels.
            base_num_f:     Number of output channels of first Conv2dBlock.
            feat_kern_size: Kernel size of 1st conv2d layer of each Conv2dBlock.
            norm_type:      Type of normalization layer used in each Conv2dBlock.
            act_type:       Type of activation function (valid values: 'relu', 'leakyrelu').
                            Choosing act_type='relu' sets negative slope = 0 for all
                            leaky relu activation functions used in discriminator.
        """

        super(VGG128Discriminator, self).__init__()

        # Choose slope for leaky relu activation functions in conv2dBlock.
        if act_type == "lrelu":
            slope = 0.2
        elif act_type == "relu":
            slope = 0.0
        else:
            print(f"WARNING (discriminator): activation type {act_type} has not been implemented - defaulting to leaky ReLU (0.2)")
            slope = 0.2

        # Build feature extractor block.
        self.features = nn.Sequential(
            Conv2dBlock(
                in_num_ch       = in_num_ch,
                out_num_ch      = base_num_f,
                feat_kern_size1 = feat_kern_size,
                lrelu_neg_slope = slope,
                norm_type       = norm_type,
                drop_first_norm = False
            ),  # [in_num_ch, 128, 128] -> [base_num_f, 64, 64]
            Conv2dBlock(
                in_num_ch       = base_num_f,
                out_num_ch      = 2 * base_num_f,
                feat_kern_size1 = feat_kern_size,
                lrelu_neg_slope = slope,
                norm_type       = norm_type,
                drop_first_norm = False
            ),  # [base_num_f, 64, 64] -> [2*base_num_f, 32, 32]
            Conv2dBlock(
                in_num_ch       = 2 * base_num_f,
                out_num_ch      = 4 * base_num_f,
                feat_kern_size1 = feat_kern_size,
                lrelu_neg_slope = slope,
                norm_type       = norm_type,
                drop_first_norm = False
            ),  # [2*base_num_f, 32, 32] -> [4*base_num_f, 16, 16]
            Conv2dBlock(
                in_num_ch       = 4 * base_num_f,
                out_num_ch      = 8 * base_num_f,
                feat_kern_size1 = feat_kern_size,
                lrelu_neg_slope = slope,
                norm_type       = norm_type,
                drop_first_norm = False
            ),  # [4*base_num_f, 16, 16] -> [8*base_num_f, 8, 8]
            Conv2dBlock(
                in_num_ch       = 8 * base_num_f,
                out_num_ch      = 8 * base_num_f,
                feat_kern_size1 = feat_kern_size,
                lrelu_neg_slope = slope,
                norm_type       = norm_type,
                drop_first_norm = False
            )   # [8*base_num_f, 8, 8] -> [8*base_num_f, 4, 4]
        )

        # Assemble the full discriminator network.
        self.classifier = nn.Sequential(
            nn.Linear(base_num_f * 8 * 4 * 4, 100),
            nn.LeakyReLU(negative_slope = slope),
            nn.Linear(100, 1)
        )
        print("Finished building VGG128Discriminator.")

    def forward(self, x):
        #print("x-shape:", x.shape)
        #x = self.features[0](x)
        #print("x-shape:", x.shape)
        #x = self.features[1](x)
        #print("x-shape:", x.shape)
        #x = self.features[2](x)
        #print("x-shape:", x.shape)
        #x = self.features[3](x)
        #print("x-shape:", x.shape)
        #x = self.features[4](x)
        #print("x-shape:", x.shape)
        x = self.features(x)
        x = x.reshape(x.shape[0], -1) # Flattening.
        #print("x-shape:", x.shape)
        return self.classifier(x)

#----------------------------------------------------------------------------