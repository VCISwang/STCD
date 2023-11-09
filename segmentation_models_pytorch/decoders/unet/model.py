from typing import Optional, Union, List

from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import (
    SegmentationModel,
    SegmentationHead,
    ClassificationHead,
)
from .decoder import UnetDecoder
import torch
from ...base import initialization as init
import torch
import torch.nn as nn
import torch.nn.functional as F
from .se import ChannelSpatialSELayer


class Unet(SegmentationModel):
    """Unet_ is a fully convolution neural network for image semantic segmentation. Consist of *encoder*
    and *decoder* parts connected with *skip connections*. Encoder extract features of different spatial
    resolution (skip connections) which are used by decoder to define accurate segmentation mask. Use *concatenation*
    for fusing decoder blocks with skip connections.

    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features
            two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and
            other pretrained weights (see table with available weights for each encoder_name)
        decoder_channels: List of integers which specify **in_channels** parameter for convolutions used in decoder.
            Length of the list should be the same as **encoder_depth**
        decoder_use_batchnorm: If **True**, BatchNorm2d layer between Conv2D and Activation layers
            is used. If **"inplace"** InplaceABN will be used, allows to decrease memory consumption.
            Available options are **True, False, "inplace"**
        decoder_attention_type: Attention module used in decoder of the model. Available options are
            **None** and **scse** (https://arxiv.org/abs/1808.08127).
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**,
                **callable** and **None**.
            Default is **None**
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax"
                    (could be **None** to return logits)

    Returns:
        ``torch.nn.Module``: Unet

    .. _Unet:
        https://arxiv.org/abs/1505.04597

    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()


class UnetSeg(torch.nn.Module):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )
        self.inchannels = in_channels
        self.encoder_channels = self.encoder.out_channels  # list(self.encoder.out_channels)

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )
        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )
        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        # self.fuse = torch.nn.Conv2d(2, 1, 3, 1, 1) # fuse decison and feature levels
        self.name = "u-{}".format(encoder_name)
        self.initialize()  # adopt default value

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        x_decode = self.decoder(*self.encoder(x))

        masks = self.segmentation_head(x_decode)
        return masks


class CDNet(torch.nn.Module):
    def __init__(
        self,
        decoder_channels=None,
        classes: int = 1,
    ):
        super().__init__()
        if decoder_channels is None:
            decoder_channels = [256, 128, 64, 32, 16]
        self.Deconv1 = Block(dim=decoder_channels[0], dim_out=decoder_channels[0])
        self.Deconv2 = Block(dim=decoder_channels[1], dim_out=decoder_channels[1])
        self.Deconv3 = Block(dim=decoder_channels[2], dim_out=decoder_channels[2])
        self.Deconv4 = Block(dim=decoder_channels[3], dim_out=decoder_channels[3])
        self.Deconv5 = Block(dim=decoder_channels[4], dim_out=decoder_channels[4])

        dim_out = 256 + 128 + 64 + 32 + 16
        self.AttBlock = AttentionBlock(dim=dim_out, dim_out=dim_out)

        self.cd1 = nn.Conv2d(dim_out, 64, kernel_size=3, padding=1)
        self.cd2 = nn.Conv2d(64, classes, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x1, x2):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        input_shape = x1[4].shape[-2:]

        diff1 = self.Deconv1(x1[0], x2[0])
        diff1 = F.interpolate(diff1, size=input_shape, mode='bilinear', align_corners=False)

        diff2 = self.Deconv2(x1[1], x2[1])
        diff2 = F.interpolate(diff2, size=input_shape, mode='bilinear', align_corners=False)

        diff3 = self.Deconv3(x1[2], x2[2])
        diff3 = F.interpolate(diff3, size=input_shape, mode='bilinear', align_corners=False)

        diff4 = self.Deconv4(x1[3], x2[3])
        diff4 = F.interpolate(diff4, size=input_shape, mode='bilinear', align_corners=False)

        diff5 = self.Deconv5(x1[4], x2[4])

        diff_all = torch.cat((diff1, diff2, diff3, diff4, diff5), dim=1)

        diff_all = self.AttBlock(diff_all)

        cd_map = self.cd2(self.relu(self.cd1(diff_all)))

        return cd_map


class Block(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.ReLU()
        )
        self.relu = nn.ReLU()

    def forward(self, x1, x2):
        # x12 = self.block(self.relu(x1 - x2))
        # x21 = self.block(self.relu(x2 - x1))
        # xmn = self.block(self.relu(torch.max(x1, x2) - torch.min(x1, x2)))
        # x = x12 + x21 + xmn
        x = torch.abs(x1 - x2)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.ReLU(),
            ChannelSpatialSELayer(num_channels=dim_out, reduction_ratio=2)
        )

    def forward(self, x):
        return self.block(x)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("BasicConv") != -1:
        torch.nn.init.normal_(m.conv.weight.data, 0.0, 0.02)
        torch.nn.init.normal_(m.bn.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bn.bias.data, 0.0)
    elif classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class SegCD(torch.nn.Module):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )
        self.inchannels = in_channels
        self.encoder_channels = self.encoder.out_channels  # list(self.encoder.out_channels)

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )
        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        # self.fuse = torch.nn.Conv2d(2, 1, 3, 1, 1) # fuse decison and feature levels
        self.name = "u-{}".format(encoder_name)

        self.initialize()  # adopt default value

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)

    def forward(self, A, B):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        x1_decode = self.decoder(*self.encoder(A))
        x2_decode = self.decoder(*self.encoder(B))

        mask_t1 = self.segmentation_head(x1_decode)
        mask_t2 = self.segmentation_head(x2_decode)

        diffea = self.segmentation_head(torch.abs(x1_decode - x2_decode))

        diffseg = torch.abs(mask_t1-mask_t2)
        # 3. fuse feature and decision levels
        change = torch.min(diffea, diffseg)  # compress false alarms
        # unchange = mask_t1+mask_t2-change

        return mask_t1, mask_t2, change


class FFCTLCD(torch.nn.Module):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )
        self.inchannels = in_channels
        self.encoder_channels = self.encoder.out_channels  # list(self.encoder.out_channels)

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )
        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )
        # cd_channels = [256, 128, 64, 32, 16]
        # self.Deconv1 = Block(dim=cd_channels[0], dim_out=cd_channels[0])
        # self.Deconv2 = Block(dim=cd_channels[1], dim_out=cd_channels[1])
        # self.Deconv3 = Block(dim=cd_channels[2], dim_out=cd_channels[2])
        # self.Deconv4 = Block(dim=cd_channels[3], dim_out=cd_channels[3])
        # self.Deconv5 = Block(dim=cd_channels[4], dim_out=cd_channels[4])
        #
        # dim_out = 256 + 128 + 64 + 32 + 16
        # self.AttBlock = AttentionBlock(dim=dim_out, dim_out=dim_out)
        #
        # self.cd1 = nn.Conv2d(dim_out, 64, kernel_size=3, padding=1)
        # self.cd2 = nn.Conv2d(64, classes, kernel_size=3, padding=1)
        # self.relu = nn.ReLU()

        # self.fuse = torch.nn.Conv2d(2, 1, 3, 1, 1) # fuse decison and feature levels
        self.name = "u-{}".format(encoder_name)

        self.initialize()  # adopt default value

        # self.Deconv1.apply(weights_init_normal)
        # self.Deconv2.apply(weights_init_normal)
        # self.Deconv3.apply(weights_init_normal)
        # self.Deconv4.apply(weights_init_normal)
        # self.Deconv5.apply(weights_init_normal)
        #
        # self.AttBlock.apply(weights_init_normal)
        # self.cd1.apply(weights_init_normal)
        # self.cd2.apply(weights_init_normal)

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)

    def forward(self, A, B):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features1 = self.encoder(A)
        features2 = self.encoder(B)

        #  1. difference in feature level
        featurediff = [torch.abs(f1-f2) for f1, f2 in zip(features1, features2)]  # N C H W
        diffea = self.segmentation_head(self.decoder(*featurediff))

        # 2. difference in decision level
        mask_t1 = self.segmentation_head(self.decoder(*features1))
        mask_t2 = self.segmentation_head(self.decoder(*features2))
        diffseg = torch.abs(mask_t1-mask_t2)

        #  3. fuse feature and decision levels
        change = torch.min(diffea, diffseg)  # compress false alarms
        return mask_t1, mask_t2, change