import torch
import torch.nn as nn
import torch.nn.functional as F
from .pyramid_vig import Stem, Downsample, Grapher, FFN, Seq, act_layer
from .ChangeFormer import MLP, conv_diff, make_prediction, UpsampleConvLayer, ResidualBlock, ConvLayer, resize


class Conv_De_Head(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=embed_dim, kernel_size=1, stride=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class EncoderV1(nn.Module):
    def __init__(self, k=9, conv="mr", act="gelu", norm="batch", bias=True, dropout=0.0, use_dilation=True,
                 epsilon=0.2, use_stochastic=False, drop_path_rate=0.0, blocks=[2, 2, 6, 2],
                 channels=[48, 96, 240, 384], num_classes=2, emb_dims=1024, img_size=256):
        super(EncoderV1, self).__init__()
        k = k
        act = act
        norm = norm
        bias = bias
        epsilon = epsilon
        stochastic = use_stochastic
        conv = conv
        emb_dims = emb_dims
        drop_path = drop_path_rate

        blocks = blocks
        self.n_blocks = sum(blocks)
        channels = channels
        reduce_ratios = [4, 2, 1, 1]
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]  # stochastic depth decay rule
        num_knn = [int(x.item()) for x in torch.linspace(k, k, self.n_blocks)]  # number of knn's k
        max_dilation = 49 // max(num_knn)

        self.stem = Stem(out_dim=channels[0], act=act)
        self.pos_embed = nn.Parameter(torch.zeros(1, channels[0], img_size // 4, img_size // 4))
        HW = img_size // 4 * img_size // 4

        self.backbone = nn.ModuleList([])
        idx = 0
        for i in range(len(blocks)):
            if i > 0:
                self.backbone.append(Downsample(channels[i - 1], channels[i]))
                HW = HW // 4
            for j in range(blocks[i]):
                self.backbone += [
                    Seq(Grapher(channels[i], num_knn[idx], min(idx // 4 + 1, max_dilation), conv, act, norm,
                                bias, stochastic, epsilon, reduce_ratios[i], n=HW, drop_path=dpr[idx],
                                relative_pos=True),
                        FFN(channels[i], channels[i] * 4, act=act, drop_path=dpr[idx])
                        )]
                idx += 1
        self.backbone = Seq(*self.backbone)

        # self.prediction = Seq(nn.Conv2d(channels[-1], 1024, 1, bias=True),
        #                       nn.BatchNorm2d(1024),
        #                       act_layer(act),
        #                       nn.Dropout(dropout),
        #                       nn.Conv2d(1024, num_classes, 1, bias=True))
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward_features(self, inputs):
        outs = []
        x = self.stem(inputs) + self.pos_embed
        B, C, H, W = x.shape

        for i in range(len(self.backbone)):
            x = self.backbone[i](x)
            if i in [1, 4, 11, 14]:
                outs.append(x)
        return outs

    def forward(self, x):
        x = self.forward_features(x)
        return x


class DecoderV1(nn.Module):
    """
    Transformer Decoder
    """

    def __init__(self, input_transform='multiple_select', in_index=[0, 1, 2, 3], align_corners=True,
                 in_channels=[32, 64, 128, 256], embedding_dim=64, output_nc=2,
                 decoder_softmax=False, feature_strides=[2, 4, 8, 16], decoder_heads="MLP"):
        super(DecoderV1, self).__init__()
        # assert
        assert len(feature_strides) == len(in_channels)
        assert min(feature_strides) == feature_strides[0]

        # settings
        self.feature_strides = feature_strides
        self.input_transform = input_transform
        self.in_index = in_index
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.output_nc = output_nc
        self.decoder_heads = decoder_heads
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        # MLP decoder heads
        if decoder_heads == "MLP":
            self.decoder_heads_c4 = MLP(input_dim=c4_in_channels, embed_dim=self.embedding_dim)
            self.decoder_heads_c3 = MLP(input_dim=c3_in_channels, embed_dim=self.embedding_dim)
            self.decoder_heads_c2 = MLP(input_dim=c2_in_channels, embed_dim=self.embedding_dim)
            self.decoder_heads_c1 = MLP(input_dim=c1_in_channels, embed_dim=self.embedding_dim)
        elif decoder_heads == "Conv":
            self.decoder_heads_c4 = Conv_De_Head(input_dim=c4_in_channels, embed_dim=self.embedding_dim)
            self.decoder_heads_c3 = Conv_De_Head(input_dim=c3_in_channels, embed_dim=self.embedding_dim)
            self.decoder_heads_c2 = Conv_De_Head(input_dim=c2_in_channels, embed_dim=self.embedding_dim)
            self.decoder_heads_c1 = Conv_De_Head(input_dim=c1_in_channels, embed_dim=self.embedding_dim)

        # convolutional Difference Modules
        self.diff_c4 = conv_diff(in_channels=2 * self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c3 = conv_diff(in_channels=2 * self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c2 = conv_diff(in_channels=2 * self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c1 = conv_diff(in_channels=2 * self.embedding_dim, out_channels=self.embedding_dim)

        # taking outputs from middle of the encoder
        self.make_pred_c4 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c3 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c2 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c1 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)

        # Final linear fusion layer
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(in_channels=self.embedding_dim * len(in_channels), out_channels=self.embedding_dim,
                      kernel_size=1),
            nn.BatchNorm2d(self.embedding_dim)
        )

        # Final predction head
        self.convd2x = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_2x = nn.Sequential(ResidualBlock(self.embedding_dim))
        self.convd1x = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_1x = nn.Sequential(ResidualBlock(self.embedding_dim))
        self.change_probability = ConvLayer(self.embedding_dim, self.output_nc, kernel_size=3, stride=1, padding=1)

        # Final activation
        self.output_softmax = decoder_softmax
        self.active = nn.Sigmoid()

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def forward(self, inputs1, inputs2):
        # Transforming encoder features (select layers)
        x_1 = self._transform_inputs(inputs1)  # len=4, 1/2, 1/4, 1/8, 1/16
        x_2 = self._transform_inputs(inputs2)  # len=4, 1/2, 1/4, 1/8, 1/16

        # img1 and img2 features
        c1_1, c2_1, c3_1, c4_1 = x_1
        c1_2, c2_2, c3_2, c4_2 = x_2

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4_1.shape

        outputs = []
        if self.decoder_heads == "MLP":
            _c4_1 = self.decoder_heads_c4(c4_1).permute(0, 2, 1).reshape(n, -1, c4_1.shape[2], c4_1.shape[3])
            _c4_2 = self.decoder_heads_c4(c4_2).permute(0, 2, 1).reshape(n, -1, c4_2.shape[2], c4_2.shape[3])
        elif self.decoder_heads == "Conv":
            _c4_1 = self.decoder_heads_c4(c4_1)
            _c4_2 = self.decoder_heads_c4(c4_2)
        _c4 = self.diff_c4(torch.cat((_c4_1, _c4_2), dim=1))
        p_c4 = self.make_pred_c4(_c4)
        outputs.append(p_c4)
        _c4_up = resize(_c4, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Stage 3: x1/16 scale
        if self.decoder_heads == "MLP":
            _c3_1 = self.decoder_heads_c3(c3_1).permute(0, 2, 1).reshape(n, -1, c3_1.shape[2], c3_1.shape[3])
            _c3_2 = self.decoder_heads_c3(c3_2).permute(0, 2, 1).reshape(n, -1, c3_2.shape[2], c3_2.shape[3])
        elif self.decoder_heads == "Conv":
            _c3_1 = self.decoder_heads_c3(c3_1)
            _c3_2 = self.decoder_heads_c3(c3_2)
        _c3 = self.diff_c3(torch.cat((_c3_1, _c3_2), dim=1)) + F.interpolate(_c4, scale_factor=2, mode="bilinear")
        p_c3 = self.make_pred_c3(_c3)
        outputs.append(p_c3)
        _c3_up = resize(_c3, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Stage 2: x1/8 scale
        if self.decoder_heads == "MLP":
            _c2_1 = self.decoder_heads_c2(c2_1).permute(0, 2, 1).reshape(n, -1, c2_1.shape[2], c2_1.shape[3])
            _c2_2 = self.decoder_heads_c2(c2_2).permute(0, 2, 1).reshape(n, -1, c2_2.shape[2], c2_2.shape[3])
        elif self.decoder_heads == "Conv":
            _c2_1 = self.decoder_heads_c2(c2_1)
            _c2_2 = self.decoder_heads_c2(c2_2)

        _c2 = self.diff_c2(torch.cat((_c2_1, _c2_2), dim=1)) + F.interpolate(_c3, scale_factor=2, mode="bilinear")
        p_c2 = self.make_pred_c2(_c2)
        outputs.append(p_c2)
        _c2_up = resize(_c2, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Stage 1: x1/4 scale
        if self.decoder_heads == "MLP":
            _c1_1 = self.decoder_heads_c1(c1_1).permute(0, 2, 1).reshape(n, -1, c1_1.shape[2], c1_1.shape[3])
            _c1_2 = self.decoder_heads_c1(c1_2).permute(0, 2, 1).reshape(n, -1, c1_2.shape[2], c1_2.shape[3])
        elif self.decoder_heads == "Conv":
            _c1_1 = self.decoder_heads_c1(c1_1)
            _c1_2 = self.decoder_heads_c1(c1_2)
        _c1 = self.diff_c1(torch.cat((_c1_1, _c1_2), dim=1)) + F.interpolate(_c2, scale_factor=2, mode="bilinear")
        p_c1 = self.make_pred_c1(_c1)
        outputs.append(p_c1)

        # Linear Fusion of difference image from all scales
        _c = self.linear_fuse(torch.cat((_c4_up, _c3_up, _c2_up, _c1), dim=1))

        # #Dropout
        # if dropout_ratio > 0:
        #     self.dropout = nn.Dropout2d(dropout_ratio)
        # else:
        #     self.dropout = None

        # Upsampling x2 (x1/2 scale)
        x = self.convd2x(_c)
        # Residual block
        x = self.dense_2x(x)
        # Upsampling x2 (x1 scale)
        x = self.convd1x(x)
        # Residual block
        x = self.dense_1x(x)

        # Final prediction
        cp = self.change_probability(x)

        outputs.append(cp)

        if self.output_softmax:
            temp = outputs
            outputs = []
            for pred in temp:
                outputs.append(self.active(pred))

        return outputs


class ChangeGNNV1(nn.Module):

    def __init__(self, input_nc=3, output_nc=2, decoder_softmax=False, embed_dim=256, decoder_heads="MLP",
                 img_size=256):
        super(ChangeGNNV1, self).__init__()
        # Transformer Encoder
        self.embed_dims = [80, 160, 400, 640]
        self.depths = [3, 3, 4, 3]  # [3, 3, 6, 18, 3]
        self.embedding_dim = embed_dim
        self.drop_rate = 0.1
        self.attn_drop = 0.1
        self.drop_path_rate = 0.1

        self.opt = None

        self.encoder = EncoderV1(k=9, conv="mr", act="gelu", norm="batch", bias=True, dropout=0.0, use_dilation=True,
                                epsilon=0.2, use_stochastic=False, drop_path_rate=0.0, blocks=[2, 2, 6, 2],
                                channels=self.embed_dims, num_classes=2, emb_dims=1024, img_size=img_size)

        # Transformer Decoder
        self.decoder = DecoderV1(input_transform='multiple_select', in_index=[0, 1, 2, 3], align_corners=False,
                                 in_channels=self.embed_dims, embedding_dim=self.embedding_dim, output_nc=output_nc,
                                 decoder_softmax=decoder_softmax, feature_strides=[2, 4, 8, 16],
                                 decoder_heads=decoder_heads)

    def forward(self, x1, x2):
        [fx1, fx2] = [self.encoder(x1), self.encoder(x2)]
        cp = self.decoder(fx1, fx2)
        return cp


class Cross_ConCat(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Cross_ConCat, self).__init__()
        self.in_channels = in_channels
        self.diff = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1, stride=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        self.conv_res = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(),
            nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels)
        )
        self.act = nn.ReLU()

    def forward(self, inputs1, inputs2):
        assert inputs1.shape == inputs2.shape
        batch, channels, height, width = inputs1.shape
        inputs = torch.ones(batch, channels * 2, height, width).to(device=inputs1.device)
        inputs[:, 0::2, :, :] = inputs1
        inputs[:, 1::2, :, :] = inputs2
        out = self.diff(inputs)
        return self.act(self.conv_res(out) + self.conv(out))


class Global_Local(nn.Module):
    def __init__(self, in_channels, out_channels=None, ratio=8):
        super(Global_Local, self).__init__()
        if out_channels == None:
            out_channels = in_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(2, 1),
                                      groups=out_channels)
        self.channel_bn = nn.BatchNorm2d(in_channels)
        self.channel_act = nn.ReLU()
        self.spatial_conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=5, padding=2)
        self.spatial_act = nn.ReLU()

        self.local_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                     groups=out_channels)
        self.local_conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1,
                                     groups=out_channels)
        self.local_conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, padding=3,
                                     groups=out_channels)
        self.local_conv4 = nn.Conv2d(in_channels=out_channels * 3, out_channels=out_channels, kernel_size=1)
        self.local_conv5 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.local_bn = nn.BatchNorm2d(out_channels)
        self.local_act = nn.ReLU()

        self.sigmoid = nn.Sigmoid()
        self.bt = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        channel_avg_out = self.avg_pool(x)
        channel_max_out = self.max_pool(x)
        channel_out = self.channel_act(
            self.channel_bn(self.channel_conv(torch.cat([channel_avg_out, channel_max_out], dim=2))))
        spatial_avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_max_out = torch.max(x, dim=1, keepdim=True, out=None)[0]
        spatial_out = self.spatial_act(self.spatial_conv(torch.cat([spatial_avg_out, spatial_max_out], dim=1)))
        channel_spatial_out = self.sigmoid(channel_out * spatial_out) * x

        local_out = self.local_conv4(torch.cat([self.local_conv1(x), self.local_conv2(x), self.local_conv3(x)], dim=1))
        local_out = self.local_conv5(self.local_act(self.local_bn(local_out)))

        return channel_spatial_out + local_out


class Upsampling(nn.Module):
    def __init__(self, in_channels, bilinear=False):
        super(Upsampling, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.up(x)
        return x


class HFFM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HFFM, self).__init__()
        self.cross_conc = Cross_ConCat(in_channels=in_channels, out_channels=out_channels)
        self.global_local = Global_Local(in_channels=out_channels)

    def forward(self, input1, input2):
        return self.global_local(self.cross_conc(input1, input2))


class VFFM(nn.Module):
    def __init__(self, in_channels=64, r=4):
        super(VFFM, self).__init__()
        inter_channels = int(in_channels // r)

        self.up = Upsampling(in_channels=in_channels, bilinear=False)

        self.global_avg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )

        self.global_max = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )

        self.local_att = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, low_feature, high_feature):
        # low_feature and high_feature have same numbers of channels
        high_feature = self.up(high_feature)
        mixed_feature = low_feature + high_feature
        wei = self.sigmoid(self.global_avg(mixed_feature) + self.global_max(mixed_feature) + self.local_att(mixed_feature))

        xo = 2 * low_feature * wei + 2 * high_feature * (1 - wei)
        return xo


class EncoderV2(nn.Module):
    def __init__(self, k=9, conv="mr", act="gelu", norm="batch", bias=True, dropout=0.0, use_dilation=True,
                 epsilon=0.2, use_stochastic=False, drop_path_rate=0.0, blocks=[2, 2, 6, 2],
                 channels=[48, 96, 240, 384], num_classes=2, emb_dims=1024, img_size=256):
        super(EncoderV2, self).__init__()
        k = k
        act = act
        norm = norm
        bias = bias
        epsilon = epsilon
        stochastic = use_stochastic
        conv = conv
        emb_dims = emb_dims
        drop_path = drop_path_rate

        blocks = blocks
        self.n_blocks = sum(blocks)
        channels = channels
        reduce_ratios = [4, 2, 1, 1]
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]  # stochastic depth decay rule
        num_knn = [int(x.item()) for x in torch.linspace(k, k, self.n_blocks)]  # number of knn's k
        max_dilation = 49 // max(num_knn)

        self.stem = Stem(out_dim=channels[0], act=act)
        self.pos_embed = nn.Parameter(torch.zeros(1, channels[0], img_size // 4, img_size // 4))
        HW = img_size // 4 * img_size // 4

        self.backbone = nn.ModuleList([])
        idx = 0
        for i in range(len(blocks)):
            if i > 0:
                self.backbone.append(Downsample(channels[i - 1], channels[i]))
                HW = HW // 4
            for j in range(blocks[i]):
                self.backbone += [
                    Seq(Grapher(channels[i], num_knn[idx], min(idx // 4 + 1, max_dilation), conv, act, norm,
                                bias, stochastic, epsilon, reduce_ratios[i], n=HW, drop_path=dpr[idx],
                                relative_pos=True),
                        FFN(channels[i], channels[i] * 4, act=act, drop_path=dpr[idx])
                        )]
                idx += 1
        self.backbone = Seq(*self.backbone)

        # self.prediction = Seq(nn.Conv2d(channels[-1], 1024, 1, bias=True),
        #                       nn.BatchNorm2d(1024),
        #                       act_layer(act),
        #                       nn.Dropout(dropout),
        #                       nn.Conv2d(1024, num_classes, 1, bias=True))
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward_features(self, inputs):
        outs = []
        x = self.stem(inputs) + self.pos_embed
        B, C, H, W = x.shape
        for i in range(len(self.backbone)):
            x = self.backbone[i](x)
            if i in [1, 4, 11, 14]:
                outs.append(x)
        return outs

    def forward(self, x):
        x = self.forward_features(x)
        return x


class DecoderV2(nn.Module):
    """
    Transformer Decoder
    """

    def __init__(self, input_transform='multiple_select', in_index=[0, 1, 2, 3], align_corners=True,
                 in_channels=[32, 64, 128, 256], embedding_dim=64, output_nc=2,
                 decoder_softmax=False, feature_strides=[2, 4, 8, 16], decoder_heads="MLP"):
        super(DecoderV2, self).__init__()
        # assert
        assert len(feature_strides) == len(in_channels)
        assert min(feature_strides) == feature_strides[0]

        # settings
        self.feature_strides = feature_strides
        self.input_transform = input_transform
        self.in_index = in_index
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.output_nc = output_nc
        self.decoder_heads = decoder_heads
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        # convolutional Difference Modules
        self.hffm4 = HFFM(in_channels=c4_in_channels, out_channels=self.embedding_dim)
        self.hffm3 = HFFM(in_channels=c3_in_channels, out_channels=self.embedding_dim)
        self.hffm2 = HFFM(in_channels=c2_in_channels, out_channels=self.embedding_dim)
        self.hffm1 = HFFM(in_channels=c1_in_channels, out_channels=self.embedding_dim)

        self.vffm3 = VFFM(in_channels=self.embedding_dim)
        self.vffm2 = VFFM(in_channels=self.embedding_dim)
        self.vffm1 = VFFM(in_channels=self.embedding_dim)

        # self.spp4 = ASPP(in_channels=self.embedding_dim, atrous_rates=[2, 4, 6])
        # self.spp3 = ASPP(in_channels=self.embedding_dim, atrous_rates=[4, 6, 12])
        # taking outputs from middle of the encoder
        # self.make_pred_c4 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        # self.make_pred_c3 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        # self.make_pred_c2 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        # self.make_pred_c1 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)

        # Final linear fusion layer
        # self.linear_fuse = nn.Sequential(
        #     nn.Conv2d(in_channels=self.embedding_dim * len(in_channels), out_channels=self.embedding_dim,
        #               kernel_size=1),
        #     nn.BatchNorm2d(self.embedding_dim)
        # )

        # Final predction head
        self.convd2x = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_2x = nn.Sequential(ResidualBlock(self.embedding_dim))
        self.convd1x = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_1x = nn.Sequential(ResidualBlock(self.embedding_dim))
        self.change_probability = ConvLayer(self.embedding_dim, self.output_nc, kernel_size=3, stride=1, padding=1)

        # Final activation
        self.output_softmax = decoder_softmax
        self.active = nn.Sigmoid()

    def forward(self, inputs1, inputs2):
        c1_1, c2_1, c3_1, c4_1 = inputs1
        c1_2, c2_2, c3_2, c4_2 = inputs2

        n, _, h, w = c4_1.shape
        outputs = []
        # c4 = self.hffm4(c4_1, c4_2)
        # c3 = self.hffm3(c3_1, c3_2)
        # c3 = self.vffm3(c3, c4)
        # c2 = self.hffm2(c2_1, c2_2)
        # c2 = self.vffm2(c2, c3)
        # c1 = self.hffm1(c1_1, c1_2)
        # c1 = self.vffm1(c1, c2)
        c = self.vffm1(self.hffm1(c1_1, c1_2), self.vffm2(self.hffm2(c2_1, c2_2), self.vffm3(self.hffm3(c3_1, c3_2), self.hffm4(c4_1, c4_2))))
        # Upsampling x2 (x1/2 scale)
        x = self.convd2x(c)
        # Residual block
        x = self.dense_2x(x)
        # Upsampling x2 (x1 scale)
        x = self.convd1x(x)
        # Residual block
        x = self.dense_1x(x)

        # Final prediction
        cp = self.change_probability(x)

        outputs.append(cp)

        if self.output_softmax:
            temp = outputs
            outputs = []
            for pred in temp:
                outputs.append(self.active(pred))

        return outputs


class ChangeGNNV2(nn.Module):

    def __init__(self, input_nc=3, output_nc=2, decoder_softmax=False, embed_dim=256, decoder_heads="MLP",
                 img_size=256):
        super(ChangeGNNV2, self).__init__()
        # Transformer Encoder
        self.embed_dims = [80, 160, 400, 640]
        self.depths = [3, 3, 4, 3]  # [3, 3, 6, 18, 3]
        self.embedding_dim = embed_dim
        self.drop_rate = 0.1
        self.attn_drop = 0.1
        self.drop_path_rate = 0.1

        self.opt = None

        self.encoder = EncoderV2(k=9, conv="mr", act="gelu", norm="batch", bias=True, dropout=0.0,
                                use_dilation=True,
                                epsilon=0.2, use_stochastic=False, drop_path_rate=0.0, blocks=[2, 2, 6, 2],
                                channels=self.embed_dims, num_classes=2, emb_dims=1024)

        # Transformer Decoder
        self.decoder = DecoderV2(input_transform='multiple_select', in_index=[0, 1, 2, 3], align_corners=False,
                                 in_channels=self.embed_dims, embedding_dim=self.embedding_dim,
                                 output_nc=output_nc,
                                 decoder_softmax=decoder_softmax, feature_strides=[2, 4, 8, 16],
                                 decoder_heads=decoder_heads)

    def forward(self, x1, x2):
        [fx1, fx2] = [self.encoder(x1), self.encoder(x2)]
        cp = self.decoder(fx1, fx2)
        return cp


class Sub(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Sub, self).__init__()
        self.in_channels = in_channels
        self.conv_res = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(),
            nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels)
        )
        self.act = nn.ReLU()

    def forward(self, inputs1, inputs2):
        assert inputs1.shape == inputs2.shape
        batch, channels, height, width = inputs1.shape
        out = torch.sub(inputs1, inputs2)
        return self.act(self.conv_res(out) + self.conv(out))


class Abs(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Abs, self).__init__()
        self.in_channels = in_channels
        self.conv_res = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(),
            nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels)
        )
        self.act = nn.ReLU()

    def forward(self, inputs1, inputs2):
        assert inputs1.shape == inputs2.shape
        batch, channels, height, width = inputs1.shape
        out = torch.abs(torch.sub(inputs1, inputs2))
        return self.act(self.conv_res(out) + self.conv(out))


class Conc(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conc, self).__init__()
        self.in_channels = in_channels
        self.diff = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        self.conv_res = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(),
            nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels)
        )
        self.act = nn.ReLU()

    def forward(self, inputs1, inputs2):
        assert inputs1.shape == inputs2.shape
        batch, channels, height, width = inputs1.shape
        out = self.diff(torch.cat([inputs1, inputs2], dim=1))
        return self.act(self.conv_res(out) + self.conv(out))


class HFFM_Compare(nn.Module):
    def __init__(self, in_channels, out_channels, diff_mode="sub"):
        super(HFFM_Compare, self).__init__()
        if diff_mode == "sub":
            self.diff = Sub(in_channels=in_channels, out_channels=out_channels)
        elif diff_mode == "abs":
            self.diff = Abs(in_channels=in_channels, out_channels=out_channels)
        elif diff_mode == "conc":
            self.diff = Conc(in_channels=in_channels, out_channels=out_channels)
        self.global_local = Global_Local(in_channels=out_channels)

    def forward(self, input1, input2):
        return self.global_local(self.diff(input1, input2))


class DecoderV2_Compare(nn.Module):
    """
    Transformer Decoder
    """

    def __init__(self, input_transform='multiple_select', in_index=[0, 1, 2, 3], align_corners=True,
                 in_channels=[32, 64, 128, 256], embedding_dim=64, output_nc=2,
                 decoder_softmax=False, feature_strides=[2, 4, 8, 16], decoder_heads="MLP", diff_mode="sub"):
        super(DecoderV2_Compare, self).__init__()
        # assert
        assert len(feature_strides) == len(in_channels)
        assert min(feature_strides) == feature_strides[0]

        # settings
        self.feature_strides = feature_strides
        self.input_transform = input_transform
        self.in_index = in_index
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.output_nc = output_nc
        self.decoder_heads = decoder_heads
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        # convolutional Difference Modules
        self.hffm4 = HFFM_Compare(in_channels=c4_in_channels, out_channels=self.embedding_dim, diff_mode=diff_mode)
        self.hffm3 = HFFM_Compare(in_channels=c3_in_channels, out_channels=self.embedding_dim, diff_mode=diff_mode)
        self.hffm2 = HFFM_Compare(in_channels=c2_in_channels, out_channels=self.embedding_dim, diff_mode=diff_mode)
        self.hffm1 = HFFM_Compare(in_channels=c1_in_channels, out_channels=self.embedding_dim, diff_mode=diff_mode)

        self.vffm3 = VFFM(in_channels=self.embedding_dim)
        self.vffm2 = VFFM(in_channels=self.embedding_dim)
        self.vffm1 = VFFM(in_channels=self.embedding_dim)

        # self.spp4 = ASPP(in_channels=self.embedding_dim, atrous_rates=[2, 4, 6])
        # self.spp3 = ASPP(in_channels=self.embedding_dim, atrous_rates=[4, 6, 12])
        # taking outputs from middle of the encoder
        # self.make_pred_c4 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        # self.make_pred_c3 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        # self.make_pred_c2 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        # self.make_pred_c1 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)

        # Final linear fusion layer
        # self.linear_fuse = nn.Sequential(
        #     nn.Conv2d(in_channels=self.embedding_dim * len(in_channels), out_channels=self.embedding_dim,
        #               kernel_size=1),
        #     nn.BatchNorm2d(self.embedding_dim)
        # )

        # Final predction head
        self.convd2x = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_2x = nn.Sequential(ResidualBlock(self.embedding_dim))
        self.convd1x = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_1x = nn.Sequential(ResidualBlock(self.embedding_dim))
        self.change_probability = ConvLayer(self.embedding_dim, self.output_nc, kernel_size=3, stride=1, padding=1)

        # Final activation
        self.output_softmax = decoder_softmax
        self.active = nn.Sigmoid()

    def forward(self, inputs1, inputs2):
        c1_1, c2_1, c3_1, c4_1 = inputs1
        c1_2, c2_2, c3_2, c4_2 = inputs2

        n, _, h, w = c4_1.shape
        outputs = []
        # c4 = self.hffm4(c4_1, c4_2)
        # c3 = self.hffm3(c3_1, c3_2)
        # c3 = self.vffm3(c3, c4)
        # c2 = self.hffm2(c2_1, c2_2)
        # c2 = self.vffm2(c2, c3)
        # c1 = self.hffm1(c1_1, c1_2)
        # c1 = self.vffm1(c1, c2)
        c = self.vffm1(self.hffm1(c1_1, c1_2), self.vffm2(self.hffm2(c2_1, c2_2), self.vffm3(self.hffm3(c3_1, c3_2), self.hffm4(c4_1, c4_2))))
        # Upsampling x2 (x1/2 scale)
        x = self.convd2x(c)
        # Residual block
        x = self.dense_2x(x)
        # Upsampling x2 (x1 scale)
        x = self.convd1x(x)
        # Residual block
        x = self.dense_1x(x)

        # Final prediction
        cp = self.change_probability(x)

        outputs.append(cp)

        if self.output_softmax:
            temp = outputs
            outputs = []
            for pred in temp:
                outputs.append(self.active(pred))

        return outputs


class ChangeGNNV2_Compare(nn.Module):

    def __init__(self, input_nc=3, output_nc=2, decoder_softmax=False, embed_dim=256, decoder_heads="MLP",
                 img_size=256, diff_mode="sub"):
        super(ChangeGNNV2_Compare, self).__init__()
        # Transformer Encoder
        self.embed_dims = [80, 160, 400, 640]
        self.depths = [3, 3, 4, 3]  # [3, 3, 6, 18, 3]
        self.embedding_dim = embed_dim
        self.drop_rate = 0.1
        self.attn_drop = 0.1
        self.drop_path_rate = 0.1

        self.opt = None

        self.encoder = EncoderV2(k=9, conv="mr", act="gelu", norm="batch", bias=True, dropout=0.0,
                                use_dilation=True,
                                epsilon=0.2, use_stochastic=False, drop_path_rate=0.0, blocks=[2, 2, 6, 2],
                                channels=self.embed_dims, num_classes=2, emb_dims=1024)

        # Transformer Decoder
        self.decoder = DecoderV2_Compare(input_transform='multiple_select', in_index=[0, 1, 2, 3], align_corners=False,
                                 in_channels=self.embed_dims, embedding_dim=self.embedding_dim,
                                 output_nc=output_nc,
                                 decoder_softmax=decoder_softmax, feature_strides=[2, 4, 8, 16],
                                 decoder_heads=decoder_heads,
                                 diff_mode=diff_mode)

    def forward(self, x1, x2):
        [fx1, fx2] = [self.encoder(x1), self.encoder(x2)]
        cp = self.decoder(fx1, fx2)
        return cp
























class conv_diff_V20(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv_diff_V20, self).__init__()
        self.in_channels = in_channels
        self.diff = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, stride=1, groups=in_channels//2),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU()
        )
        self.conv_res = nn.Sequential(
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels // 2, out_channels // 2, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(),
            nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels)
        )
        self.act = nn.ReLU()

    def forward(self, inputs1, inputs2):
        assert inputs1.shape == inputs2.shape
        batch, channels, height, width = inputs1.shape
        inputs = torch.ones(batch, channels * 2, height, width).to(device=inputs1.device)
        inputs[:, 0::2, :, :] = inputs1
        inputs[:, 1::2, :, :] = inputs2
        out = self.diff(inputs)
        return self.act(self.conv_res(out) + self.conv(out))


class csam_V20(nn.Module):
    def __init__(self, in_channels, out_channels=None, ratio=8):
        super(csam_V20, self).__init__()
        if out_channels == None:
            out_channels = in_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv1_1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(2, 1),
                               groups=in_channels)
        self.batch_normal1 = nn.BatchNorm2d(in_channels)
        self.act = nn.GELU()

        self.liner1 = nn.Linear(in_channels, in_channels // ratio, bias=False)
        self.relu1 = nn.ReLU()
        self.liner2 = nn.Linear(in_channels // ratio, out_channels)


        self.conv2_1 = nn.Conv2d(2, 1, 3, padding=1, bias=False)
        self.conv2_2 = nn.Conv2d(1, 1, 3, padding=1, bias=False)

        # self.conv3_1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1, groups=in_channels)
        # self.conv3_2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, groups=in_channels)

        self.sigmoid = nn.Sigmoid()
        self.bt = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        channel_avg_out = self.avg_pool(x)
        channel_max_out = self.max_pool(x)
        channel_out = self.act(self.batch_normal1(self.conv1_1(torch.cat([channel_avg_out, channel_max_out], dim=2))))
        channel_out = self.liner2(self.relu1(self.liner1(channel_out.permute(0, 2, 3, 1)))).permute(0, 3, 1, 2)

        spatial_avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_max_out = torch.max(x, dim=1, keepdim=True, out=None)[0]
        spatial_out = self.conv2_2(self.relu1(self.conv2_1(torch.cat([spatial_avg_out, spatial_max_out], dim=1))))

        # value = self.conv3_2(self.relu1(self.batch_normal(self.conv3_1(x))))
        # return self.bt(torch.sigmoid(channel_out + spatial_out) * x)
        return self.bt(((torch.sigmoid(channel_out) + torch.sigmoid(spatial_out)) * x))

class AFF(nn.Module):
    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo


class EncoderVIG_V20_2(nn.Module):
    def __init__(self, k=9, conv="mr", act="gelu", norm="batch", bias=True, dropout=0.0, use_dilation=True,
                 epsilon=0.2, use_stochastic=False, drop_path_rate=0.0, blocks=[2, 2, 6, 2],
                 channels=[48, 96, 240, 384], num_classes=2, emb_dims=1024):
        super(EncoderVIG_V20_2, self).__init__()
        k = k
        act = act
        norm = norm
        bias = bias
        epsilon = epsilon
        stochastic = use_stochastic
        conv = conv
        emb_dims = emb_dims
        drop_path = drop_path_rate

        blocks = blocks
        self.n_blocks = sum(blocks)
        channels = channels
        reduce_ratios = [4, 2, 1, 1]
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]  # stochastic depth decay rule
        num_knn = [int(x.item()) for x in torch.linspace(k, k, self.n_blocks)]  # number of knn's k
        max_dilation = 49 // max(num_knn)

        self.stem = Stem(out_dim=channels[0], act=act)
        self.pos_embed = nn.Parameter(torch.zeros(1, channels[0], 256 // 4, 256 // 4))
        HW = 256 // 4 * 256 // 4

        self.backbone = nn.ModuleList([])
        idx = 0
        for i in range(len(blocks)):
            if i > 0:
                self.backbone.append(Downsample(channels[i - 1], channels[i]))
                HW = HW // 4
            for j in range(blocks[i]):
                self.backbone += [
                    Seq(Grapher(channels[i], num_knn[idx], min(idx // 4 + 1, max_dilation), conv, act, norm,
                                bias, stochastic, epsilon, reduce_ratios[i], n=HW, drop_path=dpr[idx],
                                relative_pos=True),
                        FFN(channels[i], channels[i] * 4, act=act, drop_path=dpr[idx])
                        )]
                idx += 1
        self.backbone = Seq(*self.backbone)

        # self.prediction = Seq(nn.Conv2d(channels[-1], 1024, 1, bias=True),
        #                       nn.BatchNorm2d(1024),
        #                       act_layer(act),
        #                       nn.Dropout(dropout),
        #                       nn.Conv2d(1024, num_classes, 1, bias=True))
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward_features(self, inputs):
        outs = []
        x = self.stem(inputs) + self.pos_embed
        B, C, H, W = x.shape
        for i in range(len(self.backbone)):
            x = self.backbone[i](x)
            if i in [1, 4, 11, 14]:
                outs.append(x)
        return outs

    def forward(self, x):
        x = self.forward_features(x)
        return x


class DecoderVIG_V20_2(nn.Module):
    """
    Transformer Decoder
    """

    def __init__(self, input_transform='multiple_select', in_index=[0, 1, 2, 3], align_corners=True,
                 in_channels=[32, 64, 128, 256], embedding_dim=64, output_nc=2,
                 decoder_softmax=False, feature_strides=[2, 4, 8, 16], decoder_heads="MLP"):
        super(DecoderVIG_V20_2, self).__init__()
        # assert
        assert len(feature_strides) == len(in_channels)
        assert min(feature_strides) == feature_strides[0]

        # settings
        self.feature_strides = feature_strides
        self.input_transform = input_transform
        self.in_index = in_index
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.output_nc = output_nc
        self.decoder_heads = decoder_heads
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        # convolutional Difference Modules
        self.diff_c4 = conv_diff_V20(in_channels=2 * c4_in_channels, out_channels=self.embedding_dim)
        self.diff_c3 = conv_diff_V20(in_channels=2 * c3_in_channels, out_channels=self.embedding_dim)
        self.diff_c2 = conv_diff_V20(in_channels=2 * c2_in_channels, out_channels=self.embedding_dim)
        self.diff_c1 = conv_diff_V20(in_channels=2 * c1_in_channels, out_channels=self.embedding_dim)

        # self.spp4 = ASPP(in_channels=self.embedding_dim, atrous_rates=[2, 4, 6])
        # self.spp3 = ASPP(in_channels=self.embedding_dim, atrous_rates=[4, 6, 12])
        # taking outputs from middle of the encoder
        # self.make_pred_c4 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        # self.make_pred_c3 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        # self.make_pred_c2 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        # self.make_pred_c1 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.trans_conv4 = nn.ConvTranspose2d(self.embedding_dim, self.embedding_dim, kernel_size=2, stride=2)
        self.trans_conv3 = nn.ConvTranspose2d(self.embedding_dim, self.embedding_dim, kernel_size=2, stride=2)
        self.trans_conv2 = nn.ConvTranspose2d(self.embedding_dim, self.embedding_dim, kernel_size=2, stride=2)

        self.csam4 = csam_V20(in_channels=self.embedding_dim)
        self.csam3 = csam_V20(in_channels=self.embedding_dim)
        self.csam2 = csam_V20(in_channels=self.embedding_dim)
        self.csam1 = csam_V20(in_channels=self.embedding_dim)

        self.aff3 = AFF(channels=self.embedding_dim, r=4)
        self.aff2 = AFF(channels=self.embedding_dim, r=4)
        self.aff1 = AFF(channels=self.embedding_dim, r=4)

        # Final linear fusion layer
        # self.linear_fuse = nn.Sequential(
        #     nn.Conv2d(in_channels=self.embedding_dim * len(in_channels), out_channels=self.embedding_dim,
        #               kernel_size=1),
        #     nn.BatchNorm2d(self.embedding_dim)
        # )

        # Final predction head
        self.convd2x = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_2x = nn.Sequential(ResidualBlock(self.embedding_dim))
        self.convd1x = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_1x = nn.Sequential(ResidualBlock(self.embedding_dim))
        self.change_probability = ConvLayer(self.embedding_dim, self.output_nc, kernel_size=3, stride=1, padding=1)

        # Final activation
        self.output_softmax = decoder_softmax
        self.active = nn.Sigmoid()

    def forward(self, inputs1, inputs2):
        c1_1, c2_1, c3_1, c4_1 = inputs1
        c1_2, c2_2, c3_2, c4_2 = inputs2

        n, _, h, w = c4_1.shape
        outputs = []
        c4 = self.diff_c4(c4_1, c4_2)
        c4 = self.csam4(c4)
        # c4 = self.spp4(c4)
        c4 = self.trans_conv4(c4)
        # p_c4 = self.make_pred_c4(_c4)
        # outputs.append(p_c4)
        # _c4_up = resize(_c4, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        c3 = self.diff_c3(c3_1, c3_2)
        c3 = self.csam3(c3)
        # c3 = self.spp3(c3)
        c3 = self.trans_conv3(self.aff3(c3, c4))
        # p_c3 = self.make_pred_c3(_c3)
        # outputs.append(p_c3)
        # _c3_up = resize(_c3, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        c2 = self.diff_c2(c2_1, c2_2)
        c2 = self.csam2(c2)
        c2 = self.trans_conv2(self.aff2(c2, c3))
        # p_c2 = self.make_pred_c2(_c2)
        # outputs.append(self.make_pred_c2(c2))
        # _c2_up = resize(_c2, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Stage 1: x1/4 scale
        c1 = self.diff_c1(c1_1, c1_2)
        c1 = self.csam1(c1)
        c1 = self.aff1(c1, c2)

        # p_c1 = self.make_pred_c1(_c1)
        # outputs.append(self.make_pred_c1(c1))

        # Linear Fusion of difference image from all scales
        # _c = self.linear_fuse(torch.cat((_c4_up, _c3_up, _c2_up, _c1), dim=1))

        # #Dropout
        # if dropout_ratio > 0:
        #     self.dropout = nn.Dropout2d(dropout_ratio)
        # else:
        #     self.dropout = None

        # Upsampling x2 (x1/2 scale)
        x = self.convd2x(c1)
        # Residual block
        x = self.dense_2x(x)
        # Upsampling x2 (x1 scale)
        x = self.convd1x(x)
        # Residual block
        x = self.dense_1x(x)

        # Final prediction
        cp = self.change_probability(x)

        outputs.append(cp)

        if self.output_softmax:
            temp = outputs
            outputs = []
            for pred in temp:
                outputs.append(self.active(pred))

        return outputs


class VIG_V20_2(nn.Module):

    def __init__(self, input_nc=3, output_nc=2, decoder_softmax=False, embed_dim=256, decoder_heads="MLP"):
        super(VIG_V20_2, self).__init__()
        # Transformer Encoder
        self.embed_dims = [80, 160, 400, 640]
        self.depths = [3, 3, 4, 3]  # [3, 3, 6, 18, 3]
        self.embedding_dim = embed_dim
        self.drop_rate = 0.1
        self.attn_drop = 0.1
        self.drop_path_rate = 0.1

        self.opt = None

        self.VIG_x2 = EncoderVIG_V20_2(k=9, conv="mr", act="gelu", norm="batch", bias=True, dropout=0.0,
                                       use_dilation=True,
                                       epsilon=0.2, use_stochastic=False, drop_path_rate=0.0, blocks=[2, 2, 6, 2],
                                       channels=self.embed_dims, num_classes=2, emb_dims=1024)

        # Transformer Decoder
        self.TDec_x2 = DecoderVIG_V20_2(input_transform='multiple_select', in_index=[0, 1, 2, 3], align_corners=False,
                                        in_channels=self.embed_dims, embedding_dim=self.embedding_dim,
                                        output_nc=output_nc,
                                        decoder_softmax=decoder_softmax, feature_strides=[2, 4, 8, 16],
                                        decoder_heads=decoder_heads)

    def forward(self, x1, x2):
        [fx1, fx2] = [self.VIG_x2(x1), self.VIG_x2(x2)]
        cp = self.TDec_x2(fx1, fx2)

        # # Save to mat
        # save_to_mat(x1, x2, fx1, fx2, cp, "ChangeFormerV4")

        # exit()
        return cp