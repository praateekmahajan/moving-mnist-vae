import os

import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F

# from pixlcnn import *
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

MNIST_PATH = 'data/mnist'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# replace this with torch.distributions
def kl_divergence(encoding_mu, encoding_logvar):
    return -0.5 * torch.sum(encoding_logvar - (encoding_logvar).exp() - encoding_mu.pow(2) + 1)


class VAE_Encoder(nn.Module):
    def __init__(self, in_channels, intermediate_channels, z_dimensions, need_logvar=True, input_image_size=32):
        super(VAE_Encoder, self).__init__()
        self.input_image_size = input_image_size
        self.conv1 = nn.Conv2d(in_channels, intermediate_channels, kernel_size=5, stride=2, padding=2, bias=False)
        self.conv2 = nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.conv3 = nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.conv4 = nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.conv5 = nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.conv6 = nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, stride=1, padding=1,
                               bias=False)

        self.conv7 = nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.bn3 = nn.BatchNorm2d(intermediate_channels)
        self.bn4 = nn.BatchNorm2d(intermediate_channels)
        self.bn5 = nn.BatchNorm2d(intermediate_channels)
        self.bn6 = nn.BatchNorm2d(intermediate_channels)
        self.bn7 = nn.BatchNorm2d(intermediate_channels)

        if input_image_size > 32:
            self.conv8 = nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, stride=1, padding=1,
                                   bias=False)
            self.bn8 = nn.BatchNorm2d(intermediate_channels)
            self.conv9 = nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, stride=2, padding=1,
                                   bias=False)
            self.bn9 = nn.BatchNorm2d(intermediate_channels)

        self.conv_mu = nn.Conv2d(intermediate_channels, z_dimensions, kernel_size=3, stride=2, padding=1, bias=False)

        if need_logvar == True:
            self.conv_logvar = nn.Conv2d(intermediate_channels, z_dimensions, kernel_size=3, stride=2, padding=1,
                                         bias=False)
        else:
            self.conv_logvar = None
            # N x Z x 2 x 2

    def forward(self, x):
        '''Block'''
        x_1 = F.relu(self.bn1(self.conv1(x)))  # Stride, 32/2
        x_2 = F.relu(self.bn2(self.conv2(x_1)))  # No stride

        '''Block'''
        x_3 = F.relu(self.bn3(self.conv3(x_1 + x_2)))  # Stride 32/4 = 8, 64/4
        x_4 = F.relu(self.bn4(self.conv4(x_3)))  # No Stride

        '''Block'''
        x_5 = F.relu(self.bn5(self.conv5(x_4 + x_3)))  # Stride 32/8 = 4, 64/8
        x_6 = F.relu(self.bn6(self.conv6(x_5)))  # no stride

        '''Block'''
        x_7 = F.relu(self.bn7(self.conv7(x_6 + x_5)))  # stride  32/16 = 2, 64/16 = 4
        if self.input_image_size > 32:
            x_8 = F.relu(self.bn8(self.conv8(x_7)))  # Conv7 has no stride
            '''Add'''
            x_9 = F.relu(self.bn9(self.conv9(x_8 + x_7)))  # Conv8 has stride  64/32
            final = x_9

        else:
            final = x_7

        mu = self.conv_mu(final)  # 32/64
        logvar = None
        if self.conv_logvar is not None:
            logvar = self.conv_logvar(final)

        return mu, logvar

    def rsample(self, mu, logvar):
        m = Normal(mu, torch.exp(logvar * 0.5))
        return m.rsample()


class VAE_Decoder(nn.Module):
    def __init__(self, in_channels, intermediate_channels, out_channels, input_image_size=32):
        super(VAE_Decoder, self).__init__()
        self.input_image_size = input_image_size

        self.conv1 = nn.Conv2d(in_channels, intermediate_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.conv3 = nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.conv4 = nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, stride=1, padding=1,
                               bias=False)
        if input_image_size > 32:
            self.conv5 = nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, stride=1, padding=1,
                                   bias=False)
        self.conv6 = nn.Conv2d(intermediate_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        x = F.relu(self.conv1(F.interpolate(x, scale_factor=2)))  # 2
        x = F.relu(self.conv2(F.interpolate(x, scale_factor=2)))  # 4
        x = F.relu(self.conv3(F.interpolate(x, scale_factor=2)))  # 8
        x = F.relu(self.conv4(F.interpolate(x, scale_factor=2)))  # 16
        if self.input_image_size > 32:
            x = F.relu(self.conv5(F.interpolate(x, scale_factor=2)))  # 32
        x = self.conv6(F.interpolate(x, scale_factor=2))  # 64
        return x


class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)


class PixelCNN(nn.Module):
    def __init__(self, in_channels, intermediate_channels, out_channels, layers=4, activation="ReLu"):
        super(PixelCNN, self).__init__()
        self.bn = nn.ModuleList([nn.InstanceNorm2d(intermediate_channels)] * (layers - 1))
        self.bn.append(nn.InstanceNorm2d(out_channels))
        self.bn1 = nn.InstanceNorm2d(in_channels)

        self.layers = []
        for i in range(layers):
            if i == 0:
                self.layers.append(MaskedConv2d('A', in_channels, intermediate_channels, 7, 1, 3, bias=True))
            elif i == layers - 1:
                self.layers.append(MaskedConv2d('B', intermediate_channels, out_channels, 7, 1, 3, bias=True))
            else:
                self.layers.append(MaskedConv2d('B', intermediate_channels, intermediate_channels, 7, 1, 3, bias=True))
        self.layers = nn.ModuleList(self.layers)
        if activation == "ReLu":
            self.lu = nn.ReLU()
        elif activation == "Elu":
            self.lu = nn.ELU()

    def forward(self, x):
        x = self.bn1(x)
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = self.bn[i](x)
            x = self.lu(x)
        x = self.layers[i + 1](x)
        return x


class VAE(nn.Module):
    def __init__(self, in_channels, intermediate_channels, decoder_out_channels=1, pixelcnn_out_channels=2,
                 z_dimension=32,
                 pixelcnn=True, only_pixelcnn=True, pixelcnn_layers=4, pixelcnn_activation="ReLu", nll=1, kl=1, mmd=0,
                 require_rsample=True, sigma_decoder=0.1, input_image_size=64):
        '''

        Args:
            in_channels: In channels for the VAE (1 for MNIST, 3 for CIFAR)
            intermediate_channels: Intermediate Channels for Encoder/Decoder/PixelCNN
            out_channels: Out channels for the VAE (1 or 256 for MNIST)
            z_dimension: Bottleneck dimension for the VAE (default : 32)
            nll: Coefficient of the nll term in loss
            kl: Coefficient of the klterm in loss
            mmd: Coefficient of the mmd term in loss
            require_rsample: Is it required for the output of decoder to be follow normal distribution?
            sigma_decoder: If decoder follows normal distribution, with how much standard deviation do we sample p(x/z)
        '''

        super(VAE, self).__init__()
        self.in_channels = in_channels
        self.z_dimensions = z_dimension
        self.decoder_out_channels = decoder_out_channels
        self.pixelcnn_out_channels = pixelcnn_out_channels
        self.num_pixelcnn_layers = pixelcnn_layers
        self.require_rsample = require_rsample

        self.nll, self.kl, self.mmd = nll, kl, mmd
        self.sigma_decoder = sigma_decoder
        self.input_image_size = input_image_size
        self.only_pixelcnn = only_pixelcnn

        if not only_pixelcnn:
            if pixelcnn:
                # TODO : Add abilitiy to choose if you to stack 256 channels of VAE output with 1 channel of input
                # TODO : Or stack 256 channel of VAE output with 256 channel of VAE input
                self.pixelcnn = PixelCNN(decoder_out_channels + in_channels, intermediate_channels,
                                         pixelcnn_out_channels,
                                         pixelcnn_layers, \
                                         pixelcnn_activation)

            else:
                self.pixelcnn = None

            self.encoder = VAE_Encoder(in_channels, intermediate_channels, z_dimension, require_rsample,
                                       input_image_size)
            self.decoder = VAE_Decoder(z_dimension, intermediate_channels, decoder_out_channels, input_image_size)
            if input_image_size > 32:
                self.adjust = (64 - input_image_size) // 2
            else:
                self.adjust = (32 - input_image_size) // 2

        else:
            self.pixelcnn = PixelCNN(in_channels, intermediate_channels, pixelcnn_out_channels, pixelcnn_layers, \
                                     pixelcnn_activation)

    def forward(self, x, sample=None):
        mu, logvar, encoding, reconstruction = None, None, None, None
        if not self.only_pixelcnn:
            mu, logvar = self.encoder(x)

            if self.require_rsample:
                encoding = self.encoder.rsample(mu, logvar)
            else:
                encoding = mu

            decoder_output = self.decoder(encoding)

            if self.adjust != 0:
                decoder_output = decoder_output[:, :, self.adjust:-self.adjust, self.adjust:-self.adjust]

            if self.pixelcnn is not None:
                if self.training:
                    concat = torch.cat([decoder_output, x], dim=1)
                else:
                    concat = torch.cat([decoder_output, sample], dim=1)
                reconstruction = self.pixelcnn(concat)
            else:
                reconstruction = decoder_output

        else:
            reconstruction = self.pixelcnn(x)

        return mu, logvar, encoding, reconstruction

    def get_z_image(self, encoding):
        decoder_output = self.decoder(encoding)
        if self.adjust != 0:
            decoder_output = decoder_output[:, :, self.adjust:-self.adjust, self.adjust:-self.adjust]
        return decoder_output

    def run_pixelcnn(self, concat):
        return self.pixelcnn(concat)

    def get_reconstruction(self, encoding, sample=None):
        decoder_output = self.decoder(encoding)
        if self.adjust != 0:
            decoder_output = decoder_output[:, :, self.adjust:-self.adjust, self.adjust:-self.adjust]
        if self.pixelcnn is not None:
            concat = torch.cat([decoder_output, sample], dim=1)
            reconstruction = self.pixelcnn(concat)
        else:
            reconstruction = decoder_output
        return reconstruction

    def kl_divergence(self, encoding_mu, encoding_logvar):
        return -0.5 * torch.sum(encoding_logvar - (encoding_logvar).exp() - encoding_mu.pow(2) + 1)

    def compute_kernel(self, x, y):
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)
        x = x.unsqueeze(1)  # (x_size, 1, dim)
        y = y.unsqueeze(0)  # (1, y_size, dim)
        tiled_x = x.expand(x_size, y_size, dim)
        tiled_y = y.expand(x_size, y_size, dim)
        kernel_input = (tiled_x - tiled_y).pow(2).mean(2) / float(dim)
        return torch.exp(-kernel_input)  # (x_size, y_size)

    def compute_mmd(self, x, y):
        x_kernel = self.compute_kernel(x, x)
        y_kernel = self.compute_kernel(y, y)
        xy_kernel = self.compute_kernel(x, y)
        mmd = x_kernel.sum() + y_kernel.sum() - 2 * xy_kernel.sum()
        return mmd

    def loss(self, target, encoding_mu, encoding_logvar, encoding, reconstruction, device, args):
        kl = torch.tensor(0.).to(device)
        mmd = torch.tensor(0.).to(device)

        # In case of requrie rsample to be False, encoding_logvar would be None
        if encoding_mu is not None and encoding_logvar is not None:
            kl = self.kl_divergence(encoding_mu, encoding_logvar)

        # In case of only pixelcnn architecture the encoding would be none
        if encoding is not None:
            true_samples = torch.randn(target.shape[0], encoding.shape[1]).to(device)
            mmd = self.compute_mmd(true_samples, encoding.view(-1, encoding.shape[1]))
        # Need cross entropy when pixelcnn exists or when it is plain vae, but number of output_channels from deocder is
        # greater than number of input_channels
        if self.pixelcnn is not None or (self.pixelcnn is None and self.decoder_out_channels > self.in_channels):
            px_given_z = self.nll * F.cross_entropy(reconstruction, target, reduction='none',
                                                    weight=args.data_ratio_of_labels).sum()
        else:
            px_given_z = - self.nll * Normal(reconstruction, self.sigma_decoder).log_prob(target).sum()

        loss = (px_given_z + (self.kl * kl) + (self.mmd * mmd)) / target.shape[0]
        return loss, px_given_z.item() / target.shape[0], kl.item() / target.shape[0], mmd.item() / target.shape[0]

    def __repr__(self):
        string = ""

        pixelcnn_input = str(self.input_image_size) + "x" + str(self.input_image_size) + "x"
        if self.only_pixelcnn:
            pixelcnn_used = "by itself"
            pixelcnn_input += str(self.in_channels)
        else:
            pixelcnn_used = "in the decoder"
            pixelcnn_input += str(self.in_channels + self.decoder_out_channels)
            rsample_text = ""
            if self.require_rsample:
                rsample_text = " Where Z is rsampled from a Normal Distribution."
            string += "We are using an encoder which takes input of " + str(self.input_image_size) + "x" + str(
                self.input_image_size) + "x" + str(self.in_channels) + " and encodes into " + str(self.z_dimensions) + \
                      " dimensional latent space." + rsample_text + \
                      " \nIt is then pushed into a decoder which outputs an image of dimension " + \
                      str(self.input_image_size) + "x" + str(self.input_image_size) + "x" + str(
                self.decoder_out_channels) + ".\n"

        if self.pixelcnn is None:
            if self.decoder_out_channels == self.in_channels:
                string += "We assume p(x/z) follows a normal distribution with mean x_recon and sigma " + str(
                    self.sigma_decoder) + ".\n"
            else:
                string += "We assume p(x/z) follows a categorical distribution. \n"
        else:
            string += "We are using PixelCNN " + pixelcnn_used + " which takes an input of " + pixelcnn_input + \
                      " dimension goes through " + str(self.num_pixelcnn_layers) + " layers and outputs a " \
                      + str(self.input_image_size) + "x" + str(self.input_image_size) + "x" + str(
                self.pixelcnn_out_channels) + " dimensions image"
        return string
