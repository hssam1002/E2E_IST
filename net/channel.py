import torch.nn as nn
import numpy as np
import os
import torch
import time


class Channel(nn.Module):
    """
    Currently the channel model is either error free, erasure channel,
    rayleigh channel or the AWGN channel.
    """
    def __init__(self, args, config):
        super(Channel, self).__init__()
        self.config = config
        self.chan_type = args.channel_type
        self.device = config.device

    def gaussian_noise_layer(self, input_layer, std):
        device = input_layer.get_device()
        noise_real = torch.normal(mean=0.0, std = std, size=input_layer.shape, device=device)
        noise_imag = torch.normal(mean=0.0, std = std, size=input_layer.shape, device=device)
        noise = noise_real + 1j * noise_imag
        return input_layer + noise
    
    def rayleigh_noise_layer(self, input_layer, std):
        device = input_layer.device
        noise_real = torch.normal(mean=0.0, std=std, size=np.shape(input_layer))
        noise_imag = torch.normal(mean=0.0, std=std, size=np.shape(input_layer))
        noise = noise_real + 1j * noise_imag

        h = torch.sqrt(torch.normal(mean=0.0, std=1, size=input_layer.shape, device = device) ** 2
                       + torch.normal(mean=0.0, std=1, size=input_layer.shape, device=device) ** 2) / np.sqrt(2)
        return input_layer * h + noise

    def complex_normalize(self, x, power):
        pwr = torch.mean(x ** 2) * 2
        out = np.sqrt(power) * x / torch.sqrt(pwr)
        return out, pwr

    def forward(self, input, chan_param, avg_pwr=False):
        channel_tx = input
        input_shape = channel_tx.shape
        channel_in = channel_tx.reshape(-1)
        L = channel_in.shape[0]

        channel_in = channel_in[:L // 2] + channel_in[L // 2:] * 1j

        # Channel Effect
        if self.chan_type == 'awgn':
            sigma = np.sqrt(1.0 / (2 * 10 ** (chan_param / 10)))
            channel_output = self.gaussian_noise_layer(channel_in, std=sigma)
            
        elif self.chan_type == 'rayleigh':
            sigma = np.sqrt(1.0 / (2 * 10 ** (chan_param / 10)))
            channel_output = self.rayleigh_noise_layer(channel_in, std=sigma)
        else: # none
            channel_output = channel_in

        # Real Valued reconversion
        channel_output = torch.cat([torch.real(channel_output), torch.imag(channel_output)])
        channel_output = channel_output.reshape(input_shape)
        
        return channel_output

