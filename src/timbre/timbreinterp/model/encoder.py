import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, encoder_configs):
        super().__init__()

        self.in_channel = encoder_configs["in_channel"]
        self.num_layers = encoder_configs["num_layers"]
        self.kernel_size = encoder_configs["kernel_size"]
        self.stride = encoder_configs["stride"]
        self.inflate_factor = encoder_configs["inflate_factor"]

        self.conv_layers, self.bns = self.init_layers()
        self.relu = nn.ReLU()

    def init_layers(self):

        conv_layers = nn.ModuleList()
        bns = nn.ModuleList()
        
        in_channel = self.in_channel
        out_channel = self.inflate_factor
        for i in range(self.num_layers):
            conv_layers.append(nn.Conv2d(in_channel, out_channel, self.kernel_size, self.stride, padding=1))
            bns.append(nn.BatchNorm2d(out_channel))
            in_channel = out_channel
            out_channel *= self.inflate_factor
        
        return conv_layers, bns
    
    def forward(self, x):
        out = x
        for i in range(self.num_layers):
            out = self.conv_layers[i](out)
            out = self.relu(self.bns[i](out))
        
        return out
