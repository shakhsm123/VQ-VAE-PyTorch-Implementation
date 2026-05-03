import torch.nn as nn


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.upsample   = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv       = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.batchnorm  = nn.BatchNorm2d(out_channels)
        self.relu       = nn.ReLU()
        self.conv2      = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)
        self.relu2      = nn.ReLU()

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu2(x)
        return x


class Decoder(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.conv_project = nn.Conv2d(config["codebook_dim"], 512, kernel_size=1)
        self.decoders = nn.Sequential(
            DecoderBlock(512, 256),
            DecoderBlock(256, 256),
            DecoderBlock(256, 128),
            DecoderBlock(128, 128),
        )
        self.final_conv = nn.Conv2d(128, 3, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv_project(x)
        x = self.decoders(x)
        x = self.final_conv(x)
        x = self.tanh(x)
        return x
