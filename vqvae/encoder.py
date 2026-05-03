import torch.nn as nn


class EncoderBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1)
        self.norm  = nn.BatchNorm2d(output_dim)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(output_dim)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        return x


class EncoderModule(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.encoders = nn.Sequential(
            EncoderBlock(3,   128),
            EncoderBlock(128, 256),
            EncoderBlock(256, 256),
            EncoderBlock(256, 512),
        )
        self.final = nn.Conv2d(512, config["codebook_dim"], kernel_size=1)

    def forward(self, x):
        x = self.encoders(x)
        x = self.final(x)
        return x
