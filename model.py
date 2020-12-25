import torch.nn as nn


""" Optional conv block """
def conv_block(in_channels, out_channels):

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )


""" Define your own model """
class FewShotModel(nn.Module):
    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()

        # embedding layers
        self.f = nn.Sequential(
            conv_block(x_dim, hid_dim),
            nn.MaxPool2d(2),
            conv_block(hid_dim, hid_dim),
            nn.MaxPool2d(2),
            conv_block(hid_dim, hid_dim),
            nn.MaxPool2d(2),
            conv_block(hid_dim, z_dim),
            nn.MaxPool2d(2)
        )
            
    def forward(self, x):
        embedding_vector = self.f(x).view([x.shape[0],-1])
        return embedding_vector


class AlexNet(nn.Module):

    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

    def forward(self, x):
        x = self.features(x).view([x.shape[0],-1])
        return x
