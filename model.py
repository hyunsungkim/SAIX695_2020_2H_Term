import torch.nn as nn
import torch


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


class RNModel(nn.Module):
    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()

        # embedding layers
        self.f = nn.Sequential(
            conv_block(x_dim, hid_dim),
            nn.MaxPool2d(2),
            conv_block(hid_dim, hid_dim),
            nn.MaxPool2d(2),
            conv_block(hid_dim, hid_dim),
        )

        self.g = nn.Sequential(
            conv_block(x_dim, hid_dim),
            nn.MaxPool2d(2),
            conv_block(hid_dim, hid_dim),
            nn.MaxPool2d(2),
            conv_block(hid_dim, z_dim),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveMaxPool2d(z_dim),
            nn.ReLU(),
            nn.Linear(64,8),
            nn.ReLU(),
            nn.Linear(8,1),
            nn.Sigmoid()
        )

    def forward(self, s, q, args):
        print(f"s.shape, q.shape {s.shape}, {q.shape}")

        x = torch.cat([s,q], dim=0)
        x = self.f(x)

        print(f"x.shape {x.shape}")

        n_way = args.nway # num of class
        n_shot = int(s.shape[0]//n_way) # num of support image for each class
        n_query = int(q.shape[0]//n_way) # num of query image for each class

        q = x[s.shape[0]:]
        s = x[:s.shape[0]]

        proto_shots = torch.mean(s[:n_shot], dim=1)
        for i in range(args.nways):
            shots = s[i*n_shot:(i+1)*n_shot]
            proto_shots = torch.cat([proto_shots, torch.mean(shots, dim=1)])

        n, m = proto_shots.shape[0], q.shape[0]

        proto_shots = proto_shots.unsqueeze(1).expand(n, m, -1)
        q = q.unsqueeze(0).expand(n, m, -1)

        proto_shots = proto_shots.expand(n, m,)
        print(proto_shots.shape, q.shape)
        proto_shots = torch.mean(proto_shots, dim=1)
        print(proto_shots.shape, q.shape)
        c = 0

        re = self.g(c)
        similarity = self.classifier(re)

        return x