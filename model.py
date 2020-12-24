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
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )
            
    def forward(self, x):
        shot, query= x  # (nway*kshot, 3, h, w)
        # embeddings
        e_shot = self.f(shot)
        e_query = self.f(query)

        # prototype 
        

        # metric



        # distribution
        pred

        return pred

def classifier(x):
    
    pred = torch.argmax(embedding_vector)
    return pred
