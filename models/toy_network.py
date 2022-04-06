import torch
import torch.nn as nn
from torchvision import transforms
import os
import errno


class toy_network(nn.Module):
    def __init__(self, input_size=30, emb_dim=128):
        self.conv_size_1 = 32
        self.conv_size_2 = 64
        self.kernel_size = 5
        self.input_size = input_size
        super(toy_network, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, self.conv_size_1, self.kernel_size),
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3),
            nn.Conv2d(self.conv_size_1, self.conv_size_2, self.kernel_size),
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3)
        )
        self.conv_final_dim = int(int(input_size/2)/2) - 3
        self.fc = nn.Sequential(
            nn.Linear(64*self.conv_final_dim*self.conv_final_dim, 512),
            nn.PReLU(),
            nn.Linear(512, emb_dim)
        )

    def forward(self, x):
        x = self.check_input_size(x)
        x = self.conv(x)
        x = x.view(-1, 64*self.conv_final_dim*self.conv_final_dim)
        x = self.fc(x)
        # x = nn.functional.normalize(x)
        return x

    def check_input_size(self, x):
        if isinstance(self.input_size, int):
            H = self.input_size
            W = self.input_size
        else:
            H = self.input_size[0]
            W = self.input_size[1]
        if x.shape[2] != H and x.shape[3] != W:
            x = transforms.resuze(x, (H, W))
        return x

def load_weights(model, weights_path):
    if not os.path.isfile(weights_path):
        # raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), weights_path)
        raise ValueError("Couldn't find network weights path: "+str(weights_path)+"\nMaybe you need to train first?")
    model.load_state_dict(torch.load(weights_path))
