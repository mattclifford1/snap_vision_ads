import torch
import torch.nn as nn
from torchvision import transforms
import os
import errno


class network(nn.Module):
    def __init__(self, input_size=512, emb_dim=128):
        super(network, self).__init__()

        self.conv_size_1 = 16
        self.conv_size_2 = 32
        self.conv_size_3 = 64
        self.conv_size_4 = 128
        self.conv_size_5 = 256
        self.kernel_size = 10
        self.input_size = input_size
        self.final_conv_dims = 7*7*self.conv_size_5

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.conv_size_1, self.kernel_size),
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3))
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.conv_size_1, self.conv_size_2, self.kernel_size),
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3))
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.conv_size_2, self.conv_size_3, self.kernel_size),
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3))
        self.conv4 = nn.Sequential(
            nn.Conv2d(self.conv_size_3, self.conv_size_4, self.kernel_size),
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3))
        self.conv5 = nn.Sequential(
            nn.Conv2d(self.conv_size_4, self.conv_size_5, self.kernel_size),
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3))

        self.fc = nn.Sequential(
            nn.Linear(self.final_conv_dims, emb_dim*4),
            # nn.Linear(64*self.conv_final_dim*self.conv_final_dim, emb_dim*2),
            nn.PReLU(),
            nn.Linear(emb_dim*4, emb_dim)
        )

    def forward(self, x):
        x = self.check_input_size(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(-1, self.final_conv_dims)
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
