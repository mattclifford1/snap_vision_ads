import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

import sys
sys.path.append('.')
sys.path.append('..')
from data_loader.load import get_training_data
from data_loader.augmentation import *
from evaluation.compare_similar import eval

import multiprocessing
cores = multiprocessing.cpu_count()
prefetch_factor = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_dims = 16
batch_size = 16
epochs = 2

input_size = 512


# # test out speed of data loaders
# for i in tqdm(range(len(transformed_dataset))):
#     # sample = transformed_dataset[i]
#     if i == 100:
#         break
# for i, sample in tqdm(enumerate(dataloader)):
#     if i == 1000:
#         break


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()

class Network(nn.Module):
    def __init__(self, input_size=30, emb_dim=128):
        self.conv_size_1 = 32
        self.conv_size_2 = 64
        self.kernel_size = 5
        super(Network, self).__init__()
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
        x = self.conv(x)
        x = x.view(-1, 64*self.conv_final_dim*self.conv_final_dim)
        x = self.fc(x)
        # x = nn.functional.normalize(x)
        return x


# input_size = 32
# net = Network(input_size)
# print(net)
# input = torch.randn(1, 3, input_size, input_size)
# out = net(input)
# print(out)

# def init_weights(m):
#     if isinstance(m, nn.Conv2d):
#         torch.nn.init.kaiming_normal_(m.weight)
#
#
#
#
if __name__ == '__main__':

    # get data loader
    trans = transforms.Compose([Rescale((input_size+100, input_size+100)),
                                RandomCrop(input_size),
                                ToTensor()])
    transformed_dataset = get_training_data(transform=trans)

    dataloader = DataLoader(transformed_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=int(cores/2),
                            prefetch_factor=prefetch_factor)
    model = Network(input_size, embedding_dims)
    # model.apply(init_weights)
    model = torch.jit.script(model).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # criterion = torch.jit.script(TripletLoss())
    criterion = TripletLoss()


    #train
    model.train()
    for epoch in tqdm(range(epochs), desc="Epochs"):
        running_loss = []
        for step, sample in enumerate(tqdm(dataloader, desc="Training", leave=False)):
            anchor_img = sample['image'].to(device=device, dtype=torch.float)
            positive_img = sample['positive'].to(device=device, dtype=torch.float)
            negative_img = sample['negative'].to(device=device, dtype=torch.float)

            # print(anchor_img.shape, positive_img.shape, negative_img.shape)

            optimizer.zero_grad()
            anchor_out = model(anchor_img)
            positive_out = model(positive_img)
            negative_out = model(positive_img)

            loss = criterion(anchor_out, positive_out, negative_out)
            loss.backward()
            optimizer.step()

            running_loss.append(loss.cpu().detach().numpy())
            if step%10 == 0:
                print("Epoch: {}/{} - Loss: {:.4f}".format(epoch+1, epochs, np.mean(running_loss)))
                break
        torch.save(model.state_dict(), 'data/files_to_gitignore/trained_simple_model'+str(epoch)+'.pth')
