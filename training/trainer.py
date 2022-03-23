import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import multiprocessing

import sys
sys.path.append('.')
sys.path.append('..')
from data_loader.load import get_data
from data_loader.augmentation import *
from evaluation.compare_similar import eval_torch_model
from models.toy_network import Network


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


def run(model, optimiser, criterion, dataloader, device):
    model = model.to(device)
    model.train()
    evaluation = []
    for epoch in tqdm(range(epochs), desc="Epochs"):
        running_loss = []
        for step, sample in enumerate(tqdm(dataloader, desc="Training", leave=False)):
            anchor_img = sample['image'].to(device=device, dtype=torch.float)
            positive_img = sample['positive'].to(device=device, dtype=torch.float)
            negative_img = sample['negative'].to(device=device, dtype=torch.float)

            optimiser.zero_grad()
            anchor_out = model(anchor_img)
            positive_out = model(positive_img)
            negative_out = model(negative_img)

            loss = criterion(anchor_out, positive_out, negative_out)
            loss.backward()
            optimiser.step()

            running_loss.append(loss.cpu().detach().numpy())
            # if step%10 == 0:
            #     print("Step Loss: {:.4f}".format(loss.cpu().detach().numpy()))
        if epoch%5 == 0:
            print("Epoch: {}/{} - Loss: {:.4f}".format(epoch+1, epochs, np.mean(running_loss)))
            torch.save(model.state_dict(), 'data/files_to_gitignore/trained_simple_model'+str(epoch)+'.pth')
            evalled = eval_torch_model(model, input_size)
            evaluation.append(evalled.accuracy*100)
    print(evaluation)
    return model


if __name__ == '__main__':
    cores = multiprocessing.cpu_count()
    prefetch_factor = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_dims = 64
    batch_size = 32
    epochs = 100
    input_size = 256

    # get data loader
    trans = transforms.Compose([Rescale((input_size+100, input_size+100)),
                                RandomCrop(input_size),
                                ToTensor()])
    transformed_dataset = get_data(transform=trans)

    dataloader = DataLoader(transformed_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=int(cores/2),
                            prefetch_factor=prefetch_factor)

    model = Network(input_size, embedding_dims)
    optimiser = optim.Adam(model.parameters(), lr=0.001)
    criterion = TripletLoss()

    # train
    model = run(model, optimiser, criterion, dataloader, device)
    # eval
    model.eval()
    eval_torch_model(model, input_size)
