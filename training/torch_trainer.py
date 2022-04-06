import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import ExponentialLR

from tqdm import tqdm
import multiprocessing

import sys
sys.path.append('.')
sys.path.append('..')
from data_loader.load import get_data
from data_loader.augmentation import *
from evaluation import eval_torch_model
from models.toy_network import toy_network


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


def _train(model, optimiser, criterion, dataloader, device, epochs):
    model = model.to(device)
    model.train()
    evaluation = []
    scheduler = ExponentialLR(optimiser, gamma=0.9)
    for epoch in tqdm(range(epochs), desc="Epochs"):

        running_loss = []
        for step, sample in enumerate(tqdm(dataloader, desc="Steps", leave=False)):
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
        if (epoch)%2 == 0:
            print("Epoch: {}/{} - Loss: {:.4f}".format(epoch+1, epochs, np.mean(running_loss)))
            torch.save(model.state_dict(), 'data/files_to_gitignore/trained_'+model.__class__.__name__+'_epoch_'+str(epoch)+'.pth')
            acc = eval_torch_model.run(model)['in_any']*100
            print(acc)
            evaluation.append(acc)
            model.train()   # put back into train mode
        else:
            torch.save(model.state_dict(), 'data/files_to_gitignore/trained_'+model.__class__.__name__+'temp_save.pth')
        scheduler.step() # lower optimiser learning rate
    print('training eval: ', evaluation)
    return model


def run(model, input_size, epochs=5, batch_size=32):
    cores = multiprocessing.cpu_count()
    prefetch_factor = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get data loader
    trans = transforms.Compose([Rescale((input_size+100, input_size+100)),
    RandomCrop(input_size)])
    transformed_dataset = get_data(transform=trans)

    dataloader = DataLoader(transformed_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=int(cores/2),
    prefetch_factor=prefetch_factor)

    optimiser = optim.Adam(model.parameters(), lr=0.001)
    criterion = TripletLoss()

    # train
    model = _train(model, optimiser, criterion, dataloader, device, epochs)
    # eval
    model.eval()
    return model



if __name__ == '__main__':
    embedding_dims = 64
    epochs = 100
    input_size = 256
    model = toy_network(input_size, embedding_dims)
    model = run(model, epochs)
    eval_torch_model.run(model)
