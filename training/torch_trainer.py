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
from training.utils import *

import warnings
warnings.filterwarnings("ignore", category=UserWarning)   # porch warning we dont care about


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


def _train(model, lr, dataloader, device, epochs, save_dir):
    # optimser
    optimiser = optim.Adam(model.parameters(), lr)
    # criterion = TripletLoss()
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    # load pretrained model if availbale
    start_epoch = load_pretrained(model, save_dir, lr)
    # set up model for training
    model = model.to(device)
    model.train()
    evaluation = []
    scheduler = ExponentialLR(optimiser, gamma=0.9)
    for epoch in tqdm(range(epochs), desc="Epochs"):
        if epoch >= start_epoch:
            running_loss = []
            for step, sample in enumerate(tqdm(dataloader, desc="Steps", leave=False)):
                # get training batch sample
                anchor_img = sample['image'].to(device=device, dtype=torch.float)
                positive_img = sample['positive'].to(device=device, dtype=torch.float)
                negative_img = sample['negative'].to(device=device, dtype=torch.float)

                # zero the parameter gradients
                optimiser.zero_grad()

                # forward
                anchor_out = model(anchor_img)
                positive_out = model(positive_img)
                negative_out = model(negative_img)
                # loss
                loss = criterion(anchor_out, positive_out, negative_out)
                # backward
                loss.backward()
                optimiser.step()

                # log the loss
                running_loss.append(loss.cpu().detach().numpy())
                # if step%10 == 0:
                #     print("Step Loss: {:.4f}".format(loss.cpu().detach().numpy()))
            # get validation stats and save the trained model
            if (epoch)%2 == 0:
                print("Epoch: {}/{} - Loss: {:.4f}".format(epoch+1, epochs, np.mean(running_loss)))
                save_model(save_dir, model, lr, epoch+1)
                acc = eval_torch_model.run(model, batch_size=dataloader.batch_size)['in_any']*100
                print(acc)
                evaluation.append(acc)
                model.train()   # put back into train mode
            else:
                save_model(save_dir, model, lr, 'temp_save')
        scheduler.step() # lower optimiser learning rate
    save_model(save_dir, model, lr, epoch+1)
    print('training eval: ', evaluation)
    return model


def run(model, input_size, epochs=5, batch_size=32, save_dir='data/files_to_gitignore'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cores = multiprocessing.cpu_count()

    # get data loader
    prefetch_factor = 2
    trans = transforms.Compose([Rescale((input_size+100, input_size+100)),
                                RandomCrop(input_size)])
    transformed_dataset = get_data(transform=trans)
    dataloader = DataLoader(transformed_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=int(cores/2),
                            prefetch_factor=prefetch_factor)

    # train
    lr = 0.0001
    model = _train(model, lr, dataloader, device, epochs, save_dir)
    model.eval()
    return model


if __name__ == '__main__':
    embedding_dims = 64
    epochs = 100
    input_size = 256
    model = toy_network(input_size, embedding_dims)
    model = run(model, epochs)
    eval_torch_model.run(model)
