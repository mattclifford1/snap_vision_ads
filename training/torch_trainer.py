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


class trainer():
    def __init__(self, model, epochs=5, batch_size=16, save_dir='data/files_to_gitignore'):
        self.lr = 0.0001
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.save_dir = save_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cores = multiprocessing.cpu_count()
        # get data loader
        prefetch_factor = 2
        trans = transforms.Compose([Rescale((self.model.input_size+100, self.model.input_size+100)),
        RandomCrop(self.model.input_size)])
        transformed_dataset = get_data(transform=trans)
        self.dataloader = DataLoader(transformed_dataset,
                                     batch_size=self.batch_size,
                                     shuffle=True,
                                     num_workers=int(self.cores/2),
                                     prefetch_factor=prefetch_factor)


    def start(self):
        self.setup()
        self.evaluation = []
        for epoch in tqdm(range(self.epochs), desc="Epochs"):
            if epoch >= self.start_epoch:
                running_loss = []
                for step, sample in enumerate(tqdm(self.dataloader, desc="Train Steps", leave=False)):
                    loss = self.train_step(sample)
                    # log the loss
                    running_loss.append(loss.cpu().detach().numpy())

                # get validation stats and save the trained model
                if (epoch)%2 == 0:
                    save_model(self.save_dir, self.model, self.lr, epoch+1)
                    eval = eval_torch_model.run(self.model, batch_size=self.dataloader.batch_size)
                    acc = eval['in_any']*100
                    print('Training loss: '+str(np.mean(running_loss))+' Eval score: '+str(acc))
                    self.evaluation.append(acc)
                    self.model.train()   # put back into train mode
                else:
                    save_model(self.save_dir, self.model, self.lr, 'temp_save')
            self.scheduler.step() # lower optimiser learning rate
        save_model(self.save_dir, self.model, self.lr, epoch+1)
        print('training eval: ', self.evaluation)
        self.model.eval()
        return self.model

    def train_step(self, sample):
        # get training batch sample
        anchor_img = sample['image'].to(device=self.device, dtype=torch.float)
        positive_img = sample['positive'].to(device=self.device, dtype=torch.float)
        negative_img = sample['negative'].to(device=self.device, dtype=torch.float)

        # zero the parameter gradients
        self.optimiser.zero_grad()

        # forward
        anchor_out = self.model(anchor_img)
        positive_out = self.model(positive_img)
        negative_out = self.model(negative_img)
        # loss
        loss = self.criterion(anchor_out, positive_out, negative_out)
        # backward pass
        loss.backward()
        self.optimiser.step()
        return loss

    def setup(self):
        # optimser
        self.optimiser = optim.Adam(self.model.parameters(), self.lr)
        # criterion = TripletLoss()
        self.criterion = nn.TripletMarginLoss(margin=1.0, p=2)
        # load pretrained model if availbale
        self.start_epoch = load_pretrained(self.model, self.save_dir, self.lr)
        # set up model for training
        self.model = self.model.to(self.device)
        self.model.train()
        self.scheduler = ExponentialLR(self.optimiser, gamma=0.9)


if __name__ == '__main__':
    embedding_dims = 64
    # epochs = 100
    # input_size = 256
    # model = toy_network(input_size, embedding_dims)
    # model = run(model, epochs)
    # eval_torch_model.run(model)
