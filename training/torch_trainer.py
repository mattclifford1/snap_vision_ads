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
    def __init__(self, model, lr=0.001, lr_decay=0.95, epochs=5, batch_size=16, save_dir='data/files_to_gitignore'):
        # training settings
        self.lr_decay_epoch = 10
        self.save_epoch = 20
        self.eval_epoch = 2
        # save inputs to self
        self.lr = lr
        self.lr_decay = lr_decay
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.save_dir = save_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cores = multiprocessing.cpu_count()
        # get data loader
        self.get_data_loader(prefetch_factor=2)

    def get_data_loader(self, prefetch_factor=2):
        trans = transforms.Compose([Rescale((self.model.input_size+100, self.model.input_size+100)),
        RandomCrop(self.model.input_size)])
        transformed_dataset = get_data(transform=trans)
        self.dataloader = DataLoader(transformed_dataset,
                                     batch_size=self.batch_size,
                                     shuffle=True,
                                     num_workers=int(self.cores/2),
                                     prefetch_factor=prefetch_factor)

    def setup(self):
        # optimser
        self.optimiser = optim.Adam(self.model.parameters(), self.lr)
        # criterion = TripletLoss()
        self.criterion = nn.TripletMarginLoss(margin=1.0, p=2)
        # init the saver util
        self.saver = train_saver(self.save_dir, self.model, self.lr, self.lr_decay, self.batch_size)
        # load pretrained model if availbale
        self.start_epoch = self.saver.load_pretrained(self.model)
        # set up model for training
        self.model = self.model.to(self.device)
        self.model.train()
        self.scheduler = ExponentialLR(self.optimiser, gamma=self.lr_decay)
        self.running_loss = [0]

    def start(self):
        self.setup()
        self.eval(0)
        # training loop
        for epoch in tqdm(range(self.epochs), desc="Epochs"):
            if epoch >= self.start_epoch:
                self.running_loss = []
                # train for one epoch
                for step, sample in enumerate(tqdm(self.dataloader, desc="Train Steps", leave=False)):
                    self.train_step(sample)

                # maybe eval, save model and lower learning rate
                if epoch%self.eval_epoch == 0:
                    # get validation stats and log results
                    self.eval(epoch+1)
                if epoch%self.save_epoch == 0:
                    # save the trained model
                    self.saver.save_model(self.model, epoch+1)
            if epoch%self.lr_decay_epoch  == 0:
                # lower optimiser learning rate
                self.scheduler.step()

        # training finished
        self.saver.save_model(self.model, epoch+1)
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
        self.running_loss.append(loss.cpu().detach().numpy()) # save the loss stats

    def eval(self, epoch):
        eval = eval_torch_model.run(self.model, batch_size=self.dataloader.batch_size)
        eval_acc = eval['in_any']*100
        stats = {'epoch': [epoch],
                 'mean training loss': [np.mean(self.running_loss)],
                 'evaluation score': [eval_acc]}
        self.saver.log_training_stats(stats)
        # print('Training loss: '+str(np.mean(running_loss))+' Eval score: '+str(eval_acc))
        self.model.train()   # put back into train mode


if __name__ == '__main__':
    embedding_dims = 64
    # epochs = 100
    # input_size = 256
    # model = toy_network(input_size, embedding_dims)
    # model = run(model, epochs)
    # eval_torch_model.run(model)
