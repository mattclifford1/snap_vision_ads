import sys
sys.path.append('.')
sys.path.append('..')
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from data_loader.load import get_database, get_data
from wrangling.database_creator import contruct_database
from torchvision import transforms
from data_loader.augmentation import *
from training.utils import *

input_size = 256
batch_size = 64

def run_loop(cores, prefetch_factor=1):
    print('core: '+str(cores))
    print('prefetch: '+str(prefetch_factor))
    data = get_database()
    trans = transforms.Compose([Rescale((input_size+100, input_size+100)),RandomCrop(input_size)])
    transformed_dataset = get_data(transform=trans)
    dataloader = DataLoader(transformed_dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=int(cores),
                                 prefetch_factor=prefetch_factor)
    for step, sample in enumerate(tqdm(dataloader, desc="Steps", leave=True)):
        anchor_img = sample['image']

run_loop(1)
run_loop(12)

run_loop(1, 2)
run_loop(12, 2)
