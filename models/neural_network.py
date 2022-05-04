'''
load and run neural network models (for eval purposes mainly)
'''
from skimage import io
from skimage import transform
import torch
import sys
sys.path.append('..')
sys.path.append('.')
from training.utils import train_saver

class run_net:
    def __init__(self, save_dir, torch_net, lr, lr_decay, batch_size, checkpoint='latest'):
        self.input_size = torch_net.input_size
        self.net = torch_net
        self.net.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = self.net.to(self.device)
        self.checkpoint = checkpoint
        self.loader = train_saver(save_dir, torch_net, lr, lr_decay, batch_size)
        self.load_checkpoint()

    def load_checkpoint(self):
        if self.checkpoint == 'latest':
            self.loader.load_pretrained(self.net)
        else:
            try:
                #TODO: load from earlier checkpoint than latest
                x=1
                # load from that checkpoiint
            except:
                raise Exception('Load type: '+str(self.checkpoint)+' failed')

    def preprocess_input(self, image_path):
        image = io.imread(image_path)
        image = transform.resize(image, (self.input_size, self.input_size))
        image = torch.from_numpy(image.transpose((2, 0, 1)))
        image = image[None, :].to(device=self.device, dtype=torch.float) #expand dimension to batch size of 1
        if image.max() > 1:
            image = image/255
        return image

    def postprocess_output(self, output):
        embeddings_array = output.cpu().detach().numpy()
        return list(embeddings_array[0, :])

    def get_embedding(self, image_path):
        '''
        get embedding of a single image
        '''
        image = self.preprocess_input(image_path)
        embedding = self.net(image)
        return self.postprocess_output(embedding)


if __name__ == '__main__':
    from models import FaceNet, toy_network, network
    # net = FaceNet.FaceNetInception()
    net = network.network()
    # net = toy_network.toy_network()
    # print(net)
    m = run_net('data/files_to_gitignore/models', net, 0.001, 0.98, 16)
    embedding = m.get_embedding('data/uob_image_set512/16288974/16288974_1.jpg')
    print(embedding)
