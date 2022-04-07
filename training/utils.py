import os
import torch


def get_save_dir(base_dir, model, lr):
    model_name = model.__class__.__name__
    path = os.path.join(base_dir, model_name+'_LR_'+str(lr))
    return path

def load_pretrained(model, base_dir, lr):
    dir = get_save_dir(base_dir, model, lr)
    if not os.path.isdir(dir):
        os.mkdir(dir)
        return 0
    checkpoints = os.listdir(dir)
    saves = []
    for checkpoint in checkpoints:
        name, ext = os.path.splitext(checkpoint)
        try:
            epoch = int(name)
            saves.append(epoch)
        except:
            pass
    if len(saves) > 0:
        latest_epoch = max(saves)
        weights_path = os.path.join(dir, str(latest_epoch)+'.pth')
        model.load_state_dict(torch.load(weights_path))
        print('Loaded pretrained model at epoch: '+str(latest_epoch))
        return latest_epoch
    else:
        return 0 #no pretrained found

def save_model(base_dir, model, lr, epoch):
    dir = get_save_dir(base_dir, model, lr)
    torch.save(model.state_dict(), os.path.join(dir, str(epoch)+'.pth'))