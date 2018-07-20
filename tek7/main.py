"""
Usage: main.py [options] --dataroot <dataroot> --cuda
"""

import os

import random
import torch
import torch.backends.cudnn as cudnn

from config import get_config
from trainer import Trainer

from dataloader import get_loader

def main(config):
    if config.outf is None:
        config.outf = 'samples'
    os.system('mkdir {0}'.format(config.outf))

    config.manual_seed = random.randint(1, 10000)
    print("Random Seed: ", config.manual_seed)
    random.seed(config.manual_seed)
    torch.manual_seed(config.manual_seed)

    if config.cuda:
        torch.cuda.manual_seed_all(config.manual_seed)

    cudnn.benchmark = True

    dataroot = config.dataroot
    h_datapath = os.path.join(dataroot,"HV")
    t_datapath = os.path.join(dataroot,'testRV')

    # dataroot, cache, image_size, n_channels, image_batch, video_batch, video_length):
    h_loader, test_loader = get_loader(h_datapath, t_datapath, 1)

    trainer = Trainer(config, h_loader)
    trainer.train()

if __name__ == "__main__":
    config = get_config()
    main(config)


"""
[1/10][1/30] - time: 29.02, h_loss: 6418545.000
[1/10][2/30] - time: 46.79, h_loss: 6370203.000
[1/10][3/30] - time: 64.69, h_loss: 7697052.500
[1/10][4/30] - time: 77.65, h_loss: 10917174.000
-.-?
"""