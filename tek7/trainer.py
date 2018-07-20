import time
import torch
from torch import nn
from torch.autograd import Variable
import torch.optim as optim
import vis_tool
from config import get_config

from model.C3D import C3D, GRU


class Trainer(object):
    def __init__(self, config, h_loader):
        self.config = config
        self.h_loader = h_loader

        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.weight_decay = config.weight_decay

        self.n_epochs = config.n_epochs
        self.log_interval = int(config.log_interval)
        self.checkpoint_step = int(config.checkpoint_step)

        self.use_cuda = config.cuda
        self.outf = config.outf
        self.build_model()
        self.vis = vis_tool.Visualizer()

    def load_model(self):
        self.p3d.load_state_dict(torch.load(self.config.pretrained_path))

        # FC layer removal & fixing pretrained layers' parameter
        fc_removed = list(self.p3d.children())[:-6]

        _p3d_net = []
        relu = nn.ReLU()

        for layer in fc_removed:
            for param in layer.parameters():
                param.requires_grad = False
            if layer.__class__.__name__ == 'MaxPool3d':
                _p3d_net.extend([layer, relu])
            else:
                _p3d_net.append(layer)

        p3d_net = nn.Sequential(*_p3d_net).cuda()

        self.p3d = p3d_net

    def build_model(self):
        self.p3d = C3D().cuda()
        self.load_model()

        self.gru = GRU(self.p3d).cuda()
        print("MODEL:")
        print(self.gru)

    def train(self):
        # create optimizers
        cfig = get_config()
        opt_model = optim.Adam(filter(lambda p: p.requires_grad, self.gru.parameters()),
                               lr=self.lr, betas=(self.beta1, self.beta2),
                               weight_decay=self.weight_decay)

        start_time = time.time()

        self.gru.train()

        for epoch in range(self.n_epochs):
            for step, h in enumerate(self.h_loader):
                h_video = h

                # highlight video
                h_video = Variable(h_video.cuda())

                self.gru.zero_grad()

                h_loss = Variable(self.gru(h_video).cuda(), requires_grad=True)

                h_loss.backward()
                opt_model.step()

                step_end_time = time.time()

                print('[%d/%d][%d/%d] - time: %.2f, h_loss: %.3f'
                      % (epoch + 1, self.n_epochs, step + 1, len(self.h_loader), step_end_time - start_time, h_loss))

                self.vis.plot('LOSS with lr:%.4f, b1:%.1f, b2:%.3f, wd:%.5f'
                              %(cfig.lr, cfig.beta1, cfig.beta2, cfig.weight_decay),
                              (h_loss.data).cpu().numpy())


            if epoch % self.checkpoint_step == 0:
                torch.save(self.gru.state_dict(), 'chkpoint' + str(epoch+1) + '.pth')
                print("checkpoint saved")
