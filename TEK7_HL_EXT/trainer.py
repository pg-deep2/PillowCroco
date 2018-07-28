import time
import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import vis_tool
from config import get_config
import os
from cnn_extractor import CNN, GRU


class Trainer(object):
    def __init__(self, config, h_loader, r_loader, t_loader):
        self.config = config
        self.h_loader = h_loader
        self.r_loader = r_loader
        self.t_loader = t_loader

        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.weight_decay = config.weight_decay

        self.n_epochs = config.n_epochs
        self.n_steps = config.n_steps
        self.log_interval = int(config.log_interval)  # in case
        self.checkpoint_step = int(config.checkpoint_step)

        self.use_cuda = config.cuda
        self.outf = config.outf
        self.build_model()
        self.vis = vis_tool.Visualizer()

    def build_model(self):
        self.c2d = CNN().cuda()
        self.c2d.load_state_dict(torch.load('cnn.pkl'))  # load pre-trained cnn extractor
        for l, p in self.c2d.named_parameters():
            p.requires_grad = False

        self.gru = GRU(self.c2d).cuda()
        self.gru.load_state_dict(torch.load('./samples/lr_0.0010_chkpoint1.pth'))  # load pre-trained cnn extractor

    def train(self):
        # create optimizers
        cfig = get_config()
        opt = optim.Adam(filter(lambda p: p.requires_grad,self.gru.parameters()),
                               lr=self.lr, betas=(self.beta1, self.beta2),
                               weight_decay=self.weight_decay)

        start_time = time.time()

        max_acc = 0.

        for epoch in range(self.n_epochs):
            epoch_loss = []
            for step, h in enumerate(self.h_loader):
                # test모드 지나고 다시 train모드
                self.gru.train()

                # if step == 3: break
                h_video = h

                # self.vis.img("h",h_video)
                # self.vis.img("r", r_video)

                # highlight video
                h_video = Variable(h_video).cuda()

                self.gru.zero_grad()

                h_loss = self.gru(h_video)
                h_loss.backward()
                opt.step()

                step_end_time = time.time()

                epoch_loss.append((h_loss.data).cpu().numpy())

                print('[%d/%d][%d/%d] - time: %.2f, h_loss: %.3f'
                      % (epoch + 1, self.n_epochs, step + 1, self.n_steps, step_end_time - start_time,
                         h_loss
                         ))

                self.vis.plot('H_LOSS with lr:%.4f, b1:%.1f, b2:%.3f, wd:%.5f'
                              % (cfig.lr, cfig.beta1, cfig.beta2, cfig.weight_decay),
                              (h_loss.data).cpu().numpy())

            self.vis.plot("Avg loss plot with lr:%.4f, b1:%.1f, b2:%.3f, wd:%.5f"
                          % (cfig.lr, cfig.beta1, cfig.beta2, cfig.weight_decay),
                          np.mean(epoch_loss))

            if epoch % self.checkpoint_step == 0:
                accuracy, savelist = self.test(self.t_loader)

                if accuracy > max_acc:
                    max_acc = accuracy
                    torch.save(self.gru.state_dict(), './samples/lr_%.4f_chkpoint' % cfig.lr + str(epoch + 1) + '.pth')
                    for f in savelist:
                        np.save("./samples/" + f[0] + ".npy", f[1])
                    print("checkpoint saved")

    def test(self, t_loader):
        # test mode
        self.gru.eval()
        accuracy = 0.

        savelist = []

        total_len = len(t_loader)
        # test 데이터 개수

        for step, (tv, label, filename) in enumerate(t_loader):
            filename = filename[0].split(".")[0]

            label = label.squeeze()
            # 진짜 highlight로 표시된 프레임들

            start = 0
            end = 30
            # 30프레임 단위로 찢어본다.

            correct = []
            ext_hv_frames = np.zeros(tv.shape[1])
            # [0,0,0,0,0,00,.... 프레임수만큼.,,,0]

            while end < tv.shape[1]:  # 갈수있는만큼 30프레임만큼 가라.

                t_video = Variable(tv[:, start:end, :, :, :]).cuda()
                loss = self.gru(t_video)
                # loss값. scalar.

                gt_label = label[start:end]
                # start~end까지 ground truth 갖고옴.

                if len(gt_label[gt_label == 1.]) > 24:
                    gt_label = torch.ones(1, dtype=torch.float32).cuda()

                else:
                    gt_label = torch.zeros(1, dtype=torch.float32).cuda()
                    # 30프레임단위 24프레임이상 gt가 하이라이트면 그 단위도 하이라이트, 아니면 아님

                if loss < 0.1:
                    ext_hv_frames[start:end] = 1.
                    # loss가 0.1보다 작으면 내가 보는 30단위는 다 1로 추출할거임.

                loss[loss < 0.1] = 1.
                loss[loss >= 0.1] = 0.
                correct.append((loss == gt_label).item())
                # 24프레임 내에서의 정확도를 correct 추가함

                start += 6
                end += 6

            accuracy += sum(correct) / len(correct) / total_len

            savelist.append([filename, ext_hv_frames])

        print("Accuracy:", round(accuracy, 4))
        self.vis.plot("Accuracy with lr:%.3f" % self.lr, accuracy)

        return accuracy, savelist
