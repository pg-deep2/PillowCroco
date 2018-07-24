import cv2
import numpy as np
import visdom, time
import torch
from torchvision import transforms
from PIL import Image

cap = cv2.VideoCapture(r"C:\Users\DongHoon\Documents\PROGRAPHY DATA_ver2\HV\HV00.mp4")
frames = []

# reading video using cv2
while True:
    ret, frame = cap.read()
    if ret:
        # 짜잔 r이랑 b랑 스왑
        b, g, r = cv2.split(frame)
        frame = cv2.merge([r, g, b])
        # HWC2CHW
        frame = frame.transpose(2, 0, 1)
        frames.append(frame)
    else:
        break
cap.release()

out = np.concatenate(frames)
print(out.shape, type(out))
out = out.reshape(-1, 3, 270, 480)
print(out.shape, type(out))

image_transforms = transforms.Compose([
    Image.fromarray,
    transforms.CenterCrop(260),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #씨발 totensor를 normalize보다 앞에 썻어야 시 발 시발 쉬발ㅇ러 ㅁ이ㅏㅓㅠㅠㅠㅠㅠㅠㅠㅠㅠㅠㅠㅠ

    # lambda x: x[:n_channels, ::],
])

vid = []
for im in out:
    vid.append(image_transforms(im.transpose(1, 2, 0)))

vid = torch.stack(vid)
viz = visdom.Visdom(use_incoming_socket=False)
for f in range(0, out.shape[0]):
    viz.image(vid[f, :, :, :], win="gt video", opts={'title': 'GT'})
    time.sleep(0.01)


# 아 R이랑 B랑 바뀌어서 나오네ㅋㅋㅋㅋ



class Trainer(object):
    def __init__(self, config, h_loader, r_loader):
        self.config = config
        self.h_loader = h_loader
        self.r_loader = r_loader

        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.weight_decay = config.weight_decay

        self.n_epochs = config.n_epochs
        self.n_steps = config.n_steps
        self.log_interval = int(config.log_interval)    #in case
        self.checkpoint_step = int(config.checkpoint_step)

        self.use_cuda = config.cuda
        self.outf = config.outf
        self.build_model()
        self.vis = vis_tool.Visualizer()

    def build_model(self):

        self.c2d = C2D().cuda()
        self.gru = GRU(self.c2d).cuda()
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
                      % (epoch + 1, self.n_epochs, step + 1, self.n_steps, step_end_time - start_time, h_loss))

                self.vis.plot('LOSS with lr:%.4f, b1:%.1f, b2:%.3f, wd:%.5f'
                              %(cfig.lr, cfig.beta1, cfig.beta2, cfig.weight_decay),
                              (h_loss.data).cpu().numpy())


            if epoch % self.checkpoint_step == 0:
                torch.save(self.gru.state_dict(), 'chkpoint' + str(epoch+1) + '.pth')
                print("checkpoint saved")



import torch.nn as nn
import numpy as np
import torch


class C2D(nn.Module):
    def __init__(self):
        super(C2D, self).__init__()
        # 48(batch_size)*3*240*400 -> 64*120*133
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), padding=(1, 1))
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 3), stride=(2, 3))

        # -> 128 40 43
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))

        # -> 256 20 21
        self.conv3a = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1))
        self.conv3b = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1))
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 3), stride=(2, 2))

        # -> 512 10 10
        self.conv4a = nn.Conv2d(256, 512, kernel_size=(3, 3), padding=(1, 1))
        self.conv4b = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1, 1))
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 3), stride=(2, 2))

        # -> 1024 5 5
        self.conv5a = nn.Conv2d(512, 1024, kernel_size=(3, 3), padding=(1, 1))
        self.conv5b = nn.Conv2d(1024, 1024, kernel_size=(3, 3), padding=(1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2), stride=(2, 2))

        # -> 2048 2 2
        self.conv6a = nn.Conv2d(1024, 2048, kernel_size=(3, 3), padding=(1, 1))
        self.conv6b = nn.Conv2d(2048, 2048, kernel_size=(3, 3), padding=(1, 1))
        self.pool6 = nn.MaxPool3d(kernel_size=(3, 3), stride=(2, 2))

class GRU(nn.Module):
    def __init__(self, c2d):
        super(GRU, self).__init__()

        # 48 2048 2 2
        # after flatten : (48) 8192
        self.c2d = c2d

        # 48 * 8192 -> 48 * 1
        self.gru = nn.GRUCell(8192, 75)
        self.fc = nn.Sequential(nn.Linear(48 * 75, 1000),
                                nn.Linear(1000, 100),
                                nn.Linear(100, 10),
                                nn.Linear(10, 1)
                                )
        self.sig = nn.Sigmoid()

    def forward(self, input):
        # input.shape = 1 x f x 3 x 240 x 400
        self.input = input
        print(input.shape)

        """
        inputback = input.cpu()
        inputback = torch.from_numpy(np.flip(inputback.numpy(), 1).copy())
        inputback = inputback.cuda()
        self.inputback = inputback
        """

        #scoring every frame and count how many each frame got scored
        f_score_list = np.zeros(input.shape[1])
        f_score_count = np.zeros(input.shape[1])

        #b_score_list = np.zeros(input.shape[1])

        step = 0
        start = 0
        end = 48
        while end < input.shape[1]:
            x = input[0, start:end, :, :, :]
            # x.shape: 48, 3, 240, 400

            h = self.c2d(x)
            # h.shape: 48 2048 2 2

            h = self.gru(h)
            h = self.fc(h)
            h = self.sig(h)
            # h.shape : 1

            #??? 어캐해야하징~~
            f_score_list[i] += h.item()
            f_score_count[i] += 1
            start += 6
            end += 6
            step += 1



        return total_loss
