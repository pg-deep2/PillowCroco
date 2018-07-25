import torch.nn as nn
import numpy as np
import torch
from dataloader import get_loader


class GRU(nn.Module):
    def __init__(self):
        super(GRU, self).__init__()

        # 3 240 400
        # after flatten : 12 512
        self.c2d = nn.Sequential(nn.Conv2d(3, 64, kernel_size=(3, 4), stride=(3, 4)),  # 64 80 100
                                 nn.LeakyReLU(0.2),
                                 nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),  # 128 80 100
                                 nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),  # 128 40 50
                                 nn.LeakyReLU(0.2),

                                 nn.Conv2d(128, 256, kernel_size=(2, 3), stride=(2, 3), padding=(0, 1)),  # 256 20 17
                                 nn.LeakyReLU(0.2),
                                 nn.Conv2d(256, 512, kernel_size=(2, 2), stride=(2, 2), padding=(0, 1)),  # 512 10 9
                                 nn.LeakyReLU(0.2),
                                 nn.Conv2d(512, 512, kernel_size=(2, 2), stride=(2, 2), padding=(0, 1)),  # 512 5 5
                                 nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(1, 1)),  # 512 3 3
                                 nn.LeakyReLU(0.2),

                                 nn.Conv2d(512, 1024, kernel_size=(2, 2), stride=(1, 1)),  # 512 2 2
                                 nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),  # 512 1 1
                                 nn.LeakyReLU(0.2)
                                 )

        self.gru = nn.GRU(1024, 1)

        # self.fc = nn.Sequential(nn.Linear(24 * 75, 600),
        #                         nn.Linear(600, 100),
        #                         nn.Linear(100, 10),
        #                         nn.Linear(10, 1)
        #                         )
        self.sig = nn.Sigmoid()

    def forward(self, input):
        # input.shape = 1 x f x 3 x 240 x 400
        self.input = input
        # print(input.shape)

        """
        inputback = input.cpu()
        inputback = torch.from_numpy(np.flip(inputback.numpy(), 1).copy())
        inputback = inputback.cuda()
        self.inputback = inputback
        """

        # scoring every frame and count how many each frame got scored
        f_score_list = np.zeros(input.shape[1])
        # f_score_count = np.zeros(input.shape[1])

        # b_score_list = np.zeros(input.shape[1])

        step = 0
        start = 0
        end = 24
        while end < input.shape[1]:
            x = input[0, start:end, :, :, :]
            # print(x.shape)
            x = x.squeeze()

            h = self.c2d(x)
            # print(h.shape)
            h = h.squeeze()
            h = h.view(-1, 24, 1024)
            # print("h", h.shape)
            # h.shape: 48 2048 2 2

            h, _ = self.gru(h)
            # h = self.fc(h)
            # print("h", h.shape, h)
            h = self.sig(h)
            h = h.squeeze()
            # print("h", h.shape, h)

            # h.shape : 1

            # ??? 어캐해야하징~~
            f_score_list[start:end] += h.data
            # print(f_score_list)

            start += 6
            end += 6
            step += 1

        f_score_list[6:12] /= 2
        f_score_list[12:18] /= 3
        f_score_list[18:start] /= 4
        f_score_list[start:start+6] /= 3
        f_score_list[start+12:start+18] /= 2

        f_score_list = f_score_list[:end - 6]
        return torch.from_numpy(f_score_list).cuda()


if __name__ == "__main__":
    gru = GRU().cuda()
    h_l, r_l, t_l = get_loader('/home/ubuntu/PillowCroco/PROGRAPHY DATA_ver3/HV',
                               '/home/ubuntu/PillowCroco/PROGRAPHY DATA_ver3/RV',
                               '/home/ubuntu/PillowCroco/PROGRAPHY DATA_ver3/testRV', 1)

    for idx, video in enumerate(h_l):
        # print(video)
        video = video[0].cuda()
        break
    print(video.shape)
    out = gru(video)
