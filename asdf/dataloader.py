import numpy as np
import cv2
import torch
import os
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import functools

"""
open cv library인 cv2를 이용하여 video를 불러온다.
highlight video는 최소 60f의 길이를 갖고, 12fps 480x270 RGB channel이다.
통째로 hDataset에 모두 들어간다.
한 번에 비디오 클립 하나만 들어간다(batch_size=1)
"""

# cap.read는 맨 앞의 frame을 ndarray로 출력해주고 자기 내에선 삭제함(pop up)
# 각 프레임은 270x480x3으로, h x w x d의 꼴임.
"""
cap = cv2.VideoCapture("C:\\Users\\DongHoon\\Documents\\PROGRAPHY DATA_ver2\\HV\\HV00.mp4")
frameset = []
# label = self.videos[item].split("\\")[-2]
while (True):
    ret, frame = cap.read()
    if (ret == 0):
        break
    frame = np.transpose(frame, (2, 0, 1))
    frameset.append(frame)
print(np.array(frameset).shape)
"""


# 이를 통해서 frameset의 shape는 frames x 3 x 270 x 480!


class hDataset(Dataset):
    def __init__(self, dataroot, transform=None):
        # dataroot는 str
        # os.listdir는 dir내에 하부 요소들을 list로 return
        # os.path.join(a,b)는 경로 두개를 합침. 사실 a는 큰경로, b는 파일이름 으로 보는게 좋음
        # os.path.splitext(a) 는 a의 확장자를 기준으로 두개로 나눔(pair를 return함)
        self.dataroot = dataroot
        videos = os.listdir(dataroot)
        self.videos = [os.path.join(self.dataroot, v) for v in videos if os.path.splitext(v)[-1] == ".mp4"]

        # transform의 값을 따르되, 정의 안되어있으면 f(x)=x라는 함수가 됨.
        self.transforms = transform if transform is not None else lambda x: x \

    def __getitem__(self, item):
        # item번째 video를 가져와서 프레임들의 묶음(fx3x270x480)을 뱉어라. 우선 label은 return 안하게 놔뒀음
        cap = cv2.VideoCapture(self.videos[item])
        frameset = []
        # label = self.videos[item].split("\\")[-2]
        while (True):
            ret, frame = cap.read()
            if (ret == 0):
                break
            frame = np.transpose(frame, (2, 0, 1))
            frameset.append(frame)
        return self.transforms(frameset),  # "HV"

    def __len__(self):
        return len(self.videos)


# 여기부턴 수현이꺼 그대로.
def video_transform(video, image_transform):
    # apply image transform to every frame in a video
    vid = []
    for im in video:
        #원래 channel x h x w 를 w x h x channel로 다시 바꾼다고? ??머지
        vid.append(image_transform(im.transpose(1, 2, 0)))

    vid = torch.stack(vid)
    # vid. 10, 3, 64, 64
    vid = vid.permute(1, 0, 2, 3)
    # vid. 3, 10, 64, 64
    return vid

#하이라이트 애들만
def get_loader(dataroot, batch_size=1):  # , image_size, n_channels, image_batch, video_batch, video_length):
    image_transforms = transforms.Compose([
        Image.fromarray,
        transforms.CenterCrop(270),
        transforms.ToTensor(),
        # lambda x: x[:n_channels, ::],
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # video_transforms = video_transform(video, image_transforms)라는 뜻.
    video_transforms = functools.partial(video_transform, image_transform=image_transforms)

    # 함수가 parameter로 들어가는거 돌아버리겠네 ㅋㅋㅋ
    h_dataset = hDataset(dataroot, video_transforms)
    # h_video = h_dataset[2][0]

    # viz = visdom.Visdom()
    # for f in range(0, h_video.shape[0]):
    #     viz.image(h_video[f,:,:,:], win="gt video", opts={'title':'GT'})
    #     time.sleep(0.01)

    h_loader = DataLoader(h_dataset, batch_size=batch_size, drop_last=True, shuffle=True)
    return h_loader


if __name__ == "__main__":
    get_loader(r'C:\Users\DongHoon\Documents\PROGRAPHY DATA_ver2\HV', 1)
