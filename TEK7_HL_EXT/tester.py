import cv2
import numpy as np
import visdom
import time
import os
import torch
from config import get_config
import torch.backends.cudnn as cudnn
from dataloader import get_loader


class TestViewer():
    """
    test_video : test video 하나의 filename (각 파일명 맨 뒤에 ground true hv의 frame이 적혀있음)
    extracted_hv : test_video 랑 같은 제목, 다른 확장자(npy)를 가지는 filename. numpy array를 가지고 있으며 각 snippet(48fs)마다 0, 1값이 표시됨.
    예상되는 애들은 00000011111111111000뭐 이런식인데[얘는 구현함] 0000011100111111100111이렇게 되는 경우도 생각해보자!!
    """

    def __init__(self, test_video, extracted_hv, step):

        self.test_video = test_video
        self.extracted_hv = extracted_hv
        self.step = step

        # test video를 frame별로 불러와서 numpy array로 test_raw에 저장함.
        cap = cv2.VideoCapture(self.test_video)
        frames = []
        while True:
            ret, frame = cap.read()
            if ret:
                b, g, r = cv2.split(frame)
                frame = cv2.merge([r, g, b])
                # HWC2CHW
                frame = frame.transpose(2, 0, 1)
                frames.append(frame)
            else:
                break
        cap.release()

        test_raw = np.concatenate(frames)
        self.test_raw = test_raw.reshape(-1, 3, 270, 480)

    def show(self, item=-1):
        if item == -1:
            self.showrv()
            self.showthv()
            self.showehv()
        elif item == 0:
            self.showrv()
        elif item == 1:
            self.showthv()
        elif item == 2:
            self.showehv()
        else:
            pass

    def showrv(self):

        viz0 = visdom.Visdom(use_incoming_socket=False)

        for f in range(0, self.test_raw.shape[0]):
            viz0.image(self.test_raw[f, :, :, :], win="gt video", opts={'title': 'TEST_RAW' + str(step)}, )
            time.sleep(0.0081)

    def showthv(self):
        viz1 = visdom.Visdom(use_incoming_socket=False)
        # 이 과정은 test_true_hv를 보여주기 위해 test_raw에서 hv frame을 index함,
        filename = os.path.split(self.test_video)[-1]
        print(self.test_video)

        h_start = filename.index("(")
        h_end = filename.index(")")

        h_frames = filename[h_start + 1: h_end]
        # h_frames = "42, 120" or "nohv"

        if "," in h_frames:
            s, e = h_frames.split(',')
            h_start, h_end = int(s), int(e)
        else:
            h_start, h_end = 0, 0

        if (h_start == h_end):
            print("Ground Truth : No HV")
        for f in range(h_start, h_end):
            viz1.image(self.test_raw[f, :, :, :], win="gt1 video", opts={'title': 'TEST_TRUE_HV' + str(step)}, )
            time.sleep(0.0081)

    def showehv(self):
        viz2 = visdom.Visdom(use_incoming_socket=False)
        # 이 과정은 test_extracted_hv를 보여주기 위해 test_raw에서 hv frame을 index함.
        # 0000001111111111111111110000000000000000011111111100000 이런식이면 가장 긴 1애들만 보여줌.
        ext = np.load(self.extracted_hv)
        idx = []
        # 1이 처음으로 나오는 ext의 idx들 list

        count = []
        # idx부터 몇개의 1이 연속해서 나오느냐 count

        for i in range(len(ext)):
            if (i == 0 and ext[i] == 1):
                idx.append(i)
                count.append(1)
            elif (ext[i - 1] == 0 and ext[i] == 1):
                idx.append(i)
                count.append(1)
            elif (ext[i - 1] * ext[i] == 1):
                count[-1] += 1
            else:
                pass

        count = np.array(count)

        print("idx: ", idx, "count: ", count)

        if idx == []:
            e_start, e_end = 0, 0
        else:
            max_val = np.argmax(count)
            if (type(max_val) == np.ndarray):
                max_val = np.argmax(count)[0]
            e_start = idx[max_val]
            e_end = e_start + count[max_val]

        if (e_start == e_end):
            print("Extracted Value : No HV")
        for f in range(e_start, e_end):
            viz2.image(self.test_raw[f, :, :, :], win="gt2 video", opts={'title': 'TEST_Extracted_HV' + str(step)}, )
            time.sleep(0.0081)


if __name__ == "__main__":
    config = get_config()

    dataroot = config.dataroot
    h_datapath = os.path.join(dataroot, "HV")
    r_datapath = os.path.join(dataroot, "RV")
    t_datapath = os.path.join(dataroot, 'testRV')
    _, _, t_loader = get_loader(h_datapath, r_datapath, t_datapath)
    for step, (tv, label, filename) in enumerate(t_loader):
        f = os.path.join(t_datapath, filename[0])
        npy = f.replace(".mp4", ".npy")
        test = TestViewer(f, npy, step)
        test.show(1)  # show(0)show(1)show(2) 다 됨
        test.show(2)

    # cap = cv2.VideoCapture(test_video)
    # frames = []
    # while True:
    #     ret, frame = cap.read()
    #     if ret:
    #         b, g, r = cv2.split(frame)
    #         frame = cv2.merge([r, g, b])
    #         # HWC2CHW
    #         frame = frame.transpose(2, 0, 1)
    #         frames.append(frame)
    #     else:
    #         break
    # cap.release()

    # size = int(len(frames) / 6) - 7
    # a = np.zeros(size)
    # print(len(a))
    # for i in range(40, 45):
    #     a[i] = 1
    #
    # print(a)
    #
    # np.save(r"C:\Users\KIM\Documents\PROGRAPHY DATA_ver3\testRV\testRV00(42,120)", a)
