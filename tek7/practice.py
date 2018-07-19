import cv2
import numpy as np

cap = cv2.VideoCapture("C:\\Users\DongHoon\Documents\PROGRAPHY DATA_ver2\HV\HV00.mp4")
print(cap.isOpened())
frames = []
while True:
    ret, frame = cap.read()
    if ret:
        # HWC2CHW
        frame = frame.transpose(2, 0, 1)
        frames.append(frame)
        # print(frame)
    else:
        break
print(type(frames))
out = np.concatenate(frames)
out = out.reshape(-1, 3, 270, 480)
print(type(out))
print(out == frames)