import cv2
import numpy as np
import visdom, time

cap = cv2.VideoCapture(r"C:\Users\KIM\Documents\PROGRAPHY DATA_ver2\HV\HV00.mp4")
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
        # print(frame)
    else:
        break
cap.release()

out = np.concatenate(frames)
out = out.reshape(-1, 3, 270, 480)

viz = visdom.Visdom()
for f in range(0, out.shape[0]):
    viz.image(out[f, :, :, :], win="gt video", opts={'title': 'GT'})
    time.sleep(0.01)

# 아 R이랑 B랑 바뀌어서 나오네ㅋㅋㅋㅋ
