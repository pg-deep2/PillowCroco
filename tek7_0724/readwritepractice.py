import numpy as np
import random

data = np.zeros(50)
hvS = random.randint(0, 27)
hvF = random.randint(9, 22)

for i in range(hvS, hvF + hvS):
    data[i] = 1

np.save(r"C:\Users\DongHoon\Documents\PROGRAPHY DATA_ver2\tmp", np.array(data))
a=np.load(r"C:\Users\DongHoon\Documents\PROGRAPHY DATA_ver2\tmp.npy")

print(a)


print(data.nonzero())

data2=np.zeros(50)
print(data2.nonzero())
