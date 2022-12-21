#coding=utf-8
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')


def MDF_image(ts0, ts1, overlap, MDF, D):
    shape = MDF.shape
    for i in range(shape[0]):
        right_bound = len(ts0) - (D - 1) * (i + 1)
        for j in range(right_bound):
            #if i == 0:
            motif_idx = np.arange(j, j + D * (i + 1), (i + 1))
            M_d0 = np.array([ts0[motif_idx[0]], ts0[motif_idx[1]], ts0[motif_idx[2]]])
            M_d1 = np.array([ts1[motif_idx[0]], ts1[motif_idx[1]], ts1[motif_idx[2]]])
            M_d0_mean = np.mean(M_d0)
            H = M_d0 - M_d0_mean
            M_d1_mean = np.mean(M_d1)
            I = M_d1 - M_d1_mean
            r = (np.sum(H * I))//((np.linalg.norm(H, ord=2, axis=None, keepdims=False)) * (np.linalg.norm(I, ord=2, axis=None, keepdims=False)))
            if j < right_bound - overlap:
                MDF[i, j] = r
                MDF[shape[0]-i-1, shape[1]-j-1] = r
            else:
                MDF[i, j] = r
    return MDF

file = open("fwest0.txt")
lines = file.readlines()
rows = len(lines)
data0 = np.zeros((rows, 1))
row = 0
for line in lines:
    line = line.strip().split('\t')
    data0[row, :] = line[:]
    row += 1


file = open("east0.txt")
lines = file.readlines()
rows = len(lines)
data1 = np.zeros((rows, 1))
row = 0
for line in lines:
    line = line.strip().split('\t')
    data1[row, :] = line[:]
    row += 1


#print(data)

ts0 = data0
ts1 = data1
N = len(ts0)
D = 3
C = range(1, (N - 1)//(D - 1) + 1) #d
shape = np.array([len(range(1, (N - 1)//(D - 1) + 1)), len(range(0, N - (D - 1)*1))])
overlap = N - (D - 1)*shape[0]
MDF = np.zeros(shape)
print(shape[0])


shape1 = MDF.shape
MDF = MDF_image(ts0, ts1, overlap, MDF, D)
print(MDF)
print(MDF.shape)
#np.savetxt(r'test.txt',MDF, fmt='%d', delimiter=' ')

plt.matshow(MDF, norm = matplotlib.colors.Normalize(vmin=None, vmax=None, clip=False), cmap='coolwarm')
plt.axis('off')
images_path = "E:\\program\\lalala\\1"
plt.savefig(images_path + '/' + 'fwest0-east0' + '.png')
