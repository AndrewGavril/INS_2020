import pandas as pd
import numpy as np
from math import ceil

dat = pd.read_csv('dataset\\train.csv', sep=',')

data = dat.values

data = [(data[i, 0], data[i, 1],
         list(map(int, data[i, 2].split()))) for i in range(0, data.shape[0])]

w, h = (1600, 256)

new_data = []
for j in range(0, len(data)):
    new_enc_pixels = []
    for i in range(0, len(data[j][2])):
        p = data[j][2][i]
        x = p // h
        y = p % h
        x1 = x // 4
        y1 = y // 4
        p1 = x1*h//4 + y1
        new_enc_pixels.append(p1)
    new_data.append((data[j][0], data[j][1],
                     ' '.join(map(str, new_enc_pixels))))
        

dat = pd.DataFrame(new_data, columns=dat.columns)
dat.to_csv('new_train.csv', index=False)
