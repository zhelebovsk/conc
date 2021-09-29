import numpy as np
from matplotlib import pyplot as plt
import os
import csv

if __name__ == '__main__':
    os.chdir('data1')
    data = 'data.txt'
    tsv_file = open(data, 'r')
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    text = []
    res = []
    for id, row in enumerate(read_tsv):
        if id == 0:
            res.append(int(row[1]))
            res.append(int(row[2]))
        if id >= 2:
            text.append(row[2:-1])
            #print(row[2:-1])
    tsv_file.close()
    del id, read_tsv, row, tsv_file
    x = 160
    y = 80
    mat = np.zeros([x, y])
    dx = res[0] / x
    dy = res[1] / y
    for i in text:
        mat[int(float(i[0]) / dx), int(float(i[1]) / dy)] += 1
    plt.imshow(mat[50:112,:])
    plt.show()