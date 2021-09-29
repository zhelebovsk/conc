import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import pylab


class Snap:
    number_of_snaps = 0

    def __init__(self, path):
        self.name = os.path.basename(path)
        self.path = {'raw': path}
        self.number = Snap.number_of_snaps
        Snap.number_of_snaps += 1
        self.res = cv2.imread(path)[:, :, 0].shape


if __name__ == '__main__':
    post = np.zeros([1280,720,3],dtype=np.uint8)
    image = cv2.imread('1.jpg')
    for q in range(3):
        S = Snap('1.jpg')
        I = cv2.imread(S.name)[:, :, q]
        fourier = np.fft.fft2(I)
        fourier_shift = np.fft.fftshift(fourier)
        fourier_shift_show = np.log10(np.abs(fourier_shift))
        fourier_shift_show = fourier_shift_show / np.max(fourier_shift_show) * 255.0
        fourier_shift_show = np.array(fourier_shift_show, dtype=np.uint8)

        mask = np.ones([S.res[0], S.res[1]], dtype=np.uint8)
        #mask[100:220, 800:1120] = 0
        for i in range(S.res[0]):
            for j in range(S.res[1]):
                if np.hypot((i-S.res[0]/2)/S.res[0], (j-S.res[1]/2)/S.res[1])>0.00001:
                    mask[i,j] = 0
        plt.imshow(mask,)
        plt.show()
        #fourier_shift_real = np.log10(np.abs(fourier_shift))
        #fourier_real = np.log10(fourier_real)
        #fourier_real = fourier_real/np.max(fourier_real)*255.0
        #fourier_real = np.array(fourier_real, dtype=np.uint8)
        #F[F < 0.0000001] = 0
        fourier_shift = fourier_shift * (1-mask)
        fourier_inverse = np.abs(np.fft.ifft2(np.fft.ifftshift(fourier_shift)))
        fourier_inverse = np.array(fourier_inverse)
        post[:,:,q] = 255 - fourier_inverse.astype(np.uint8)


    #result = np.hstack([I, post])
    cv2.imshow('mat', post)
    cv2.waitKey(0)
    cv2.destroyWindow('mat')