import cv2
import numpy as np
import main


class Particle:
    def __init__(self, area, x_c, y_c, limits):
        self.area = area
        self.c = [x_c, y_c]
        self.limits = limits
        self.slice = [slice(limits[0][0], limits[1][0] + 1), slice(limits[0][1], limits[1][1] + 1)]

    def __cmp__(self, other):
        return self.area - other.area

    def __lt__(self, other):
        return self.area < other.area

    def __str__(self):
        return str(self.area)


def fill_parts_n_remove_threads(img_in, ellipse_size=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ellipse_size, ellipse_size))
    img_proc = np.copy(img_in)
    res = np.shape(img_in)
    # заливка по всем углам изображения
    if img_proc[0, 0] == 0:
        mask = np.zeros((res[0] + 2, res[1] + 2), np.uint8)
        cv2.floodFill(img_proc, mask, (0, 0), 255)
    if img_proc[-1, -1] == 0:
        mask = np.zeros((res[0] + 2, res[1] + 2), np.uint8)
        cv2.floodFill(img_proc, mask, (res[1] - 1, res[0] - 1), 255)
    if img_proc[0, -1] == 0:
        mask = np.zeros((res[0] + 2, res[1] + 2), np.uint8)
        cv2.floodFill(img_proc, mask, (res[1] - 1, 0), 255)
    if img_proc[-1, 0] == 0:
        mask = np.zeros((res[0] + 2, res[1] + 2), np.uint8)
        cv2.floodFill(img_proc, mask, (0, res[0] - 1), 255)
    img_proc = cv2.bitwise_not(img_proc)
    img_proc = cv2.bitwise_or(img_proc, img_in)
    img_proc = cv2.morphologyEx(img_proc, cv2.MORPH_OPEN, kernel)
    return img_proc


def flood(img_in, img_out, x, y, val):
    img_in[x, y] = False
    img_out[x, y] = val
    if x > 0:
        if img_in[x - 1, y]:
            flood(img_in, img_out, x-1, y, val)
    if x < np.shape(img_in)[0] - 1:
        if img_in[x + 1, y]:
            flood(img_in, img_out, x + 1, y, val)
    if y > 0:
        if img_in[x, y - 1]:
            flood(img_in, img_out, x, y - 1, val)
    if y < np.shape(img_in)[1] - 1:
        if img_in[x, y + 1]:
            flood(img_in, img_out, x, y + 1, val)


def numerate_parts(img_in):
    img_proc = np.copy(img_in)
    img_out = np.zeros_like(img_proc)
    img_proc = img_proc > 0
    k = 0
    a = np.argwhere(img_proc)
    for i in a:
        if img_proc[i[0], i[1]]:
            k += 1
            flood(img_proc, img_out, i[0], i[1], k)
    return img_out


def count(img_in):
    particles = []
    for i in range(np.max(img_in)):
        q = np.argwhere(img_in == i + 1)
        xmin = np.min(q[:, 0])
        ymin = np.min(q[:, 1])
        xmax = np.max(q[:, 0])
        ymax = np.max(q[:, 1])
        p = Particle(np.shape(q)[0],
                     np.mean(q[:, 0]),
                     np.mean(q[:, 1]),
                     ((xmin, ymin), (xmax, ymax)))
        particles.append(p)
    return particles


if __name__ == '__main__':

    def nothing(a):
        pass

    # считывание изображения
    img_original = cv2.imread('image1.bmp')[:, :, 0]
    res = img_original.shape[:2]

    # создание окна и трекбаров
    window = 'adj'
    cv2.namedWindow(window)
    cv2.createTrackbar('Gauss', window, 3, 10, nothing)
    cv2.createTrackbar('threshold1', window, 50, 400, nothing)
    cv2.createTrackbar('threshold2', window, 150, 400, nothing)

    while True:
        # снятие значений с баров
        gauss = cv2.getTrackbarPos('Gauss', window)
        threshold1 = cv2.getTrackbarPos('threshold1', window)
        #cv2.setTrackbarPos('threshold2', window, threshold1+100)
        threshold2 = cv2.getTrackbarPos('threshold2', window)

        # размытие изображение для удаления шумов
        if gauss > 0:
            img_blur = cv2.blur(img_original, (gauss, gauss))
        else:
            img_blur = img_original
        # преобразование Кэнни
        img_edges = cv2.Canny(img_blur, threshold1, threshold2)
        # заполение пустот
        img_fill = fill_parts_n_remove_threads(img_edges, 3)

        A = numerate_parts(img_fill)
        # вывод результата
        #F, c, pos = count(A)
        img_result = np.vstack((cv2.addWeighted(img_blur, 0.5, img_fill, 0.8, 0), img_edges, img_fill))
        cv2.imshow(window, cv2.resize(img_result, (int(0.8*res[1]), int(0.8*3*res[0]))))
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyWindow(window)
            break
