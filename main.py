import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os
import canny
from datetime import datetime
from threading import Thread


class Snap:
    number_of_snaps = 0

    def __init__(self, path):
        self.name = os.path.basename(path)
        self.path = {'raw': path}
        self.number = Snap.number_of_snaps
        Snap.number_of_snaps += 1
        self.particles = []


def procc(frame, background):
    subs = cv.imread(frame.path['crop'])[:, :, 0] - background
    cv.imwrite(frame.path['subs'], subs)
    edge = subs
    # Размытие по гауссу 4х4
    edge = cv.blur(edge, (4, 4))
    # Cany 100 200 границы
    edge = cv.Canny(edge, 100, 200)
    cv.imwrite(frame.path['edge'], edge)
    fill = edge
    # Открытие структурным элементом - крест
    fill = canny.fill_parts_n_remove_threads(fill, ellipse_size=3)
    cv.imwrite(frame.path['fill'], fill)
    proc = fill
    proc = canny.numerate_parts(proc)
    cv.imwrite(frame.path['proc'], proc)
    frame.particles = sorted(canny.count(proc), reverse=True)
    #print(str(round((frame.number + 1) / frame.number_of_snaps * 100, 2)) + r'%')


if __name__ == '__main__':
    # Считывание всех имен нужного формата в папке данных
    # Вход в папку с данными
    os.chdir('data1')
    directories = 'raw'
    file_format = '.bmp'
    # Считывание всех файлов нужного формата и запись пути в список
    frames = []
    for file_name in os.listdir(directories):
        if file_name[-len(file_format):] == file_format:
            frames.append(Snap(os.getcwd() + '\\' + directories + '\\' + file_name))
    # Перезапись всех папок
    directories = ('crop', 'subs', 'threshold', 'edge', 'fill', 'detected', 'proc')
    for directory in directories:
        if directory in os.listdir(os.getcwd()):
            for file in os.listdir(os.getcwd() + '\\' + directory):
                os.remove(os.getcwd() + '\\' + directory + '\\' + file)
                print(datetime.now().time(), directory, ' ', file, ' ', 'has been removed')
            os.rmdir(directory)
        os.mkdir(directory)
        for frame in frames:
            frame.path[directory] = os.getcwd() + '\\' + directory + '\\' + frame.name
    del directory, directories, frame, file_format, file_name

    # Обрезка и запись изображений
    # Поиск минимумов
    cr = cv.imread(frames[0].path['raw'])[:, :, 0]
    cv.imwrite(frames[0].path['crop'], cr)
    background = cr
    for frame in frames:
        cr = cv.imread(frame.path['raw'])[:, :, 0]
        cv.imwrite(frame.path['crop'], cr)
        background = np.minimum(cr, background)
    cv.imwrite(os.getcwd() + '\\' + 'background.bmp', background)
    print(datetime.now().time(), 'Background has been found')
    # Вычитание фона
    threads = []
    for frame in frames:
        threads.append(Thread(target=procc, args=(frame, background)))
    for t in threads:
        t.start()
    for i, t in enumerate(threads):
        t.join()
        print(str(i) + '/' + str(len(threads)))

    data = 'data.txt'
    os.remove(data)
    with open(data, 'a') as f:
        f.write('res' + '\t' + str(np.shape(background)[0]) + '\t' + str(np.shape(background)[1]) + '\n')
        f.write('frame' + '\t' + 'id' + '\t' + 'x' + '\t' + 'y' + '\t' + 'area' + '\n')
    for frame in frames:
        for id, particle in enumerate(frame.particles):
            with open(data, 'a') as f:
                f.write(str(frame.path['raw']) + '\t' + str(frame.number) + '\t' + str(id) + '\t' + str(round(particle.c[0], 0)) + '\t' +
                        str(round(particle.c[1], 0)) + '\t' + str(particle.area) + '\n')
