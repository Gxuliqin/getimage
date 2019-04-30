import numpy as np
import cv2
import math
import os
import re



def rotate(image, angle, center=None, scale=1.0): #1
    (h, w) = np.shape[image] #2
    if center is None: #3
        center = (w // 2, h // 2) #4

    M = cv2.getRotationMatrix2D(center, angle, scale) #5

    rotated = cv2.warpAffine(image, M, (w, h)) #6
    return rotated #7



def creattxt():

    os.chdir('/home/liqin/python/tf/t_swich')
    # train0 = os.listdir('./pmc')
    train0 = os.listdir('./pmcg')
    train0_r0 = os.listdir('./pmc_d10')
    train0_r1 = os.listdir('./pmc_p10')
    train1 = os.listdir('./pocg')
    train1_r0 = os.listdir('./poc_d10')
    train1_r1 = os.listdir('./poc_p10')

    test0 = os.listdir('./testm')
    test1 = os.listdir('./testo')

    file0 = open('train.txt', 'w')
    file1 = open('labels.txt','w')
    file2 = open('test.txt','w')
    file3 = open('test_labels.txt','w')
    for i in  train0:
        file0.write(i)
        file0.write('\n')
        file1.write('1,0')
        file1.write('\n')
    for i in  train0_r0:
        file0.write(i)
        file0.write('\n')
        file1.write('1,0')
        file1.write('\n')
    for i in  train0_r1:
        file0.write(i)
        file0.write('\n')
        file1.write('1,0')
        file1.write('\n')

    for i in train1:
        file0.write(i)
        file0.write('\n')
        file1.write('0,1')
        file1.write('\n')
    for i in train1_r0:
        file0.write(i)
        file0.write('\n')
        file1.write('0,1')
        file1.write('\n')
    for i in train1_r1:
        file0.write(i)
        file0.write('\n')
        file1.write('0,1')
        file1.write('\n')

    file0.close()
    file1.close()

    for i in test0:
        file2.write(i)
        file2.write('\n')
        file3.write('1,0')
        file3.write('\n')

    for i in test1:
        file2.write(i)
        file2.write('\n')
        file3.write('0,1')
        file3.write('\n')


def readtxt(path):
    lis = []
    os.chdir('/home/liqin/python/tf/conv1')
    file0 = open(path, 'r')

    l = file0.readlines()

    file0.close()
    c = 0
    for i in range(len(l)):

        lk = l[i].split('\n')
        lis.append(lk[0])

    print len(l)


    return lis

def readlabels(path):
    lis = []
    os.chdir('/home/liqin/python/tf/conv1')
    file0 = open(path, 'r')

    l = file0.readlines()

    file0.close()

    for i in range(len(l)):
        lk = l[i].split('\n')
        ls= lk[0].split(',')
        lis.append(ls)
    print len(l)
    return lis

def getdict():
    dic ={}
    image = readtxt('train.txt')
    labels = readlabels('labels.txt')
    for i in range(len(image)):
        dic[image[i]] = labels[i]
    return dic

def getrotateimage():
    os.chdir('/home/liqin/python/tf/t_swich')
    r_path = './pmc/'
    path_dir = os.listdir(r_path)
    size = 80
    for image_name in path_dir:
        image_path = r_path+image_name
        im = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
        equ = cv2.equalizeHist(im)
        # (h, w) = np.shape(im)
        # if h < w:
        #     cim = cv2.resize(im, (w * 80 / h, 80))
        # else:
        #     cim = cv2.resize(im, (80, h * size / w))
        img = cv2.resize(equ, (80, 80))
        # for i in range(80):
        #     for j in range(80):
        #         img[i][j] = cim[i][j]

        # center = (np.shape(img)[0] / 2, np.shape(img)[1] / 2)
        # T = cv2.getRotationMatrix2D(center, -10, 0.9)
        # rotate = cv2.warpAffine(img, T, (size, size))
        path_save = './pmcg/'
        name_dif = 'rmg'
        if os.path.exists(path_save):
            cv2.imwrite(path_save+name_dif+image_name,img)
        else:
            os.mkdir(path_save)
            cv2.imwrite(path_save+name_dif+image_name, img)
def main():
    creattxt()
    # getrotateimage()



if __name__ =='__main__':
    main()





