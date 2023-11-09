#!/usr/bin/env bash
from PIL import Image
import random, math
import numpy as np
import os
import sys
from glob import glob
from itertools import count
from os import makedirs
from os.path import join, basename, splitext, exists

from skimage.io import imread, imsave
from tqdm import tqdm
import cv2


CROP_SIZE = 256
STRIDE = 256


if __name__ == '__main__':
    data_dir = '/home/chrisd/change/STCD/data/WHU-AB/'
    out_dir = '/home/chrisd/change/STCD/data/WHU-AB/'

    # data_dir = sys.argv[1]
    # out_dir = sys.argv[2]

    split = False
    if split:
        # for subset in ('train'):
        subset = 'image_data'
        for subdir in ('A', 'A_label', 'B', 'B_label', 'C_label'):
            for path in tqdm(glob(join(data_dir, subset, subdir, '*.tif'))):
                name, ext = splitext(basename(path))
                # img = imread(path)
                img = cv2.imread(path)
                h, w = img.shape[:2]
                counter = count()
                out_subdir = join(out_dir, subdir)
                if not exists(out_subdir):
                    makedirs(out_subdir)
                stride = STRIDE
                for i in range(0, h-CROP_SIZE+1, stride):
                    for j in range(0, w-CROP_SIZE+1, stride):
                        print(i, j)
                        imsave(join(out_subdir, '{}{}'.format(next(counter), ext)), img[i:i+CROP_SIZE, j:j+CROP_SIZE])


    select = False
    if select:
        # for subset in ('train'):
        list_image = []
        for i in range(7434):
            list_image.append(i)
        random.shuffle(list_image)
        for subdir in ('A', 'A_label', 'B', 'B_label', 'C_label'):
            img_list = []
            for path in tqdm(glob(join(data_dir, subdir, '*.tif'))):
                name, ext = splitext(basename(path))
                img_list.append(name + ext)   # list error

            if subdir == 'A': img_data = 'A'
            if subdir == 'B': img_data = 'B'
            if subdir == 'A_label': img_data = 'A_label'
            if subdir == 'B_label': img_data = 'B_label'
            if subdir == 'C_label': img_data = 'C_label'
            for i in range(img_list.__len__()):
                img = cv2.imread(join(data_dir, subdir, img_list[list_image[i]]))

                if i < 5948:
                    out_img_dir = join(out_dir, 'train', img_data, img_list[list_image[i]])
                elif 5947 < i < 6691:
                    out_img_dir = join(out_dir, 'val', img_data, img_list[list_image[i]])
                else:
                    out_img_dir = join(out_dir, 'test', img_data, img_list[list_image[i]])

                imsave(out_img_dir, img)



    # file1 = open('/home/chrisd/change/CrossCD/data/LEVIR/train_512/list/5_train_supervised.txt', 'w')
    # file2 = open('/home/chrisd/change/CrossCD/data/LEVIR/train_512/list/5_train_unsupervised.txt', 'w')
    # img_list = []
    # for i in range(445):  # 445  64  128
    #     for j in range(9):
    #         img_name = 'test_{}_{}.png'.format(str(i+1), str(j))
    #         img_list.append(img_name)

    # random.shuffle(img_list)
    # for i in range(7120):
    #     if i < 356:
    #         file1.write(img_list[i]+'\n')
    #     else:
    #         file2.write(img_list[i]+'\n')
    # select_23 = []
    # for i in range(445):
    #     select_23.append(i+1)
    # random.shuffle(select_23)
    # for i in range(445):
    #     if i < 23:
    #         for j in range(9):
    #             file1.write('train_{}_{}.png'.format(str(select_23[i]), str(j)) + '\n')
    #     else:
    #         for j in range(9):
    #             file2.write('train_{}_{}.png'.format(str(select_23[i]), str(j)) + '\n')


    # for i in range(64):  # 445  64  128
    #     img_name = 'val_{}.png'.format(str(i+1))
    #     img_list.append(img_name)
    # for i in range(64):  # 7120  1024  2048
    #     file1.write(img_list[i] + '\n')