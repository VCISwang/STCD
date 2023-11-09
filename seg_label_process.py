import random
import os
import cv2


from PIL import Image
import blobfile as bf
import numpy as np
import glob
from torch.utils.data import DataLoader, Dataset
from os.path import join, basename, splitext, exists
from tqdm import tqdm


select_cdfiles = False
if select_cdfiles:
    data_dir = '/home/chrisd/change/STCD/data/WHU-AB/train'
    # save_dir = '/home/chrisd/change/RePaint-main/data/datasets/WHU/'

    files = sorted(glob.glob(data_dir + "/A_label/*.*"))
    # img_files = sorted(glob.glob(data_dir + "/image/*.*"))

    change_files = open('changed.txt', mode='w')

    num = 0
    better_label = 0
    for i in range(len(files)):
        label_path = files[i]
        img_name = label_path.split('/')[-1]

        # img_path = img_files[i]
        label = cv2.imread(label_path)

        label = label / 255
        if label.sum() < 98304:
            better_label = better_label + 1
            print('select better_label: ' + img_name + '  num: ' + str(better_label))
            change_files.write(img_name + '\n')

    change_files.close()


creat_list = True
if creat_list:

    data_dir = '/home/chrisd/change/STCD/data/WHU-AB/test/'
    all_files = open(data_dir + 'list/val.txt', mode='w')

    for path in tqdm(glob.glob(join(data_dir, 'A', '*.tif'))):
        name, ext = splitext(basename(path))
        all_files.write(str(name + ext) + '\n')  # list error

    all_files.close()
