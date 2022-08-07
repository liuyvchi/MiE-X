import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import cv2 as cv
import dlib
import time
import re
from PIL import Image
import cv2

num = 0

def Simulate_prepare():
    data_dir = 'EmotioNet/generated_flo_5w/'
    file_path = 'simulate_multi_labels_5w.csv'

    emotion_dict = {'Anger': 0, 'Fear': 0, 'Sadness': 0, 'Happiness':1, 'Contempt': 0, 'Surprise': 2, 'Disgust': 0, '0': 0, '1': 1, '2': 2}

    flo_name = 'flow.png'

    train_image_file = '../MER/Loading_file/5w_flo/train_image.txt'
    train_label_file = '../MER/Loading_file/5w_flo/train_label.txt'
    vali_image_file = '../MER/Loading_file/5w_flo/vali_image.txt'
    vali_label_file = '../MER/Loading_file/5w_flo/vali_label.txt'
    test_image_file = '../MER/Loading_file/5w_flo/test_image.txt'
    test_label_file = '../MER/Loading_file/5w_flo/test_label.txt'

    data_file = pd.read_csv(file_path, sep=' ')
    for index in range(len(data_file)):
        folder_name = data_file.iloc[index, 0]
        emotion = str(data_file.iloc[index, 1])
        emotion = emotion_dict[emotion]

        flo_data_path = os.path.join(data_dir, folder_name, 'flow.png')


        if index < 120000:
            with open (train_image_file, 'a') as d:
                d.write(flo_data_path + '\n')

            with open (train_label_file, 'a') as l:
                l.write(str(emotion) + '\n')
        else:
            with open(vali_image_file, 'a') as d:
                d.write(flo_data_path + '\n')

            with open(vali_label_file, 'a') as l:
                l.write(str(emotion) + '\n')

    micro_flo_prepare = '/home/user/Yuchi/iccv2019/MER/Loading_file/micro_flo_3fold'

    micro_flo = ['Micro_fold0_data0.txt', 'Micro_fold0_data1.txt', 'Micro_fold0_data2.txt']
    micro_labels = ['Micro_fold0_label0.txt', 'Micro_fold0_label1.txt', 'Micro_fold0_label2.txt']

    for i in range(3):
        with open (os.path.join(micro_flo_prepare, micro_flo[i]), 'r') as l:
            data = l.readlines()
        with open (os.path.join(micro_flo_prepare, micro_labels[i]), 'r') as l:
            label = l.readlines()
        assert (len(data) == len(label))

        for i in range(len(data)):
            with open(test_image_file, 'a') as d:
                d.write(data[i])

            with open(test_label_file, 'a') as l:
                l.write(label[i])




if __name__ == '__main__':
    start = time.time()
    Simulate_prepare()
    print('cost: ', time.time() - start)