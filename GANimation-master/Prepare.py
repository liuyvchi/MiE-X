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
    ck_data_dir = 'EmotioNet/generated_multi_faces_ckAU_1030/'
    ck_file_path = 'simulate_multi_labels_CKAU_1030.csv'
    prior_data_dir = 'EmotioNet/generated_multi_faces_5t_3050/'
    prior_file_path = 'simulate_multi_labels_5t_3050.csv'
    realAU_data_dir = 'EmotioNet/generated_multi_faces_realAU5w_f0/'
    realAU_file_path = 'simulate_multi_labels_realAU5w_f0.csv'


    # emotion_dict = {'Anger': 0, 'Fear': 0, 'Sadness': 0, 'Happiness':1, 'Contempt': 0, 'Surprise': 2, 'Disgust': 0, '0': 0, '1': 1, '2': 2}
    emotion_dict = {'Anger': 0, 'Fear': 1, 'Sadness': 2, 'Happiness':3, 'Contempt': 4, 'Surprise': 5, 'Disgust': 6}

    start_name = '0_out.png'
    micro_name = '1_out.png'
    macro_name = '2_out.png'

    train_image_label_file = '../MER/Loading_file_new/5w_prior/simulate_prior_train_3050.txt'
    vali_image_label_file = '../MER/Loading_file_new/5w_prior/simulate_prior_vali_3050.txt'

    # vali_image_file = '../MER/Loading_file/5w_realAU/vali_image.txt'
    # vali_label_file = '../MER/Loading_file/5w_realAU/vali_label.txt'

    # data_file = pd.read_csv(ck_file_path, sep=' ')
    # for index in range(len(data_file)):
    #     folder_name = data_file.iloc[index, 0]
    #     emotion = str(data_file.iloc[index, 1])
    #     emotion = emotion_dict[emotion]
    #
    #     folder_path = os.path.join(ck_data_dir, folder_name)
    #
    #     # start_path = os.path.join(folder_path, start_name)
    #     # micro_path = os.path.join(folder_path, micro_name)
    #     # macro_path = os.path.join(folder_path, macro_name)
    #     if index < 150000:
    #         if index <120000:
    #             with open(train_image_label_file, 'a') as i:
    #                 i.write(folder_path + ' ' + str(emotion) + '\n')
    #         else:
    #             with open(vali_image_label_file, 'a') as i:
    #                 i.write(folder_path + ' ' + str(emotion) + '\n')

 #  =============================================================

    data_file = pd.read_csv(prior_file_path, sep=' ')
    for index in range(len(data_file)):
        folder_name = data_file.iloc[index, 0]
        emotion = str(data_file.iloc[index, 1])
        emotion = emotion_dict[emotion]

        folder_path = os.path.join(prior_data_dir, folder_name)

        # start_path = os.path.join(folder_path, start_name)
        # micro_path = os.path.join(folder_path, micro_name)
        # macro_path = os.path.join(folder_path, macro_name)

        if index < 90000:
            if index < 75000:
                with open(train_image_label_file, 'a') as i:
                    i.write(folder_path + ' ' + str(emotion) + '\n')
            else:
                with open(vali_image_label_file, 'a') as i:
                    i.write(folder_path + ' ' + str(emotion) + '\n')
   #
   #  data_file = pd.read_csv(realAU_file_path, sep=' ')
   #  for index in range(len(data_file)):
   #      folder_name = data_file.iloc[index, 0]
   #      emotion = str(data_file.iloc[index, 1])
   #      emotion = emotion_dict[emotion]
   #
   #      folder_path = os.path.join(realAU_data_dir, folder_name)
   #
   #      # start_path = os.path.join(folder_path, start_name)
   #      # micro_path = os.path.join(folder_path, micro_name)
   #      # macro_path = os.path.join(folder_path, macro_name)
   #
   #      if index < 120000:
   #          with open(train_image_label_file, 'a') as i:
   #              i.write(folder_path + ' ' + str(emotion) + '\n')


if __name__ == '__main__':
    start = time.time()
    Simulate_prepare()
    print('cost: ', time.time() - start)