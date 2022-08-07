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
import face_recognition
from Utils import face_utils


def mix_prepare():
    Expression_version = 'Micro'
    Version = '0'

    Dataset_dir = '/home/user/Micro_expression/Micro_expression_dataset'

    apx_LABEL_FILE = os.path.join(Dataset_dir, 'combined_apx_label.csv')

    csv_path = 'combined_3class_gt.csv'
    csv_path = os.path.join(Dataset_dir, csv_path)
    data = pd.read_csv(csv_path, header=None)

# # =========================================================================================
    apex_path = 'SAMM/SAMM_label.csv'
    apex_path = os.path.join(Dataset_dir, apex_path)
    SAMM_path = 'SAMM/'
    SAMM_path = os.path.join(Dataset_dir, SAMM_path)

    apex = pd.read_csv(apex_path, header=None)
    samm_data = data[data[0] == 'samm']
    for _, row in samm_data.iterrows():
        sub_path = os.path.join(SAMM_path, str(row[1]).zfill(3))

        test_path = os.path.join(sub_path, str(row[2]))

        pixel = apex[apex[1] == row[2]]

        apex_num = int(pixel[4]) - int(pixel[3])

        print(test_path)

        with open (apx_LABEL_FILE, 'a') as l:
            l.write(row[0] + ' ' + str(row[1]).zfill(3) + ' ' + str(row[2]) + ' ' + str(apex_num) + ' ' + str(row[3]) + '\n')


# =================================================================================================

    emotion_dict = {'ne': 'negative', 'po': 'positive', 'sur': 'surprise'}

    SMIC_path = 'SMIC_all_raw/HS/'
    SMIC_path = os.path.join(Dataset_dir, SMIC_path)

    SMIC_data = data[data[0] == 'smic']

    for index, (_, row) in enumerate(SMIC_data.iterrows()):

        subject = int(row[1][1:])
        subject_name = 's' + str(subject)
        video_name = row[2].split('_')
        emotion = emotion_dict[video_name[1]]
        test_name = subject_name + '_' + video_name[1] + '_' + video_name[2]
        test_path = os.path.join(SMIC_path, subject_name, 'micro', emotion, test_name)

        pic_list = os.listdir(test_path)
        pic_list.sort()

        apex_num = len(pic_list) / 2


        print(test_path)

        with open (apx_LABEL_FILE, 'a') as l:
            l.write(row[0] + ' ' + subject_name + ' ' + test_name + ' ' +  str(apex_num) + ' ' + str(row[3]) + '\n')


# ===================================================================================================

    CASMEII_label_file = 'CASME2-coding-20140508.xlsx'
    CASMEII_label_file = os.path.join(Dataset_dir, CASMEII_label_file)
    CASMEII_path = 'CASME2_RAW_selected/'
    CASMEII_path = os.path.join(Dataset_dir, CASMEII_path)
    CASMEII_save_path = 'CASMEII_cropped/'

    CASMEII_pic_data = pd.read_excel(CASMEII_label_file, header=0)

    CASMEII_data = data[data[0] == 'casme2']

    for index, row in CASMEII_data.iterrows():
        subject, test, label = row[1], row[2], row[3]

        CASMEII_temp = CASMEII_pic_data[CASMEII_pic_data['Subject'] == int(subject[3:])]
        CASMEII_temp = CASMEII_temp[CASMEII_temp['Filename'] == test]
        apex_num = int(CASMEII_temp.values[0][4]) - int(CASMEII_temp.values[0][3])

        CASMEII_test_path = os.path.join(CASMEII_path, subject, test)

        print(CASMEII_test_path)

        with open(apx_LABEL_FILE, 'a') as l:
            l.write(row[0] + ' ' + str(row[1]) + ' ' + str(row[2]) + ' ' + str(apex_num) + ' ' + str(row[3]) + '\n')





if __name__ == '__main__':
    start = time.time()
    mix_prepare()
    print('cost: ', time.time() - start)
