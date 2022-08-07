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
import random as random


def mix_prepare():
    Expression_version = 'Micro'
    Version = '0'

    Dataset_dir = '/home/user/Micro_expression/Micro_expression_dataset'
    Save_Dataset_dir = '/home/user/Yuchi/iccv2019/MER/Data'

    File_save_dir = '/home/user/Yuchi/iccv2019/MER/Loading_file/micro_3fold'


    csv_path = 'combined_3class_gt.csv'
    csv_path = os.path.join(Dataset_dir, csv_path)
    data = pd.read_csv(csv_path, header=None)

    subjects = []
    for index, row in data.iterrows():
        if row[1] in (subjects):
            continue
        subjects.append(row[1])
    random.shuffle(subjects)

    fold1 = subjects[:24]
    fold2 = subjects[24:47]
    fold3 = subjects[47:]
    folds = [fold1, fold2, fold3]
    sub_num = len(subjects)
    assert (len(fold1)+len(fold2)+len(fold3) == 68)



    for fold_num in range(len(folds)):

        test_data = data[data[1].isin(folds[fold_num])]
        train_validation_sub_num = sub_num - len(folds[fold_num])
        sub_num_vali = int(train_validation_sub_num/4)
        if fold_num == 0:
            train_validation_subs = folds[1] + folds[2]
        elif fold_num == 1:
            train_validation_subs = folds[0] + folds[2]
        else:
            train_validation_subs = folds[0] + folds[1]

        validation_subs = train_validation_subs[:sub_num_vali]
        train_subs = train_validation_subs[sub_num_vali:]

        train_data = data[data[1].isin(train_subs)]
        validation_data = data[data[1].isin(validation_subs)]

        data_list = [train_data, validation_data, test_data]


        for data_index in range(len(data_list)):

            DATA_FILE = '{}_fold{}_data{}.txt'.format(Expression_version, fold_num, data_index)
            DATA_FILE = os.path.join(File_save_dir, DATA_FILE)
            LABEL_FILE = '{}_fold{}_label{}.txt'.format(Expression_version, fold_num, data_index)
            LABEL_FILE = os.path.join(File_save_dir, LABEL_FILE)

            if os.path.exists(DATA_FILE):
                os.remove(DATA_FILE)
                os.remove(LABEL_FILE)

            # # =========================================================================================
            apex_path = 'SAMM/SAMM_label.csv'
            apex_path = os.path.join(Dataset_dir, apex_path)
            SAMM_path = 'SAMM/'
            SAMM_path = os.path.join(Dataset_dir, SAMM_path)
            save_path = 'SAMM_cropped/'
            save_path = os.path.join(Save_Dataset_dir, save_path)
            apex = pd.read_csv(apex_path, header=None)
            samm_data = data_list[data_index][data_list[data_index][0] == 'samm']
            for index, row in samm_data.iterrows():
                sub_path = os.path.join(SAMM_path, str(row[1]).zfill(3))
                save_sub_path = os.path.join(save_path, str(row[1]).zfill(3))

                test_path = os.path.join(sub_path, str(row[2]))
                save_test_path = os.path.join(save_sub_path, str(row[2]))

                if not os.path.exists(save_test_path):
                    os.makedirs(save_test_path)

                pixel = apex[apex[1] == row[2]]

                apex_num = int(pixel[4]) - int(pixel[3])

                pic_list = os.listdir(test_path)
                pic_list.sort()

                pic_apex_name = pic_list[apex_num]
                pic_apex_path = os.path.join(test_path, pic_apex_name)
                save_apex_pic_path = os.path.join(save_test_path, pic_apex_name)

                pic_onset_name = pic_list[0]
                pic_onset_path = os.path.join(test_path, pic_onset_name)
                save_onset_pic_path = os.path.join(save_test_path, pic_onset_name)

                bb = cropped_face(pic_apex_path, save_apex_pic_path)
                cropped_face_rest(bb, pic_onset_path, save_onset_pic_path)

                print(test_path)


                with open (LABEL_FILE, 'a') as l:
                    l.write(str(row[3]) + '\n')
                with open (DATA_FILE, 'a') as dl:
                    dl.write(save_onset_pic_path + '\n')
                    dl.write(save_apex_pic_path + '\n')



        # =================================================================================================

            emotion_dict = {'ne': 'negative', 'po': 'positive', 'sur': 'surprise'}

            SMIC_path = 'SMIC_all_raw/HS/'
            SMIC_path = os.path.join(Dataset_dir, SMIC_path)
            SMIC_save_path = 'SMIC_cropped/'
            SMIC_save_path = os.path.join(Save_Dataset_dir, SMIC_save_path)

            SMIC_data = data_list[data_index][data_list[data_index][0] == 'smic']

            for index, (_, row) in enumerate(SMIC_data.iterrows()):
                subject = int(row[1][1:])
                subject_name = 's' + str(subject)
                video_name = row[2].split('_')
                emotion = emotion_dict[video_name[1]]
                test_name = subject_name + '_' + video_name[1] + '_' + video_name[2]
                test_path = os.path.join(SMIC_path, subject_name, 'micro', emotion, test_name)
                save_test_path = os.path.join(SMIC_save_path, subject_name, emotion, test_name)

                if not os.path.exists(save_test_path):
                    os.makedirs(save_test_path)

                pic_list = os.listdir(test_path)
                pic_list.sort()

                onset_pic = pic_list[0]
                apex_pic = pic_list[int(len(pic_list) / 2)]

                onset_path = os.path.join(test_path, onset_pic)
                apex_path = os.path.join(test_path, apex_pic)

                onset_save_path = os.path.join(save_test_path, onset_pic)
                apex_save_path = os.path.join(save_test_path, apex_pic)

                bb = cropped_face(onset_path, onset_save_path)
                cropped_face_rest(bb, apex_path, apex_save_path)

                print(test_path)

                with open (DATA_FILE, 'a') as d:
                    d.write(onset_save_path + '\n')
                    d.write(apex_save_path + '\n')

                with open (LABEL_FILE, 'a') as l:
                    l.write(str(row[3]) + '\n')

        # ===================================================================================================

            CASMEII_label_file = 'CASME2-coding-20140508.xlsx'
            CASMEII_label_file = os.path.join(Dataset_dir, CASMEII_label_file)
            CASMEII_path = 'CASME2_RAW_selected/'
            CASMEII_path = os.path.join(Dataset_dir, CASMEII_path)
            CASMEII_save_path = 'CASMEII_cropped/'
            CASMEII_save_path = os.path.join(Save_Dataset_dir, CASMEII_save_path)

            CASMEII_pic_data = pd.read_excel(CASMEII_label_file, header=0)

            CASMEII_data = data_list[data_index][data_list[data_index][0] == 'casme2']

            for index, row in CASMEII_data.iterrows():
                subject, test, label = row[1], row[2], row[3]

                CASMEII_temp = CASMEII_pic_data[CASMEII_pic_data['Subject'] == int(subject[3:])]
                CASMEII_temp = CASMEII_temp[CASMEII_temp['Filename'] == test]
                CASMEII_onset = 'img' + str(CASMEII_temp.values[0][3]) + '.jpg'
                CASMEII_apex = 'img' + str(CASMEII_temp.values[0][4]) + '.jpg'

                CASMEII_test_path = os.path.join(CASMEII_path, subject, test)
                CASMEII_save_test_path = os.path.join(CASMEII_save_path, subject, test)

                if not os.path.exists(CASMEII_save_test_path):
                    os.makedirs(CASMEII_save_test_path)

                CASMEII_onset_path = os.path.join(CASMEII_test_path, CASMEII_onset)
                CASMEII_onset_save_path = os.path.join(CASMEII_save_test_path, CASMEII_onset)

                CASMEII_apex_path = os.path.join(CASMEII_test_path, CASMEII_apex)
                CASMEII_apex_save_path = os.path.join(CASMEII_save_test_path, CASMEII_apex)

                bb = cropped_face(CASMEII_onset_path, CASMEII_onset_save_path)
                cropped_face_rest(bb, CASMEII_apex_path, CASMEII_apex_save_path)

                print(CASMEII_test_path)

                with open (DATA_FILE, 'a') as d:
                    d.write(CASMEII_onset_save_path + '\n')
                    d.write(CASMEII_apex_save_path + '\n')

                with open (LABEL_FILE, 'a') as l:
                    l.write(str(label) + '\n')

    # =======================================================================================================
    #
    #     image_path = './CK+/cohn-kanade-images/'
    #     label_path = './CK+/Emotion/'
    #     save_path = './CK_2pic/'
    #     au_path = './CK+/FACS/'
    #     flow_path = './CK_flow/'
    #
    #     CK_dict = {0: 1, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 1, 7: 2}
    #
    #     for subject in os.listdir(label_path):
    #         subject_label_path = os.path.join(label_path, subject)
    #         subject_image_path = os.path.join(image_path, subject)
    #         # subject_au_path = os.path.join(au_path, subject)
    #         subject_save_path = os.path.join(save_path, subject)
    #         subject_flow_path = os.path.join(flow_path, subject)
    #         for test in os.listdir(subject_label_path):
    #             test_label_path = os.path.join(subject_label_path, test)
    #             test_image_path = os.path.join(subject_image_path, test)
    #             # test_au_path = os.path.join(subject_au_path, test)
    #             test_save_path = os.path.join(subject_save_path, test)
    #             test_flow_path = os.path.join(subject_flow_path, test)
    #             if os.listdir(test_label_path):
    #
    #                 with open (os.path.join(test_label_path, os.listdir(test_label_path)[-1]), 'r') as rl:
    #                     label = rl.readline()
    #                     label = int(float(label.strip('\n')))
    #
    #                 if label != 0:
    #                     with open (LABEL_FILE, 'a') as l:
    #                         l.write(str(CK_dict[label]) + '\n')
    #
    #                     image_dir = os.listdir(test_image_path)
    #                     image_dir = [item for item in image_dir if os.path.splitext(item)[1] == '.png']
    #                     image_dir = sorted(image_dir)
    #
    #                     onset_path = os.path.join(test_image_path, image_dir[0])
    #                     onset_save_path = os.path.join(test_save_path, image_dir[0])
    #
    #                     apex_num = int(len(image_dir) / 2)
    #
    #                     apex_path = os.path.join(test_image_path, image_dir[apex_num])
    #                     apex_save_path = os.path.join(test_save_path, image_dir[apex_num])
    #
    #                     # if not os.path.exists(test_save_path):
    #                     #     os.makedirs(test_save_path)
    #
    #                     # image_landmark(onset_path, onset_save_path)
    #                     # image_landmark(apex_path, apex_save_path)
    #
    #                     with open (DATA_FILE, 'a') as d:
    #                         d.write(onset_save_path + '\n')
    #                         d.write(apex_save_path + '\n')
    #
    #                     with open (SUBJECT_FILE, 'a') as s:
    #                         s.write('Macro\n')
    #
    #                     # with open (os.path.join(test_au_path, os.listdir(test_au_path)[-1]), 'r') as ra:
    #                     #     AU_file = ra.readlines()
    #
    #                     # AU_label = [int(float(item.split('   ')[1])) for item in AU_file]
    #                     # CK_AU = ''
    #                     # for item in AU_label:
    #                     #     CK_AU += str(item) + '+'
    #                     # CK_AU = CK_AU[:-1]
    #
    #                     # with open (AU_FILE, 'a') as a:
    #                     #     a.write(CK_AU + '\n')
    #
    #                     Flow_pic_path = os.path.join(test_flow_path, 'flow.png')
    #
    #                     # if not os.path.exists(test_flow_path):
    #                     #     os.makedirs(test_flow_path)
    #
    #                     print(test_label_path)
    #
    #                     crop_pic(onset_path, apex_path, onset_save_path, apex_save_path, Flow_pic_path)
    #
    #                     # Flow_Pic(onset_path, apex_path, Flow_pic_path)
    #
    #                     with open (FLOW_FILE, 'a') as m:
    #                         m.write(Flow_pic_path + '\n')


def cropped_face(pic_path, save_path):
    img = face_recognition.load_image_file(pic_path, mode='RGB')
    bbs = face_recognition.face_locations(img)
    if len(bbs) > 0:
        y, right, bottom, x = bbs[0]
        bb = x, y, (right - x), (bottom - y)
        face = face_utils.crop_face_with_bb(img, bb)
        face = face_utils.resize_face(face)
    else:
        face = face_utils.resize_face(img)

    # face.save(save_path)
    img = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, img)

    return bbs

def cropped_face_rest(bbs, pic_path, save_path):
    img = face_recognition.load_image_file(pic_path, mode='RGB')
    if len(bbs) > 0:
        y, right, bottom, x = bbs[0]
        bb = x, y, (right - x), (bottom - y)
        face = face_utils.crop_face_with_bb(img, bb)
        face = face_utils.resize_face(face)
    else:
        face = face_utils.resize_face(img)

    # face.save(save_path)
    img = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, img)



if __name__ == '__main__':
    start = time.time()
    mix_prepare()
    print('cost: ', time.time() - start)
