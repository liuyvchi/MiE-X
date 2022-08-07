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

num = 0

def Simulate_prepare():
    data_dir = 'EmotioNet/generated_multi_faces_5w/'
    file_path = 'simulate_multi_labels_5w.csv'
    save_dir = 'EmotioNet/generated_flo_5w/'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    emotion_dict = {'Anger': 0, 'Fear': 0, 'Sadness': 0, 'Happiness':1, 'Contempt': 0, 'Surprise': 2, 'Disgust': 0, '0': 0, '1': 1, '2': 2}

    start_name = '0_out.png'
    micro_name = '1_out.png'
    macro_name = '2_out.png'

    data_file = pd.read_csv(file_path, sep=' ')
    for index in range(len(data_file)):
        folder_name = data_file.iloc[index, 0]
        emotion = str(data_file.iloc[index, 1])
        emotion = emotion_dict[emotion]

        folder_path = os.path.join(data_dir, folder_name)

        img_1_path = os.path.join(folder_path, start_name)
        img_2_path = os.path.join(folder_path, micro_name)
        img_3_path = os.path.join(folder_path, macro_name)

        save_flo_dir = os.path.join(save_dir, folder_name)
        if not os.path.exists(save_flo_dir):
            os.makedirs(save_flo_dir)
        save_flo_path = os.path.join(save_flo_dir, 'flow.png')

        Flow_Pic(img_1_path, img_2_path, save_flo_path)
        print(save_flo_path)


        # if index < 120000:
        #     with open (train_image_file, 'a') as i:
        #         i.write(folder_path + '\n')
        #
        #     with open (train_label_file, 'a') as l:
        #         l.write(str(emotion) + '\n')
        # else:
        #     with open(vali_image_file, 'a') as i:
        #         i.write(folder_path + '\n')
        #
        #     with open(vali_label_file, 'a') as l:
        #         l.write(str(emotion) + '\n')

def Flow_Pic(img_1_path, img_2_path, save_path):

    img_1 = face_recognition.load_image_file(img_1_path, mode='RGB')
    img_2 = face_recognition.load_image_file(img_2_path, mode='RGB')

    img_1 = cv2.cvtColor(img_1, cv2.COLOR_RGB2BGR)
    img_2 = cv2.cvtColor(img_2, cv2.COLOR_RGB2BGR)

    one_g = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    two_g = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

    pic_size = img_1.shape
    hsv = np.zeros(pic_size)
    hsv[:,:,1] = 255

    flow = cv2.calcOpticalFlowFarneback(one_g, two_g, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    hsv[:,:,0] = ang * (180/ np.pi / 2)
    hsv[:,:,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    # hsv = np.asarray(hsv, dtype=np.float32)

    bgr = cv2.cvtColor(np.uint8(hsv),cv2.COLOR_HSV2BGR)
    cv2.imwrite(save_path, bgr)



1


# def test():
#      path_1 = '/home/user/Yuchi/iccv2019/MER/Data/SAMM_cropped/006/006_1_2/006_05562.jpg'
#      path_2 = '/home/user/Yuchi/iccv2019/MER/Data/SAMM_cropped/006/006_1_2/006_05588.jpg'
#      Flow_Pic (path_1, path_2, './flow.png', 'floow_2.png')



if __name__ == '__main__':
    start = time.time()
    Simulate_prepare()
    print('cost: ', time.time() - start)