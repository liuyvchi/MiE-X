import cv2

import glob
import os
from PIL import Image
from facenet_pytorch import MTCNN
import numpy as np


def compute_nme(target, query):


    # target = target.cpu().numpy()
    # query = query.cpu().numpy()
    base = np.linalg.norm(query[1] - query[0])
    # print(target)
    # print(query)
    nme = np.mean(np.linalg.norm(target - query, axis=1))/base

    return nme



mtcnn = MTCNN(keep_all=True, device='cuda:0')


# query_img_path = '/home/yuchi/micro-expression/GANimation/EmotioNet/samples/N_0000000202_000130/0_out.png'
query_img_path = '/home/yuchi/micro-expression/GANimation/EmotioNet/samples/N_0000000202_000100/0_out.png'
query_img = cv2.imread(query_img_path)
# print()
query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
query_img = Image.fromarray(query_img)
q_boxes, q_probs, q_landmarks = mtcnn.detect(query_img, landmarks=True)


# print(landmarks)
# assert (0)

sample_list_path = '/home/yuchi/micro-expression/MER/Loading_file_new/5w_real+ck+prior_fold/simulate_fold0_data.txt'
sample_front_list_path = '/home/yuchi/micro-expression/MER/Loading_file_new/5w_real+ck+prior_fold/simulate_fold0_data_front.txt'
data_root = '/home/yuchi/micro-expression/GANimation/'
count=0
pose_count=0
with open(sample_front_list_path, 'w') as wf:
    with open(sample_list_path, 'r') as rf:
        lines = rf.readlines()
        for line in lines:
            sample_dir, label = line.strip().split(' ')
            sample_path = os.path.join(data_root, sample_dir)
            paths = glob.glob(os.path.join(sample_path, '*.png'))
            img_path = os.path.join(data_root, paths[0])
            # img_path = '/home/yuchi/micro-expression/GANimation/EmotioNet/samples/N_0000000202_000840/1_out.png'

            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            boxes, probs, landmarks = mtcnn.detect(img, landmarks=True)
            # print(landmarks[0])
            # print(q_landmarks[0])
            # assert (0)
            try:
                nme = compute_nme(landmarks[0], q_landmarks[0])
            except:
                print(img_path)
            # print(img_path)
            # print(nme)
            # assert (0)
            if nme>0.4:
                wf.write(sample_dir+ ' '+ '-1' + '\n')
                pose_count+=1
            else:
                wf.write(sample_dir+ ' '+ label + '\n')

            count += 1
            if count%1000==0:
                print(count)
                print(pose_count)