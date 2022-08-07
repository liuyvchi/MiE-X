from data import create_dataloader
from model import create_model
from visualizer import Visualizer
import copy
import time
import os
import torch
import numpy as np
from PIL import Image
from options import Options
import torchvision.transforms as transforms
import random
import json
import pickle
import pandas as pd
import glob

emotion_types = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
MMEW_root_dir = '../MER/Data/MMEW_out_static/'
MMEW_pool = {'0':[], '1':[], '2':[]}

## ingredients for MMEW
mae_tar_aus_0 = {'0':None, '1':None, '2': None}
mae_tar_aus_1 = {'0':None, '1':None, '2': None}
mae_tar_aus={'0':None, '1':None, '2': None}


for emotion_type in emotion_types:
    if emotion_type in ['anger', 'disgust', 'fear', 'sadness']:
        emotion_label = 0
    elif emotion_type in ['happiness'] :
        emotion_label = 1
    elif emotion_type in ['surprise']:
        emotion_label = 2
    else:
        print(emotion_type)
        assert (0)

    for _, videos_dirs,_ in os.walk(os.path.join(MMEW_root_dir, emotion_type)):
        for dir in videos_dirs:
            trail_aus_path = os.path.join(MMEW_root_dir, emotion_type, dir,dir+'.csv')
            MMEW_pool[str(emotion_label)].append(trail_aus_path)
    print(len(MMEW_pool[str(emotion_label)]))
## store ingredients ####
with open("MMEW_pool.pkl", "wb") as wf:
    pickle.dump(MMEW_pool, wf)



