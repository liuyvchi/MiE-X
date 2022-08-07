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

emotion_types = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']
oulu_root_dir = '../MER/Data/oulu_AU_static/'
oulu_pool = {'0':[], '1':[], '2':[]}

## ingredients for MMEW
mae_tar_aus_0 = {'0':None, '1':None, '2': None}
mae_tar_aus_1 = {'0':None, '1':None, '2': None}
mae_tar_aus={'0':None, '1':None, '2': None}

persons = os.listdir(os.path.join(oulu_root_dir))
for person in persons:
    if "P" not in person:
        continue
    for emotion_type in emotion_types:
        if emotion_type in ['Anger', 'Disgust', 'Fear', 'Sadness']:
            emotion_label = 0
        elif emotion_type in ['Happiness']:
            emotion_label = 1
        elif emotion_type in ['Surprise']:
            emotion_label = 2
        else:
            print(emotion_type)
            assert (0)
        trail_aus_path = os.path.join(oulu_root_dir, person, emotion_type, emotion_type+'.csv')
        print(trail_aus_path)
        oulu_pool[str(emotion_label)].append(trail_aus_path)

## store ingredients ####
with open("oulu_pool.pkl", "wb") as wf:
    pickle.dump(oulu_pool, wf)



