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



def numpy2im(image_numpy, imtype=np.uint8):
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
        # input should be [0, 1]
    # image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) / 2. + 0.5) * 255.0
    # print(image_numpy.shape)
    image_numpy = image_numpy.astype(imtype)
    im = Image.fromarray(image_numpy)
    # im = Image.fromarray(image_numpy).resize((64, 64), Image.ANTIALIAS)
    return im  # np.array(im)


sample_perID = 1

sample_perA = 1


### fix id number , increase AU diversities #####
### one person simulate 9 (3 types emotion label * 3 types of AU ) sample

src_imgs_names = []
ID_numbers = 5000


for file in glob.glob("/home/yuchi/micro-expression/ganimation_replicate/datasets/celebA/imgs/*.jpg"):
    src_imgs_names.append(file.split('/')[-1])

# for i in range(1, ID_numbers+1):
#     img_name = str(i).zfill(6) + '.jpg'
#     src_imgs_names.append(img_name)

# ## store img names ####
# with open ("celebANames_pool5k.pkl", "wb") as wf:
#     pickle.dump(src_imgs_names, wf)

## ingredients for AU_exp
expression_list = ['Happiness', 'Sadness', 'Surprise', 'Fear', 'Anger', 'Disgust', 'Contempt']
au_map = {
    'Happiness': {'12': 1, '25': 0.5, '6': 0.5},
    'Sadness': {'4': 1, '1': 0.5, '15':0.5},
    'Surprise': {'1': 1, '2': 1,'5': 0.5, '25': 0.5, '26': 0.5},
    'Fear': {'5': 0.5, '20': 0.5, '25': 0.5, '26': 0.5},
    'Anger': {'4': 1, '7': 0.5, '23': 0.5},
    'Disgust': {'9': 1, '10': 0.5, '17': 1},
    'Contempt': {'14': 1}
}
openface_map = {'1': 0, '2': 1, '4':2, '5': 3, '6': 4, '7': 5, '9': 6, '10': 7, '12': 8, '14': 9, '15': 10, '17': 11, '20': 12, '23': 13, '25': 14, '26': 15,}
AU_MiEexp_pool = {'0':[], '1':[], '2':[]}
AU_MiEexpV_pool = {'0':[], '1':[], '2':[]}

for emotion_label in range(3):
    for i in range(ID_numbers):
        if emotion_label == 0:
            expression_type = random.choice(['Sadness', 'Fear', 'Anger', 'Disgust', 'Contempt'])
        elif emotion_label == 1:
            expression_type = 'Happiness'
        else:
            expression_type = 'Surprise'

        au_activation_inter=[]
        for i in range(10):
            au_activation_inter.append(np.zeros(17))
        au_activation_apx = np.zeros(17)
        potential_au = au_map[expression_type]
        for j in potential_au.keys():
            if random.uniform(0, 1) < potential_au[j]:
                apx_value = random.uniform(0.5, 1.5)/5
                inter_gap = apx_value/9
                for i in range(10):
                    au_activation_inter[i][openface_map[j]] = inter_gap*i


        AU_MiEexpV_pool[str(emotion_label)].append(au_activation_inter)


## store ingredients ####
with open("AU_MiEexpV_pool.pkl", "wb") as wf:
    pickle.dump(AU_MiEexpV_pool, wf)

## ingredients for AU_mae
mae_pool = {'0':[], '1':[], '2':[]}
ck_label_path = '../MER/Loading_file/macro_micro_3fold/ck_label.txt'
ck_data_path = '../MER/Loading_file/macro_micro_3fold/ck_data.txt'
with open (ck_label_path, 'r') as rf1:
    with open(ck_data_path, 'r') as rf2:
        lines = rf1.readlines()
        name_lines = rf2.readlines()
        pool_size = len(lines)
        for i in range(pool_size):
            emotion_label = lines[i].strip()
            mae_name = name_lines[i*2].strip()
            subject, trail = mae_name.split('/')[-1].split('_')[:2]
            mae_pool[str(emotion_label)].append([subject, trail])

# ## store ingredients ####
# with open("mae_pool.pkl", "wb") as wf:
#     pickle.dump(mae_pool, wf)



with open('celebANames_pool5k.pkl', 'rb') as rf:
    src_imgs_names = pickle.load(rf)

with open('AU_MiEexpV_pool.pkl', 'rb') as rf:
    AU_MiEexpV_pool = pickle.load(rf)
#
# with open('mae_pool.pkl', 'rb') as rf:
#     mae_pool = pickle.load(rf)

# with open('MMEW_pool.pkl', 'rb') as rf:
#     mmew_pool = pickle.load(rf)
#
# with open('oulu_pool.pkl', 'rb') as rf:
#     oulu_pool = pickle.load(rf)


dataset_dir = '/home/yuchi/micro-expression/ganimation_replicate/datasets/celebA/imgs'


## simulate imgs #####
opt = Options().parse()

test_model = create_model(opt)

tensor_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# path_list_file = open("train_list_file_exp.txt", "w")
# json.dump(src_imgs_names, path_list_file)
# path_list_file.close()


CK_AU_PATH = '../MER/Data/ck_AU_out_static/'
MMEW_AU_PATH = '../MER/Data/MMEW_out_static/'
oulu_AU_PATH = '../MER/Data/oulu_AU_static/'

# tar_dir = 'generated5k_mmewAU'
tar_dir = 'generated5k_oulu+mmew_AU'

target_sample_dir = None
exp_tar_aus_0 = {'0':None, '1':None, '2': None}
exp_tar_aus_1 = {'0':None, '1':None, '2': None}
mae_tar_aus_0 = {'0':None, '1':None, '2': None}
mae_tar_aus_1 = {'0':None, '1':None, '2': None}
exp_tar_aus={'0':[], '1':[], '2': []}
mae_tar_aus={'0':None, '1':None, '2': None}
shared_img_tensor = None

sample_number = 5000
src_imgs_names = src_imgs_names[:5000]

for idx_global in range(5000):
    print(idx_global)
    idx = idx_global%sample_number
    if idx%sample_number == 0 and idx>0:
        src_imgs_names = random.shuffle(src_imgs_names)
    tiems_shareID = 1
    if idx%tiems_shareID == 0:
        src_img_name = src_imgs_names[idx]

        if idx_global>1000 and tiems_shareID>1:
            target_sample_dir = src_img_name.split('.')[0]+str(idx_global)

        else:
            target_sample_dir = src_img_name.split('.')[0]

        img_path = os.path.join(dataset_dir, src_img_name)
        img = Image.open(img_path)
        img_tensor = tensor_transform(img)
        img_tensor = img_tensor.unsqueeze(dim=0)
        img_tensor = torch.cat((img_tensor, img_tensor), dim=0)
        shared_img_tensor = img_tensor
    else:
        img_tensor = shared_img_tensor
        target_sample_dir = target_sample_dir+str(idx%tiems_shareID)


    tiems_shareAU = 1
    for emotion_label in range(3):
        if idx%tiems_shareAU == 0:
            sample_au = True
        else:
            sample_au = False


        ### generate AU_exp based MiEs #####
        if sample_au:

            for i in range(10):
                exp_au = torch.tensor(AU_MiEexpV_pool[str(emotion_label)][idx][i]).unsqueeze(dim=0)
                exp_tar_aus[str(emotion_label)].append(exp_au)

            # exp_tar_aus_0[str(emotion_label)] = torch.tensor(AU_MiEexp_pool[str(emotion_label)][idx][0]).unsqueeze(dim=0)
            # exp_tar_aus_1[str(emotion_label)] = torch.tensor(AU_MiEexp_pool[str(emotion_label)][idx][1]).unsqueeze(dim=0)
            # exp_tar_aus[str(emotion_label)] = torch.cat((exp_tar_aus_0[str(emotion_label)], exp_tar_aus_1[str(emotion_label)]), dim=0)
        test_batch = {'src_img': img_tensor, 'tar_aus': exp_tar_aus[str(emotion_label)], 'src_aus': exp_tar_aus[str(emotion_label)],
                  'tar_img': img_tensor}
        test_model.feed_batch(test_batch)
        test_model.forward()
        cur_gen_faces = test_model.fake_img.detach().cpu().float().numpy() # to be simple, only generate one face each time
        ## save sample in the folder ##
        print(target_sample_dir)
        assert (0)
        saved_path_dir = os.path.join(tar_dir, "%s_%s_%s" % (target_sample_dir, 'exp', str(emotion_label)))
        if not os.path.exists(saved_path_dir):
            os.makedirs(saved_path_dir)
        for i in range(10):
            cur_gen_face = numpy2im(cur_gen_faces[i])
            saved_path = os.path.join(saved_path_dir, str(i)+'.jpg')
            cur_gen_face.save(saved_path)
