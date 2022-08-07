import os
import random
import argparse
import glob
import cv2
from utils import face_utils
from utils import cv_utils
import face_recognition
from PIL import Image
import torchvision.transforms as transforms
import torch
import pickle
import numpy as np
from models.models import ModelsFactory
from options.test_options import TestOptions
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


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

openface_map = {'1': 0, '2': 1, '4':2, '5': 3, '6': 4, '7': 5, '9': 6, '10': 7, '12': 8, '14': 9, '15': 10, '17': 11, '20': 12, '23': 13, '25': 14, '26': 15, '45': 16}


class MorphFacesInTheWild:
    def __init__(self, opt):
        self._opt = opt
        self._model = ModelsFactory.get_by_name(self._opt.model, self._opt)
        self._model.set_eval()
        self._transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                   std=[0.5, 0.5, 0.5])
                                              ])

    def morph_file(self, img_path, expresion):
        # img = cv_utils.read_cv2_img(img_path)
        img = face_recognition.load_image_file(img_path, mode='RGB')
        morphed_imgs = self._img_morph(img, expresion['au'])
        dir_name = os.path.splitext(os.path.basename(img_path))[0] + expresion['type']
        dir_path = os.path.join('./EmotioNet/generated_multi_faces_5repeat', dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        self._save_imgs(morphed_imgs, dir_path)



    def _img_morph(self, img, au_activation):
        faces = []
        bbs = face_recognition.face_locations(img)
        if len(bbs) > 0:
            y, right, bottom, x = bbs[0]
            bb = x, y, (right - x), (bottom - y)
            face = face_utils.crop_face_with_bb(img, bb)
            face = face_utils.resize_face(face)
        else:
            face = face_utils.resize_face(img)

        morphed_face_0 = self._morph_face(face, au_activation[0])
        morphed_face_1 = self._morph_face(face, au_activation[1])
        morphed_face_2 = self._morph_face(face, au_activation[2])

        faces = [morphed_face_0, morphed_face_1, morphed_face_2]

        return faces

    def _morph_face(self, face, expresion):
        face = torch.unsqueeze(self._transform(Image.fromarray(face)), 0)
        expresion = torch.unsqueeze(torch.from_numpy(expresion/5.0), 0)
        test_batch = {'real_img': face, 'real_cond': expresion, 'desired_cond': expresion, 'sample_id': torch.FloatTensor(), 'real_img_path': []}
        self._model.set_input(test_batch)
        imgs, _ = self._model.forward(keep_data_for_visuals=False, return_estimates=True)
        return imgs['fake_imgs_masked']

    def _save_imgs(self, imgs, dir_path):
        for i in range (len(imgs)):
            filepath = os.path.join(dir_path, '%s_out.png' % i)
            img = cv2.cvtColor(imgs[i], cv2.COLOR_RGB2BGR)
            cv2.imwrite(filepath, img)






def main():
    opt = TestOptions().parse()

    morph = MorphFacesInTheWild(opt)

    generated_num = 0

    AUs_used_times = 0

    expression_repeat = [{},{},{}]

    with open('./EmotioNet/selected_imgs_all.csv', 'rb') as f:
        with open('simulate_multi_labels_5repeat.csv', 'wb') as f_labels:
            for line in f:
                ########## generate three faces ###########

                ###########################################

                ######### generate one face #############

                line = line.strip('\n')
                img_path = os.path.join('./EmotioNet/sample_dataset_yh_2/imgs/', line)
                print img_path
                random_number = np.random.randint(0, 3)
                for i in range(3):
                    if i == 0:
                        expression_type = 'Surprise'
                    elif i == 1:
                        expression_type = 'Happiness'
                    else:
                        expression_type = random.choice(['Sadness', 'Fear', 'Anger', 'Disgust', 'Contempt'])

                    ##### AUs used times #######
                    au_activation_0 = np.zeros(opt.cond_nc)
                    au_activation_micro = np.zeros(opt.cond_nc)
                    au_activation_macro = np.zeros(opt.cond_nc)
                    potential_au = au_map[expression_type]
                    for j in potential_au.keys():
                        if random.uniform(0, 1) < potential_au[j]:
                            au_activation_micro[openface_map[j]] = random.uniform(0.5, 1.5)
                            au_activation_macro[openface_map[j]] = random.uniform(1.5, 4)
                    au_activation = [au_activation_0, au_activation_micro, au_activation_macro]
                    expression = {'type': expression_type, 'au': au_activation}
                    if generated_num % 5 == 0:
                        expression_repeat[i] = expression
                    morph.morph_file(img_path, expression_repeat[i])
                    dir_name = os.path.splitext(os.path.basename(img_path))[0]

                    f_labels.write(dir_name + expression['type']+ ' ' + expression['type'])
                    f_labels.write('\n')

                generated_num+=1
                AUs_used_times+=1
                if generated_num > 500:
                    break



if __name__ == '__main__':
    main()
