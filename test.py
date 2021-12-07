import os
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
        morphed_img = self._img_morph(img, expresion)
        output_name = '%s_out_au1+2_1.png' % os.path.basename(img_path)
        self._save_img(morphed_img, output_name)

    def _img_morph(self, img, expresion):
        # bbs = face_recognition.face_locations(img)
        # if len(bbs) > 0:
        #     y, right, bottom, x = bbs[0]
        #     bb = x, y, (right - x), (bottom - y)
        #     face = face_utils.crop_face_with_bb(img, bb)
        #     face = face_utils.resize_face(face)
        # else:
        #     face = face_utils.resize_face(img)

        morphed_face = self._morph_face(img, expresion)

        return morphed_face

    def _morph_face(self, face, expresion):
        face = torch.unsqueeze(self._transform(Image.fromarray(face)), 0)
        expresion = torch.unsqueeze(torch.from_numpy(expresion/5.0), 0)
        test_batch = {'real_img': face, 'real_cond': expresion, 'desired_cond': expresion, 'sample_id': torch.FloatTensor(), 'real_img_path': []}
        self._model.set_input(test_batch)
        imgs, _ = self._model.forward(keep_data_for_visuals=False, return_estimates=True)
        return imgs['fake_imgs_masked']

    def _save_img(self, img, filename):
        filepath = os.path.join(self._opt.output_dir, filename)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filepath, img)


def main():
    opt = TestOptions().parse()
    if not os.path.isdir(opt.output_dir):
        os.makedirs(opt.output_dir)

    morph = MorphFacesInTheWild(opt)

    image_path = opt.input_path
    expression = np.zeros(opt.cond_nc)
    expression[0]=3
    expression[1]=3
    # # expression[4]=5
    # # expression[5]=5
    # expression[4]=0.5
    expression[8]=1
    # expression[9]=2

    morph.morph_file(image_path, expression)



if __name__ == '__main__':
    main()
