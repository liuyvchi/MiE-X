import torch
import os

from PIL import Image
from torch import stack
from torch.utils.data import Dataset
import numpy as np
import cv2



##flo
import face_recognition
from Utils import face_utils
import sys
sys.path.append("../")
from PrepareFlo_micro3fold import Flow_Pic, cropped_face, cropped_face_rest

class GAN_loader(Dataset):
    def __init__(self, args, data_path, label_path, transform=None, label_transform=None, Micro_set=False):
        self.DATA_ROOT = args.DATA_ROOT
        self.data_path = data_path
        self.label_path = label_path

        with open (self.data_path, 'r') as d:
            self.data = d.readlines()

        with open (self.label_path, 'r') as l:
            self.label = l.readlines()

        self.transform = transform
        self.label_transform = label_transform

        self.size = len(self.label)
        self.Micro_set = Micro_set

    def __getitem__(self, index):
        if self.Micro_set == False:
            relative_img_dir = self.data[index].strip('\n')
            # relative_img_dir = relative_img_dir.split('/', 1)[1]
            img_dir =  os.path.join(self.DATA_ROOT, relative_img_dir)

            start_path = os.path.join(img_dir, '0_out.png')
            # ///// used to choose image ///////
            # flag = np.random.randint(0, 2)
            #
            flag = 0
            if flag == 0:
                end_path = os.path.join(img_dir, '1_out.png')
                domain_label = 0
            else:
                end_path = os.path.join(img_dir, '2_out.png')
                domain_label = 1
        else:
            start_path = self.data[index * 2].strip('\n')
            end_path = self.data[index * 2 + 1].strip('\n')
            domain_label = 0

        start_image = Image.open(start_path).convert('RGB')
        end_image = Image.open(end_path).convert('RGB')


        label = int(float(self.label[index].strip('\n')))

        if self.transform:
            start_image = self.transform(start_image)
            end_image = self.transform(end_image)

        imges = torch.stack((start_image, end_image), 0)


        return {"images": imges, "label": label, "domain_label": domain_label}

    def __len__(self):
        return self.size

class New_loader(Dataset):
    def __init__(self, args, data_path, sample_num, mix_type = 1, transform=None, label_transform=None, Micro_set=False):
        self.DATA_ROOT = args.SIMU_DATA_ROOT
        self.data_path = data_path
        self.mix_type = mix_type


        with open (self.data_path, 'r') as d:
            self.data = d.readlines()
        self.single_type_num = round(len(self.data)/self.mix_type)
        if self.mix_type  == 1:
            self.data = self.data[:sample_num]
        elif self.mix_type == 2:
            self.data_0 = self.data[:sample_num]
            self.data_1 = self.data[self.single_type_num:self.single_type_num+sample_num]
            self.data = self.data_0 + self.data_1
        elif self.mix_type == 3:
            self.data_0 = self.data[:sample_num]
            self.data_1 = self.data[self.single_type_num:self.single_type_num+sample_num]
            self.data_2 = self.data[self.single_type_num*2:self.single_type_num*2+sample_num]
            self.data = self.data_0 + self.data_1 + self.data_2
        else:
            assert ("wrong mix_type")

        self.transform = transform
        self.label_transform = label_transform

        self.size = len(self.data)

    def __getitem__(self, index):
        relative_img_dir = self.data[index].strip('\n')
        # relative_img_dir = relative_img_dir.split('/', 1)[1]
        img_dir_label =  os.path.join(self.DATA_ROOT, relative_img_dir)
        img_dir = img_dir_label.split(' ')[0]
        label = int(img_dir_label.split(' ')[1])
        # start_path = os.path.join(img_dir, '0_out.png')
        start_path = os.path.join(img_dir, '0.jpg')

        # end_path = os.path.join(img_dir, '1_out.png')
        end_path = os.path.join(img_dir, '1.jpg')
        domain_label = 0
        if self.mix_type == 2:
            if index < self.size/2:
                domain_label = 0
            else:
                domain_label = 1
        elif self.mix_type == 3:
            if index < self.size/3:
                domain_label = 0
            elif index < 2 * self.size/3:
                domain_label = 1
            else:
                domain_label = 2

        start_image = Image.open(start_path).convert('RGB')
        end_image = Image.open(end_path).convert('RGB')

        if self.transform:
            start_image = self.transform(start_image)
            end_image = self.transform(end_image)

        imges = torch.stack((start_image, end_image), 0)

        return {"images": imges, "label": label, "domain_label": domain_label}

    def __len__(self):
        return self.size

class New_loader_flo(Dataset):
    def __init__(self, args, data_path, sample_num, mix_type = 1, transform=None, label_transform=None, Micro_set=False):
        self.DATA_ROOT = args.SIMU_DATA_ROOT
        self.data_path = data_path
        self.mix_type = mix_type


        with open (self.data_path, 'r') as d:
            self.data = d.readlines()
        self.single_type_num = round(len(self.data)/self.mix_type)
        if self.mix_type  == 1:
            self.data = self.data[:sample_num]
        elif self.mix_type == 2:
            self.data_0 = self.data[:sample_num]
            self.data_1 = self.data[self.single_type_num:self.single_type_num+sample_num]
            self.data = self.data_0 + self.data_1
        elif self.mix_type == 3:
            self.data_0 = self.data[:sample_num]
            self.data_1 = self.data[self.single_type_num:self.single_type_num+sample_num]
            self.data_2 = self.data[self.single_type_num*2:self.single_type_num*2+sample_num]
            self.data = self.data_0 + self.data_1 + self.data_2
        else:
            assert ("wrong mix_type")

        self.transform = transform
        self.label_transform = label_transform

        self.size = len(self.data)

    def __getitem__(self, index):
        relative_img_dir = self.data[index].strip('\n')
        # relative_img_dir = relative_img_dir.split('/', 1)[1]
        img_dir_label =  os.path.join(self.DATA_ROOT, relative_img_dir)
        img_dir = img_dir_label.split(' ')[0]
        label = int(img_dir_label.split(' ')[1])
        start_path = os.path.join(img_dir, '0_out.png')

        end_path = os.path.join(img_dir, '1_out.png')
        domain_label = 0
        if self.mix_type == 2:
            if index < self.size/2:
                domain_label = 0
            else:
                domain_label = 1
        elif self.mix_type == 3:
            if index < self.size/3:
                domain_label = 0
            elif index < 2 * self.size/3:
                domain_label = 1
            else:
                domain_label = 2

        img_1, bbs = cropped_face(start_path)
        img_2 = cropped_face_rest(bbs, end_path)


        bgr = Flow_Pic(img_1, img_2)

        flow_image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        start_image = Image.open(start_path).convert('RGB')
        end_image = Image.open(end_path).convert('RGB')

        if self.transform:
            start_image = self.transform(start_image)
            end_image = self.transform(end_image)
            flow_image = self.transform(flow_image)

        # imges = torch.cat((start_image, end_image, flow_image), 0)
        imges = flow_image



        return {"images": imges, "label": label, "domain_label": domain_label}

    def __len__(self):
        return self.size


class Micro_loader_flo(Dataset):
    def __init__(self, args, data_path, label_path, target_name=None, transform=None, label_transform=None, Micro_set=False):
        self.DATA_ROOT = args.DATA_ROOT
        self.data_path = data_path
        self.label_path = label_path

        with open (self.data_path, 'r') as d:
            self.data = d.readlines()

        with open (self.label_path, 'r') as l:
            self.label = l.readlines()

        self.transform = transform
        self.label_transform = label_transform

        self.size = len(self.label)
        self.Micro_set = Micro_set

        self.target_name = target_name

    def __getitem__(self, index):

        start_path = self.data[index * 2].strip('\n')
        end_path = self.data[index * 2 + 1].strip('\n')
        start_path  = os.path.join(self.DATA_ROOT, start_path)
        end_path = os.path.join(self.DATA_ROOT, end_path)
        domain_label = 0

        start_image = Image.open(start_path).convert('RGB')
        end_image = Image.open(end_path).convert('RGB')


        label = int(float(self.label[index].strip('\n')))

        img_1, bbs = cropped_face(start_path)
        img_2 = cropped_face_rest(bbs, end_path)

        bgr = Flow_Pic(img_1, img_2)

        flow_image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        if self.transform:
            start_image = self.transform(start_image)
            end_image = self.transform(end_image)
            flow_image = self.transform(flow_image)

        # imges = torch.cat((start_image, end_image, flow_image), 0)
        imges = flow_image

        return {"images": imges, "label": label, "domain_label": domain_label}

    def __len__(self):
        return self.size

class Micro_loader(Dataset):
    def __init__(self, args, data_path, label_path, target_name=None, transform=None, label_transform=None, Micro_set=False):
        self.DATA_ROOT = args.DATA_ROOT
        self.data_path = data_path
        self.label_path = label_path

        with open (self.data_path, 'r') as d:
            self.data = d.readlines()

        with open (self.label_path, 'r') as l:
            self.label = l.readlines()

        self.transform = transform
        self.label_transform = label_transform

        self.size = len(self.label)
        self.Micro_set = Micro_set

        self.target_name = target_name
        self.target_index = []
        if self.target_name != None:
            for index in range(len(self.label)):
                start_path = self.data[index * 2].strip('\n')
                end_path = self.data[index * 2 + 1].strip('\n')
                start_path = os.path.join(self.DATA_ROOT, start_path)
                end_path = os.path.join(self.DATA_ROOT, end_path)
                if self.target_name in start_path:
                    self.target_index.append(index)
            self.size = len(self.target_index)

    def __getitem__(self, index):
        if self.target_name != None:
            index = self.target_index[index]
        start_path = self.data[index * 2].strip('\n')
        end_path = self.data[index * 2 + 1].strip('\n')
        start_path  = os.path.join(self.DATA_ROOT, start_path)
        end_path = os.path.join(self.DATA_ROOT, end_path)
        domain_label = 0

        start_image = Image.open(start_path).convert('RGB')
        end_image = Image.open(end_path).convert('RGB')


        label = int(float(self.label[index].strip('\n')))

        if self.transform:
            start_image = self.transform(start_image)
            end_image = self.transform(end_image)

        imges = torch.stack((start_image, end_image), 0)

        selected_label = False

        return {"images": imges, "label": label, "domain_label": domain_label}

    def __len__(self):
        return self.size

class GAN_loader_flo(Dataset):
    def __init__(self, args, data_path, label_path, transform=None, label_transform=None, Micro_set=False):
        self.DATA_ROOT = args.DATA_ROOT
        self.data_path = data_path
        self.label_path = label_path

        with open (self.data_path, 'r') as d:
            self.data = d.readlines()

        with open (self.label_path, 'r') as l:
            self.label = l.readlines()

        assert (len(self.data) == len(self.label))

        self.transform = transform
        self.label_transform = label_transform

        self.size = len(self.label)
        self.Micro_set = Micro_set

    def __getitem__(self, index):

        flo_path = self.data[index].strip('\n')
        if self.Micro_set == False:
            flo_path = os.path.join(self.DATA_ROOT, flo_path)
        else:
            flo_path = flo_path

        flo_image = Image.open(flo_path).convert('RGB')

        domain_label = 0

        label = int(float(self.label[index].strip('\n')))

        if self.transform:
            flo_image = self.transform(flo_image)


        return {"images": flo_image, "label": label, "domain_label": domain_label}

    def __len__(self):
        return self.size


class GAN_loader_comb(Dataset):
    def __init__(self, args, data_path, label_path, transform=None, label_transform=None, Micro_set=False):
        self.DATA_ROOT = args.DATA_ROOT
        self.data_path = data_path
        self.label_path = label_path

        with open (self.data_path, 'r') as d:
            self.data = d.readlines()

        with open (self.label_path, 'r') as l:
            self.label = l.readlines()

        self.transform = transform
        self.label_transform = label_transform

        self.size = len(self.label)
        self.Micro_set = Micro_set

    def __getitem__(self, index):
        if self.Micro_set == False:
            relative_img_dir = self.data[index].strip('\n')
            # relative_img_dir = relative_img_dir.split('/', 1)[1]
            img_dir =  os.path.join(self.DATA_ROOT, relative_img_dir)

            start_path = os.path.join(img_dir, '0_out.png')
            # ///// used to choose image ///////
            # flag = np.random.randint(0, 2)
            #
            flag = 0
            if flag == 0:
                end_path = os.path.join(img_dir, '1_out.png')
                domain_label = 0
            else:
                end_path = os.path.join(img_dir, '2_out.png')
                domain_label = 1

            trail = relative_img_dir.split('\n')[-1]
            flow_path = os.path.join(self.DATA_ROOT, 'EmotioNet/generated_flo_5w/', trail, 'flow.png')


        else:
            start_path = self.data[index * 2].strip('\n')
            end_path = self.data[index * 2 + 1].strip('\n')
            domain_label = 0

            split_list = start_path.strip('\n').split('/')
            data_source = split_list[-4]+'_flo'
            subject = split_list[-3]
            trail = split_list[-2]
            flo_path = os.path.join('/home/yuchi/micro-expression/MER/Data', data_source, subject, trail, 'flow.png')

        start_image = Image.open(start_path).convert('RGB')
        end_image = Image.open(end_path).convert('RGB')
        flow_image = Image.open(flo_path).convert('RGB')


        label = int(float(self.label[index].strip('\n')))

        if self.transform:
            start_image = self.transform(start_image)
            end_image = self.transform(end_image)
            flow_image = self.transform(flow_image)

        imges = torch.stack((start_image, end_image, flow_image), 0)


        return {"images": imges, "label": label, "domain_label": domain_label}

    def __len__(self):
        return self.size