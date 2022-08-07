import os
import glob
import random

expression_list = ['Happiness', 'Sadness', 'Surprise', 'Fear', 'Anger', 'Disgust']

target_expresion_list = ['Anger', 'Fear', 'Disgust']

mapping = {'Anger': 0,'Happiness': 1, 'Surprise':2, 'Fear':3, 'Disgust':4}

MMEW_dir_path = 'Data/MMEW_final/Micro-Expression'

target_data_file = 'Loading_file_new/samm_5fold/samm_fold1_data1_5class.txt'
target_label_file = 'Loading_file_new/samm_5fold/samm_fold1_label1_5class.txt'


sample_label_dic = {}
with open('Data/SAMM_label.csv', 'r') as rf:
    for line in rf:
        line = line.strip('\n')
        info = line.split(',')
        sample_name = info[1]
        sub_id = info[0]
        exp_type = info[-2]
        if exp_type in mapping.keys():
            label = mapping[exp_type]
            sample_label_dic[sample_name] = label
            # print(sample_name)

subject_sample_dic={}

with open ('Loading_file_new/samm_5fold/samm_fold1_data1.txt', 'r') as f:
    sample_lines = f.readlines()
    num_samples = int(len(sample_lines)/2)
    for i in range(num_samples):
        line = sample_lines[2*i].strip('\n')
        info = line.split('/')
        sample_name = info[3]
        sub_id = str(int(info[2]))
        if sample_name in sample_label_dic.keys():
            label = sample_label_dic[sample_name]
            oneset_path = sample_lines[2 * i].strip()
            apex_path = sample_lines[2 *i +1].strip()
            if sub_id not in subject_sample_dic.keys():
                subject_sample_dic[sub_id] = []
            subject_sample_dic[sub_id].append([oneset_path, apex_path, label])

print(subject_sample_dic.keys())

sub_all = ['6', '7', '9', '14', '11', '12', '13', '10', '15', '16', '17', '18', '19','20','21', '22', '23', '24', '26', '28','30', '31', '32', '33','34','35','36','37']
random.shuffle(sub_all)
subs_len = int(len(sub_all))

fold0_sub = sub_all[0:6]
fold1_sub = sub_all[6:11]
fold2_sub = sub_all[11:16]
fold3_sub = sub_all[16:22]
fold4_sub = sub_all[22:28]
# fold0_sub = ['6', '7', '9', '14']
# fold1_sub = ['11', '12', '13', '10']
# fold2_sub = ['15', '16', '17', '18', '19','20']
# fold3_sub = ['21', '22', '23', '24', '26', '28']
# fold4_sub = ['30', '31', '32', '33','34','35','36','37']
fold_subs_ids = []
fold_subs_ids.append(fold0_sub)
print(fold0_sub)
fold_subs_ids.append(fold1_sub)
print(fold1_sub)
fold_subs_ids.append(fold2_sub)
print(fold2_sub)
fold_subs_ids.append(fold3_sub)
print(fold3_sub)
fold_subs_ids.append(fold4_sub)
print(fold4_sub)
fold_num = 5
all_sub = fold0_sub+fold1_sub+fold2_sub+fold3_sub+fold4_sub
for i in range(fold_num):
    excluede_sub_list = fold_subs_ids[i]
    sub_list = [id for id in all_sub if str(id) not in excluede_sub_list]
    with open('Loading_file_new/samm_5fold/samm_fold{}_train_data.txt'.format(i), 'w') as rf_data_foldtrain:
        with open('Loading_file_new/samm_5fold/samm_fold{}_train_label.txt'.format(i), 'w') as rf_label_foldtrain:
            for sub_id in sub_list:

                sub_samples = subject_sample_dic[sub_id]
                for sample in sub_samples:
                    onset_path = sample[0]
                    apex_path = sample[1]
                    label = sample[2]
                    rf_data_foldtrain.write(onset_path + '\n')
                    rf_data_foldtrain.write(apex_path + '\n')
                    rf_label_foldtrain.write(str(int(label)) + '\n')

for i in range(fold_num):
    sub_list =  fold_subs_ids[i]
    with open('Loading_file_new/samm_5fold/samm_fold{}_test_data.txt'.format(i), 'w') as rf_data_foldtest:
        with open('Loading_file_new/samm_5fold/samm_fold{}_test_label.txt'.format(i), 'w') as rf_label_foldtest:
            for sub_id in sub_list:
                sub_samples = subject_sample_dic[sub_id]
                for sample in sub_samples:
                    onset_path = sample[0]
                    apex_path = sample[1]
                    label = sample[2]
                    rf_data_foldtest.write(onset_path + '\n')
                    rf_data_foldtest.write(apex_path + '\n')
                    rf_label_foldtest.write(str(int(label)) + '\n')







