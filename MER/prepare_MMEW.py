import os
import glob
import xlrd

mapping = {'happiness':0, 'surprise':1, 'anger':2, 'disgust':3, 'fear':4, 'sadness':5}

MMEW_dir_path = 'Data/MMEW_Final/Micro_Expression'
MMEW_csv_path = 'Data/MMEW_Micro_Exp.xlsx'

target_data_file = 'Loading_file_new/MMEW_5fold/MMEW_1fold_data_6class.txt'
target_label_file = 'Loading_file_new/MMEW_5fold/MMEW_1fold_label_6class.txt'

subject_sample_dic = {}

count = 0

fold_num = 5

data = xlrd.open_workbook(MMEW_csv_path)
data= data.sheet_by_index(0)
nrows = data.nrows
with open (MMEW_csv_path, 'r') as rf:
    with open(target_data_file, "w") as tf_data:
        with open(target_label_file, "w") as tf_label:
            for i in range(1, nrows):
                info = data.row_values(i)
                sub_id = str(int(info[0]))
                expr_type = info[6]
                sample_name = info[1]
                onset_frame = info[2]
                apex_frame = info[3]
                if sub_id not in subject_sample_dic.keys():
                    subject_sample_dic[sub_id] = []
                if expr_type in mapping.keys():
                    sample_dir = os.path.join(MMEW_dir_path, expr_type)
                    onset_path = os.path.join(sample_dir, sample_name, str(int(onset_frame))+'.jpg')
                    apex_path = os.path.join(sample_dir, sample_name, str(int(apex_frame))+'.jpg')
                    print (onset_path)
                    print(apex_path)
                    assert (os.path.exists(onset_path))
                    assert (os.path.exists(apex_path))
                    label = mapping[expr_type]
                    sample = [onset_path, apex_path, label]
                    subject_sample_dic[sub_id].append(sample)
                    tf_data.write(onset_path +'\n')
                    tf_data.write(apex_path +'\n')
                    tf_label.write(str(int(label))+'\n')
                    print(label)
fold_subs_ids = []
fold0_sub = ['1', '2', '3', '4', '5', '6']
fold1_sub = ['7', '8', '9', '10', '11', '12']
fold2_sub = ['13', '14', '15', '16', '17', '18']
fold3_sub = ['19', '20', '21', '22', '23', '24']
fold4_sub = ['25', '26', '27', '28', '29', '30']
fold_subs_ids.append(fold0_sub)
fold_subs_ids.append(fold1_sub)
fold_subs_ids.append(fold2_sub)
fold_subs_ids.append(fold3_sub)
fold_subs_ids.append(fold4_sub)

all_sub = [str(i) for i in range(1, 31)]

for i in range(fold_num):
    excluede_sub_list = fold_subs_ids[i]
    sub_list = [str(id) for id in range(1 ,30) if str(id) not in excluede_sub_list]
    with open('Loading_file_new/MMEW_5fold/MMEW_fold{}_train_data.txt'.format(i), 'w') as rf_data_foldtrain:
        with open('Loading_file_new/MMEW_5fold/MMEW_fold{}_train_label.txt'.format(i), 'w') as rf_label_foldtrain:
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
    with open('Loading_file_new/MMEW_5fold/MMEW_fold{}_test_data.txt'.format(i), 'w') as rf_data_foldtest:
        with open('Loading_file_new/MMEW_5fold/MMEW_fold{}_test_label.txt'.format(i), 'w') as rf_label_foldtest:
            for sub_id in sub_list:
                sub_samples = subject_sample_dic[sub_id]
                for sample in sub_samples:
                    onset_path = sample[0]
                    apex_path = sample[1]
                    label = sample[2]
                    rf_data_foldtest.write(onset_path + '\n')
                    rf_data_foldtest.write(apex_path + '\n')
                    rf_label_foldtest.write(str(int(label)) + '\n')






