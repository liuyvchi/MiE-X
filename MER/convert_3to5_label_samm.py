import os

expression_list = ['Happiness', 'Sadness', 'Surprise', 'Fear', 'Anger', 'Disgust']

target_expresion_list = ['Anger', 'Fear', 'Disgust']

mapping = {'Anger': 0,'Happiness': 1, 'Surprise':2, 'Fear':3, 'Disgust':4}

source_data_file = 'Loading_file_new/samm_5fold/samm_fold1_data1.txt'

target_data_file = 'Loading_file_new/samm_5fold/samm_fold1_data1_5class.txt'
target_label_file = 'Loading_file_new/samm_5fold/samm_fold1_label1_5class.txt'

sample_label_dic = {}
with open('Data/SAMM_label.csv', 'r') as rf:
    for line in rf:
        line = line.strip('\n')
        info = line.split(',')
        sample_name = info[1]
        exp_type = info[-2]
        if exp_type in mapping.keys():
            label = mapping[exp_type]
            sample_label_dic[sample_name] = label
            print(sample_name)


count = 0

with open (source_data_file, 'r') as rf:
    lines = rf.readlines()
    num_sample = int(len(lines)/2)

    with open(target_data_file, "w") as tf_data:
        with open(target_label_file, "w") as tf_label:
            for i in range(num_sample):
                line = lines[i*2]
                line = line.strip('\n')
                info = line.split('/')
                sample_name = info[-2]

                if sample_name in sample_label_dic.keys():
                    label = sample_label_dic[sample_name]
                    tf_data.write(lines[i*2])
                    tf_data.write(lines[i*2+1])
                    tf_label.write(str(label)+'\n')







