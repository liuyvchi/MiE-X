import os
import glob
import xlrd
import pandas as pd
import numpy as np
import csv

mapping = {'happiness':0, 'surprise':1, 'anger':2, 'disgust':3, 'fear':4, 'sadness':5}



AU_dir = 'Data/ck_AU_out'
train_path_file = 'Loading_file_new/macro_micro_3fold/Micro_fold0_data01.txt'
train_label_file = 'Loading_file_new/macro_micro_3fold/Micro_fold0_label01.txt'

positive_AU_csv_file = 'ck_surprise.csv'


## CASME
# with open (positive_AU_csv_file, 'w') as csvfile:
#     writer = csv.writer(csvfile)
#     with open (train_label_file, 'r') as rl:
#         labels = rl.readlines()
#         with open (train_path_file, 'r') as rp:
#             pathes = rp.readlines()
#             num_samples = len(labels)
#             assert num_samples == int(len(pathes)/2)
#             for i in range(num_samples):
#                 onse_path = pathes[i*2].strip()
#                 apex_path = pathes[i*2+1].strip()
#                 if 'CASME' in onse_path:
#                     sub_name = onse_path.split('/')[-3]
#                     sample_name = onse_path.split('/')[-2]
#                     assert (sample_name == apex_path.split('/')[-2])
#                     num_apx = int(apex_path.split('/')[-1].split('.')[0][3:])-int(onse_path.split('/')[-1].split('.')[0][3:])
#                     AU_path = os.path.join(AU_dir, sub_name, sample_name, sample_name+'.csv')
#                     frames_aus = pd.read_csv(AU_path, header=None)
#                     onset_aus = frames_aus.iloc[1][5:22].values.astype(np.float)
#                     apex_aus = frames_aus.iloc[1 + num_apx][5:22].values.astype(np.float)
#                     label = int(labels[i].strip())
#                     if label == 2:
#                         print(apex_aus)
#                         writer.writerow(apex_aus)

# ## CK #####
with open (positive_AU_csv_file, 'w') as csvfile:
    writer = csv.writer(csvfile)
    with open (train_label_file, 'r') as rl:
        labels = rl.readlines()
        with open (train_path_file, 'r') as rp:
            pathes = rp.readlines()
            num_samples = len(labels)
            assert num_samples == int(len(pathes)/2)
            for i in range(num_samples):
                onse_path = pathes[i*2].strip()
                apex_path = pathes[i*2+1].strip()
                if 'CK' in onse_path:
                    sub_name = onse_path.split('/')[-3]
                    sample_name = onse_path.split('/')[-2]
                    assert (sample_name == apex_path.split('/')[-2])

                    num_apx = int(apex_path.split('/')[-1].split('.')[0].split('_')[2])-int(onse_path.split('/')[-1].split('.')[0].split('_')[2])
                    print(num_apx)
                    AU_path = os.path.join(AU_dir, sub_name, sample_name, sample_name+'.csv')
                    try:
                        frames_aus = pd.read_csv(AU_path, header=None)
                    except:
                        continue
                    onset_aus = frames_aus.iloc[1][5:22].values.astype(np.float)
                    apex_aus = frames_aus.iloc[1 + num_apx][5:22].values.astype(np.float)
                    label = int(labels[i].strip())
                    if label == 2:
                        print(apex_aus)
                        writer.writerow(apex_aus)








