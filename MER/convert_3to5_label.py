import os
import random

expression_list = ['Happiness', 'Sadness', 'Surprise', 'Fear', 'Anger', 'Disgust', 'Contempt']
target_expresion_list = ['Anger', 'Fear', 'Disgust']

mapping = {'Anger': 0,'Happiness': 1, 'Surprise':2, 'Fear':3, 'Disgust':4}

source_file = 'Loading_file_new/5w_real+ck+prior_fold/simulate_fold2_data.txt'

target_file = 'Loading_file_new/5w_real+ck+prior_fold/simulate_fold2_5classSAMM.txt'

count = 0
with open (target_file, 'w') as tf:
    with open( source_file , "r") as rf:
        lines = rf.readlines()
        for l in lines:
            count+=1
            if count%1000==0:
                print (count)
            l = l.strip()
            info = l.split(' ')[0].strip()
            sample_name = info.split('/')[-1]
            if 'generated_multi_faces_5w' in info:
                if int(float(l.split(' ')[1])) == 1 or int(float(l.split(' ')[1])) == 2:
                    if random.randint(0,10)<5:
                        continue
                    else:
                        tf.write(info + ' ' + str(l.split(' ')[1]) + '\n')

                else:
                    for type in target_expresion_list:
                        if type in sample_name:
                            tf.write(info + ' ' + str(mapping[type]) + '\n')
                            break
                        else:
                            continue
            # else:
            #     if int(float(l.split(' ')[1]))==1 or int(float(l.split(' ')[1]))==2:
            #         tf.write(info + ' ' + str(l.split(' ')[1]) + '\n')
            #     else:
            #         continue






