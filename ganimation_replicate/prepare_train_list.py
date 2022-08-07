import os


data_dir = '/home/yuchi/micro-expression/ganimation_replicate/generated5k_mmewAU+ouluAU'

g = os.walk(data_dir)
training_data_file_dir = '/home/yuchi/micro-expression/MER/Loading_file_new/generated5k_mmewAU+ouluAU'
if not os.path.exists(training_data_file_dir):
    os.mkdir(training_data_file_dir)
training_data_file_list = os.path.join(training_data_file_dir, 'simulate_fold0_data.txt')
with open (training_data_file_list, 'w') as wf:
    for path,dir_list,file_list in g:
        for dir_name in dir_list:
            if dir_name.split('_')[-1] in ['0', '1', '2']:
                emotion_label =  dir_name.split('_')[-1]
                sample_path = os.path.join(data_dir, dir_name)
                wf.writelines(sample_path+' '+emotion_label+'\n')
