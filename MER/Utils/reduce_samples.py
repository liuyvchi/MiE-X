import os

choose_num = 30000

simulate_ck_data = '/home/user/Yuchi/iccv2019/MER/Loading_file_new/5w_ck/simulate_ck_train.txt'
simulate_prior_data = '/home/user/Yuchi/iccv2019/MER/Loading_file_new/5w_prior/simulate_prior_train.txt'
simulate_realAU_data = '/home/user/Yuchi/iccv2019/MER/Loading_file_new/5w_realAU/simulate_realAU_train.txt'

with open(simulate_ck_data, 'r') as d:
    data = d.readlines()
    print (data[100])