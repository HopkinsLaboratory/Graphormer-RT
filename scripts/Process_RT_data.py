import numpy as np
import os
import matplotlib.pyplot as plt

path = r'/home/cmkstien/Desktop/RT_data/RepoRT/processed_data'
not_uniform_fl = 0
uniform_fl = 0
no_grad = 0
four_grad = 0
total_rts= 0
rp = 0
hilic = 0
normal = 0 
not_found = 0
j = 0
removed_rts = 0
UPLC = 0
waters = 0
merck = 0
restek = 0
phenomenex = 0
hilicon=0
thermo = 0
agilent = 0
amide = 0
t_max = []
col_lengthL = []
part_sizeL = []
dead_volL = []
total_rts = 0
mobile_phase_dist = np.zeros((1,9))
for root, dirs, files in os.walk(path):
    for dir in sorted(dirs):
        grad_n = dir + "_gradient.tsv"
        grad_path = os.path.join(path, dir, grad_n)

        rtdata_n = dir + "_rtdata_canonical_success.tsv"
        rtdata_path = os.path.join(path, dir, rtdata_n)

        meta_n = dir + "_metadata.tsv"        
        meta_path = os.path.join(path, dir, meta_n)
        info_n = dir + "_info.tsv"
        info_path = os.path.join(path, dir, info_n)

        try:
            meta_data = np.loadtxt(meta_path, delimiter='\t', dtype='str')
        except:
            with open(meta_path, 'r') as file:
                lines = file.readlines()
            # print(meta_path)
            with open(meta_path, 'w') as file:
                for line in lines:
                    modified_line = line.replace('/%', 'percent')
                    file.write(modified_line)

        try:
            rt_data = np.loadtxt(rtdata_path, delimiter='\t', dtype='str')
        except:
            with open(rtdata_path, 'r') as file:
                lines = file.readlines()
                
            with open(rtdata_path, 'w') as file:
                for line in lines:
                    modified_line = line.replace('#', 'Q')
                    file.write(modified_line)
        
        rt_data = np.loadtxt(rtdata_path, delimiter='\t', dtype='str', usecols=np.arange(0, 12))
        n_rts = rt_data.shape[0]
        if dir == '0186':
            continue
        total_rts += n_rts
        print(dir,n_rts,  total_rts)
        continue

        meta_data = np.loadtxt(meta_path, delimiter='\t', dtype='str')
        column_name = meta_data[1, 1]
        
        col_string = column_name.split(' ')

        company_name = col_string[0]
        col_type = col_string[-2:]
        # print(col_string)
        if 'UPLC' in col_string:
            UPLC += n_rts
        if 'Amide' in col_string:
            amide += n_rts
        if 'Waters' in col_string:
            waters += n_rts
        elif 'Merck' in col_string:
            merck += n_rts
        elif 'Restek' in col_string:
            restek += n_rts
        elif 'Phenomenex' in col_string:
            phenomenex += n_rts
        elif 'HILICON' in col_string:
            hilicon += n_rts
        elif 'Thermo' in col_string:
            thermo += n_rts
        elif 'Agilent' in col_string:
            agilent += n_rts
        
        usp_code = meta_data[1, 2]
        print(usp_code, col_string)

        col_length = meta_data[1, 3]
        if col_length == '':
            col_length = 0
        col_lengthL.append(float(col_length))
        
        col_innerdiam = meta_data[1, 4] ## this is valuable

        col_part_size = meta_data[1, 5]
        if col_part_size == '':
            col_part_size = 0
        part_sizeL.append(float(col_part_size))
        temp = meta_data[1, 6]
        col_fl = meta_data[1, 7]
        col_dead = meta_data[1, 8]
        dead = [float(col_dead)]

        # if float(col_dead) > 5:
        #     print(dir, col_dead)

        dead = np.repeat(dead, n_rts)
        dead_volL.extend(dead)
        A_mobile = meta_data[:, 9:18]
        A_add = meta_data[:, 18:48]
        A_pH = meta_data[:, 48]
        A_start = meta_data[:, 169]
        A_end = meta_data[:, 173]

        B_mobile = meta_data[:, 49:58]
        B_add = meta_data[:, 58:88]
        B_pH = meta_data[:, 88]
        B_start = meta_data[:, 170]
        B_end = meta_data[:, 174]

        # if A_mobile[1,3] != '0' or B_mobile[1,3] != '0':
        #     removed_rts += n_rts
        #     continue
        # elif A_mobile[1,4] != '0' or B_mobile[1,4] != '0':
        #     removed_rts += n_rts
        #     continue
        
        total_rts += n_rts

        # C_mobile = meta_data[:, 89:98]
        # C_add = meta_data[:, 98:128]
        # C_pH = meta_data[:, 129]
        # C_start = meta_data[:, 171]
        # C_end = meta_data[:, 175]

        # D_mobile = meta_data[:, 129:138]
        # D_add = meta_data[:, 138:168]
        # D_pH = meta_data[:, 168]
        # D_start = meta_data[:, 172]
        # D_end = meta_data[:, 176]

        mobile_labels = A_mobile[0,:]
        try:
            A_mobile_bool = np.asarray(A_mobile[1,:], dtype=float)
        except:
            A_mobile = np.where(meta_data[:, 9:18] == '', '0', meta_data[:, 9:18])
            A_mobile_bool = np.asarray(A_mobile[1,:], dtype=float)
        A_mobile_bool[A_mobile_bool != 0] = n_rts

        try:
            B_mobile_bool = np.asarray(B_mobile[1,:], dtype=float)
        except:
            B_mobile = np.where(meta_data[:, 49:58] == '', '0', meta_data[:, 49:58])
            B_mobile_bool = np.asarray(B_mobile[1,:], dtype=float)
        B_mobile_bool[B_mobile_bool != 0] = n_rts

        mobile_phase_dist = mobile_phase_dist + A_mobile_bool + B_mobile_bool

        info_data = np.loadtxt(info_path, delimiter='\t', dtype='str')
        HPLC_type = info_data[1,2]
        grad_data = np.loadtxt(grad_path, delimiter='\t', dtype='str')

        if 'RP' in HPLC_type:
            rp += n_rts
        elif 'HILIC' in HPLC_type:
            hilic += n_rts
        elif 'Normal' in HPLC_type:
            normal += n_rts
        elif 'Amide' in HPLC_type:
            amide += n_rts
        else:
            not_found += n_rts
        try:
            grad = grad_data[1:, :]
            grad = np.asarray(grad, dtype=np.float32)
        except:
            no_grad += n_rts ## checking if gradient breaks
            continue

        fl = grad[:,-1]
        times = grad[:,0]
        # print(grad)

        if grad.shape[1] > 5:
            c = grad[:,-3]
            d = grad[:,-2]
        else: 
            c= np.zeros_like(fl)
            d= np.zeros_like(fl)

        t_max.append(times[-1])

        if not np.all(c == 0) and not np.all(d == 0):
            four_grad += n_rts
            # print(grad)
            # print(dir)
            continue
        elif np.all(fl == fl[0]):
            uniform_fl += n_rts
            continue
        else:
            not_uniform_fl += n_rts
            # print(grad)
            # print(dir)
            continue
            # print(file, fl, grad)
print(f"Total RTs: {total_rts}")
# print(uniform_fl, not_uniform_fl,  no_grad, four_grad)
print(f"Uniform flow rate: {uniform_fl}")
print(f"Not Uniform flow rate: {not_uniform_fl}")
print(f"No Grad: {no_grad}")
print(f"Four Grad: {four_grad}")
print('***********Chromatography Type***************')
print(f"RP: {rp}")
print(f"HILIC: {hilic}")
print(f"Normal: {normal}")
print(f"Amide: {amide}")
print(f"Other: {not_found}")
print('**********MOBILEPHASEs**************')
print("IGNORE THE A LABEL THESE DESCRIBE BOTH")
print(mobile_labels[0], int(mobile_phase_dist[0][0]),'\n')
print(mobile_labels[1], int(mobile_phase_dist[0][1]),'\n')
print(mobile_labels[2], int(mobile_phase_dist[0][2]),'\n')
print(mobile_labels[3], int(mobile_phase_dist[0][3]),'\n')
print(mobile_labels[4], int(mobile_phase_dist[0][4]),'\n')
print(mobile_labels[5], int(mobile_phase_dist[0][5]),'\n')
print(mobile_labels[6], int(mobile_phase_dist[0][6]),'\n')
print(mobile_labels[7], int(mobile_phase_dist[0][7]),'\n')

print(removed_rts, "RTs removed if IPrOH or ACETONE") ## roughly 4k rts removed
print("***************Column Parameters*******************")

print(UPLC, "UPLC RTs")
print(waters, "Waters RTs")
print(merck, "Merck RTs")
print(restek, "Restek RTs")
print(phenomenex, "Phenomenex RTs")
print(hilicon, "Hilicon RTs")
print(thermo, "Thermo RTs")
print(agilent, "Agilent RTs")


plt.hist(t_max, bins=40)
plt.savefig('/home/cmkstien/Desktop/RT_data/t_max_dist.png', dpi=300)
plt.clf()
plt.hist(col_lengthL)
plt.savefig('/home/cmkstien/Desktop/RT_data/col_length_dist.png', dpi=300)
plt.clf()
plt.hist(part_sizeL, bins=40)
plt.savefig('/home/cmkstien/Desktop/RT_data/col_part_size_dist.png', dpi=300)
plt.clf()
dead_volL = np.asarray(dead_volL)
# dead_volL = dead_volL[dead_volL < 20]
plt.hist(dead_volL, bins=100)
print(np.mean(dead_volL))
plt.savefig('/home/cmkstien/Desktop/RT_data/dead_vol_dist.png', dpi=300)
# exit(50