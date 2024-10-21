import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

path = r'/home/cmkstien/Desktop/rt_backup/RepoRT/processed_data'

def get_inflections(pB, times):
    max_ind = np.argmax(pB) ## this is used to stop before the max value (basically ignore points after the gradient resets)
    slopes = []
    inflections = []
    for i in range(1, len(pB)):
        slope = (pB[i]-pB[i-1])/(times[i]-times[i-1])
        slopes.append(slope)
        if pB[i] != pB[i-1]:
            if i <= max_ind:
                inflections.append((times[i-1], pB[i-1]))
                inflections.append((times[i], pB[i]))
                inflections = list(set(inflections))
        
    return inflections



def plot_grads(grad, rts, t_crit, dir, filter, loc, inflections):
    sub1 =  loc + '/filtered/'
    sub2 = loc + '/not/'

    if not os.path.exists(sub1):
        os.makedirs(sub1)
    if not os.path.exists(sub2):
        os.makedirs(sub2)
    t = grad[:,0]
    B = grad[:,2]
    fl = grad[:,-1]
    prev = fl[0]
    c=0

    plt.plot(t, B, '-', label=dir)
    plt.plot([t_crit, t_crit], [0, 100], 'b--')
    plt.scatter(rts,  np.ones_like(rts) * 50, color='red', s=3, alpha=0.7)
    plt.xlabel('Time (min)')
    plt.ylabel('B (%)')
    plt.title('n =' + str(len(rts)))
    inflections_x = [coord[0] for coord in inflections]
    inflections_y = [coord[1] for coord in inflections]
    plt.scatter(inflections_x, inflections_y, color='green', marker='x')
    if filter:
        save_path = sub1 + dir + '_filtered.png'
    else:
        save_path = sub2 + dir + '.png'
    plt.savefig(save_path)
    plt.clf()
    # exit()


with open(r'/home/cmkstien/Desktop/RT_data/filtered_July31/filtered_list_july31.csv', 'r') as file:
    filter_list = file.readlines()
    filter_list = [x.strip() for x in filter_list]
    # print(filter_list)
t_rt = 0

data = []

params_header =['company_name', 'usp_code', 'col_length', 'col_innerdiam', 'col_part_size', 'temp', 'col_fl', 'col_dead', 'HPLC_type','A_solv', 'B_solv', 'time1', 'grad1', 'time2', 'grad2', 'time3', 'grad3', 'time4', 'grad4',  \
            'A_pH', 'B_pH', 'A_start', 'A_end', 'B_start', 'B_end', \
            'eluent.A.formic', 'eluent.A.formic.unit', 'eluent.A.acetic', 'eluent.A.acetic.unit','eluent.A.trifluoroacetic', 'eluent.A.trifluoroacetic.unit','eluent.A.phosphor', 'eluent.A.phosphor.unit','eluent.A.nh4ac','eluent.A.nh4ac.unit', \
            'eluent.A.nh4form','eluent.A.nh4form.unit','eluent.A.nh4carb', \
            'eluent.A.nh4carb.unit','eluent.A.nh4bicarb','eluent.A.nh4bicarb.unit', \
            'eluent.A.nh4f','eluent.A.nh4f.unit','eluent.A.nh4oh', \
            'eluent.A.nh4oh.unit','eluent.A.trieth','eluent.A.trieth.unit', \
            'eluent.A.triprop','eluent.A.triprop.unit','eluent.A.tribut', \
            'eluent.A.tribut.unit','eluent.A.nndimethylhex', \
            'eluent.A.nndimethylhex.unit','eluent.A.medronic', \
            'eluent.A.medronic.unit',\
            
            'eluent.B.formic', 'eluent.B.formic.unit', 'eluent.B.acetic', 'eluent.B.acetic.unit','eluent.B.trifluoroacetic', 'eluent.B.trifluoroacetic.unit','eluent.B.phosphor', 'eluent.B.phosphor.unit','eluent.B.nh4ac','eluent.B.nh4ac.unit', \
            'eluent.B.nh4form','eluent.B.nh4form.unit','eluent.B.nh4carb', \
            'eluent.B.nh4carb.unit','eluent.B.nh4bicarb','eluent.B.nh4bicarb.unit', \
            'eluent.B.nh4f','eluent.B.nh4f.unit','eluent.B.nh4oh', \
            'eluent.B.nh4oh.unit','eluent.B.trieth','eluent.B.trieth.unit', \
            'eluent.B.triprop','eluent.B.triprop.unit','eluent.B.tribut', \
            'eluent.B.tribut.unit','eluent.B.nndimethylhex', \
            'eluent.B.nndimethylhex.unit','eluent.B.medronic', \
            'eluent.B.medronic.unit', \
            'kPB', 'αCH2', 'αT/O', 'αC/P', 'αB/P', 'αB/P.1', 'particle size', 'pore size', \
            'H', 'S*', 'A', 'B', 'C (pH 2.8)', 'C (pH 7.0)', 'EB retention factor']
        
header = ['dir','name', 'smiles', 'rt']
col_dict = {}
col_dict['1'] = params_header
t= 0

with open('/home/cmkstien/Desktop/RT_data/rt_data_July31.csv', 'w') as file:
    file.write(','.join(header) + '\n')
file.close()

tanaka_path = r'/home/cmkstien/Desktop/rt_backup/RepoRT/resources/tanaka_database/tanaka_database.tsv'
tanaka_data = np.loadtxt(tanaka_path, delimiter='\t', dtype='str')

hsmb_path = r'/home/cmkstien/Desktop/rt_backup/RepoRT/resources/hsm_database/hsm_database.tsv'
hsmb_data = np.loadtxt(hsmb_path, delimiter='\t', dtype='str')

hsmb_dict = {}
# print(hsmb_data[0][:])
for i in hsmb_data[1:]:
    hsmb_dict[i[0]] = i[5:-2]

tanaka_dict = {}
for i in tanaka_data[1:]:
    tanaka_dict[i[0]] = i[3:-1]
n_rm = 0
removedRT = []
n_Rt_w = 0

for root, dirs, files in os.walk(path):
    for dir in dirs:
        if dir in filter_list:
            continue
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
        rts = np.asarray(rt_data[1:,3], dtype=float)

        meta_data = np.loadtxt(meta_path, delimiter='\t', dtype='str')
        column_name = meta_data[1, 1]
        
        col_string = column_name.split(' ')

        company_name = col_string[0]
        col_type = col_string[-2:]

        usp_code = meta_data[1, 2]

        col_length = meta_data[1, 3]
        if col_length == '':
            col_length = 0
        
        col_innerdiam = meta_data[1, 4] ## this is valuable

        col_part_size = meta_data[1, 5]
        if col_part_size == '':
            col_part_size = 0


        temp = meta_data[1, 6]
        col_fl = meta_data[1, 7]
        col_dead = meta_data[1, 8]
        dead = [float(col_dead)]

        t_no = dead[0]

        t_rt += n_rts
        # print(column_name)

        if 'HILIC' in column_name or 'Amide' in column_name:
            print("HILIC BROKE MY SHIT")
            continue
        
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
        switched = False

        B_ind = np.argmax(np.asarray(B_mobile[1:,:], dtype=float))   
        B_solv = B_mobile[0,B_ind]
        A_ind = np.argmax(np.asarray(A_mobile[1:,:20], dtype=float))
        A_solv = A_mobile[0,A_ind]

        info_data = np.loadtxt(info_path, delimiter='\t', dtype='str')
        HPLC_type = info_data[1,2]

        grad_data = np.loadtxt(grad_path, delimiter='\t', dtype='str')

        grad = grad_data[1:, :]
        grad = np.asarray(grad, dtype=np.float32)

        pA = np.asarray(grad[:, 1], dtype=float)
        pB = np.asarray(grad[:, 2], dtype=float)

        fl = grad[:,-1]
        times = grad[:,0]

        inflections = get_inflections(pB, times)
        loc = '/home/cmkstien/Desktop/RT_data/filtered_July31/'
        t_crit = 0
        if not np.all(fl == fl[0]):

            prev = fl[0]

            c= 0
            rt_list = []
            for i in fl:
                # print(i, prev)
                if i != prev:
                    filter_list.append(dir)
                    t_crit = grad[c,0]
                    break

                c+=1

            for rt in rts:
                if rt > t_crit:
                    rt_list.append(str(rt))

            frac = len(rt_list)/n_rts
            if len(rt_list) > 0 and frac < 0.10:
                print(rt_list)
                print(fl)
                exit()
            else:
                print('SOMETHING WENT WRONG')
                exit()
        if t_crit == 0: 
            t_crit = inflections[-1][0]
        plot_grads(grad, rts, t_crit, dir, False, loc, inflections)
        grad_l = []
        grad_l.extend([str(times[0]), str(pB[0])])
        c = 1
        count = 0
        t_prev = -1
        inflections.sort(key=lambda x: x[0])
        t_pB_max = inflections[-1][0]
        t_filter =  t_no + (0.01 * t_pB_max) ##  ## this is the trehold for compounds that are not retained on teh column

        while count < len(inflections):
            t = inflections[count][0]
            pB = inflections[count][1]

            if abs(t - t_prev) < 0.3: ## removing things that have very similar step sizes 
                # print(t, t_prev)
                removed = inflections.pop(count)
                # print(dir, 'REMOVED')
                # print(removed)
                # exit()

                continue
            t_prev = t
            count+=1


        for infl in inflections:
            grad_l.extend([str(infl[0]), str(infl[1])])
            c+=1

        pad = np.zeros((8-len(grad_l)))
        pad = np.asarray(pad, dtype=str)
        grad_l.extend(pad)
        assert grad_l[0] == '0.0'
        assert len(grad_l) == 8
        if temp == '':
            temp = '0'
        col_fl = fl[0]

        column_params = [company_name, usp_code, str(col_length), str(col_innerdiam), str(col_part_size), temp, col_fl, col_dead, HPLC_type, A_solv.split('.')[-1], B_solv.split('.')[-1]]
        column_params.extend(grad_l)
        if len(A_pH) > 1:
            A_pH = A_pH[1]
        if len(B_pH) > 1:
            B_pH = B_pH[1]
        column_params.extend([A_pH, B_pH])

        while len(A_start) > 1:
            A_start = A_start[1]
        while len(B_end) > 1:
            B_end = B_end[1]
        while len(A_end) > 1:
            A_end = A_end[1]
        while len(B_start) > 1:
            B_start = B_start[1]

        try:
            tanaka_params = tanaka_dict[column_name]
            # print(tanaka_dict[column_name], 'TANAKA')
        except:
            # t+= n_rts
            tanaka_params = np.zeros((8))
            # print(column_name, 'NOT IN TANAKA')
        try:
            
            hsmb_params = hsmb_dict[column_name]
            # print(hsmb_dict[column_name], 'HSMB')
        except:
            hsmb_params = np.zeros((7))
            
        column_params.extend([A_start, A_end, B_start, B_end])
        column_params.extend(A_add[1])
        column_params.extend(B_add[1])
        column_params.extend(tanaka_params)
        column_params.extend(hsmb_params)
        assert len(column_params) ==  len(params_header)
        col_dict[dir] = column_params

        pre_data = [dir]
        pre_data.extend(grad_l)

        with open('/home/cmkstien/Desktop/RT_data/rt_data_July31_METLINFILTERONLY.csv', 'a') as file:
            for i in rt_data[1:,:7]:
                to_write = [dir]
                sm = i[4]
                rt = float(i[3])
                if dir == '0186':
                    if rt < (200 / 60): ## 200 second threshold in line with ospienko
                        n_rm += 1
                        continue ## don't include values that are not retained on teh columns

                # if rt < t_filter: ## UNCOMMENT HTIS if you wanna filter based on t0
                    
                #     # print(rt, t_thresh, 'REMOVED')
                #     removedRT.append(dir)
                #     n_rm +=1

                #     continue

                rt = str(i[3])
                name = i[1]
                to_write.extend([sm, rt])
                # print(to_write)
                try:
                    file.write(','.join(to_write) + '\n')
                    print(n_Rt_w)
                    n_Rt_w += 1
                except Exception as e:
                    print(to_write)
                    print(e)
                    
                    exit()
            file.close()

    # Pickle the column metadata dictionary
    print(n_rm, 'REMOVED DUE TO RT FILTERING FOR NOT RETAINED')
    with open('/home/cmkstien/Desktop/RT_data/col_metadataJuly31.pickle', 'wb') as file:
        pickle.dump(col_dict, file)
        print('pickled')
        break
    file.close()

unique_values, counts = np.unique(np.asarray(removedRT), return_counts=True)
print(unique_values, counts)

for value, count in zip(unique_values, counts):
    print(f"{value}: {count}")










