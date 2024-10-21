import numpy as np
import os
import matplotlib.pyplot as plt


def plot_grads(grad, rts, t_crit, dir, filter, loc):
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
    if filter:
        save_path = sub1 + dir + '_filtered.png'
    else:
        save_path = sub2 + dir + '.png'
    plt.savefig(save_path)
    plt.clf()


def get_inflections(pB, times):
    max_ind = np.argmax(pB) ## this is used to stop before the max value (basically ignore points after the gradient resets)
    slopes = []
    inflections = []
    # print(max_ind)
    for i in range(1, len(pB)):
        if pB[i] != pB[i-1]:
            if i != max_ind:
                inflections.append((times[i-1], pB[i-1]))
                inflections.append((times[i], pB[i]))
                inflections = list(set(inflections))
    # print(pB, times)
    # print(inflections)
    return inflections

def remove_rts(directory, rts):
    path1 = r'/home/cmkstien/Desktop/RT_data/RepoRT/processed_data/' + directory + '/' + directory + '_rtdata_canonical_success.tsv'
    path2 = r'/home/cmkstien/Desktop/RT_data/RepoRT/processed_data/'+ directory + '/' + directory +'_rtdata_isomeric_success.tsv'
    path3 = r'/home/cmkstien/Desktop/RT_data/RepoRT/processed_data/' + directory + '/' + directory + '_rtdata_canonical_failed.tsv'
    path4 = r'/home/cmkstien/Desktop/RT_data/RepoRT/processed_data/' + directory + '/' + directory + '_rtdata_isomeric_failed.tsv'

    data_paths = [path1, path2, path3, path4]

    for data_path in data_paths:
        try:
            data = open(data_path, 'r')
        except Exception as e:
            print(e)
            continue
        lines = data.readlines()

        # Find rows with values in rts and delete them
        new_lines = []
        for line in lines:
            values = line.split('\t')
            
            if values[3] not in rts:
                new_lines.append(line)
            else:
                print('Removed RT:', values[3], data_path)

        # Write the modified lines back to the file

        with open(data_path, 'w') as file:
            file.writelines(new_lines)

        data = open(data_path, 'r')
        lines = data.readlines()
        print('Successfully removed RTs from', data_path)




def gen_filter_list(path):
    not_uniform_fl = 0
    uniform_fl = 0
    no_grad = 0
    four_grad = 0
    total_rts= 0

    t_max = []
    col_lengthL = []
    part_sizeL = []
    HILIC_list = []
    dead_volL = []
    i = 0
    filter_list = []
    print('hullo')

    filter_list = ['0392', '0008', '0028', '0080', '0155'] ## starting filter_list 
    print(filter_list)
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
            # print(col_string)

            usp_code = meta_data[1, 2]

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
            if 'HILIC' in column_name or 'Amide' in column_name:
                print(dir, column_name)
                x = 1
            else:
                filter_list.append(dir)
                # print('NOT HILIC', dir)
                continue
            t_no = dead[0]
            dead = np.repeat(dead, n_rts)
            dead_volL.extend(dead)
            grad_data = np.loadtxt(grad_path, delimiter='\t', dtype='str')

            try:
                grad = grad_data[1:, :]
                grad = np.asarray(grad, dtype=np.float32)
            except:
                no_grad += n_rts ## checking if gradient breaks
                filter_list.append(dir)
                print('dir, NOGRAD')
                continue

            t = grad[:,0]

            pA = np.asarray(grad[:, 1], dtype=float)
            pB = np.asarray(grad[:, 2], dtype=float)

            if pB[0] < 50:
                A_mobile = meta_data[:, 9:18]
                A_index = np.argmax(np.asarray(A_mobile[1:,:], dtype=float))
                A_add = meta_data[:, 18:48]
                A_pH = meta_data[:, 48]
                A_start = meta_data[:, 169]
                A_end = meta_data[:, 173]

                B_mobile = meta_data[:, 49:58]
                B_index = np.argmax(np.asarray(B_mobile[1:,:], dtype=float))
                print(A_mobile[:,A_index],B_mobile[:,B_index], dir)

                # print(B_mobile[:,B_index])
                B_add = meta_data[:, 58:88]
                B_pH = meta_data[:, 88]
                B_start = meta_data[:, 170]
                B_end = meta_data[:, 174]
                switched = False
            elif pB[0] > 50:
                x = ':)'
                grad[:, [1, 2]] = grad[:, [2, 1]]
                B_mobile = meta_data[:, 9:18]
                B_add = meta_data[:, 18:48]
                B_pH = meta_data[:, 48]
                B_start = meta_data[:, 169]
                B_end = meta_data[:, 173]

                A_mobile = meta_data[:, 49:58]
                A_add = meta_data[:, 58:88]
                A_pH = meta_data[:, 88]
                A_start = meta_data[:, 170]
                A_end = meta_data[:, 174]

                B_index = np.argmax(np.asarray(B_mobile[1:,:], dtype=float))   
                B_solv = B_mobile[0,B_index]
                A_index = np.argmax(np.asarray(A_mobile[1:,:], dtype=float))
                A_solv = A_mobile[0,A_index]
                print(A_mobile[:,A_index],B_mobile[:,B_index], dir)


                pB = np.asarray(grad[:, 1], dtype=float)
                pA = np.asarray(grad[:, 2], dtype=float)
                switched = True
                # print(grad, "GRAD")
                # print('TEST')

            pB_max = np.max(pB)


            if t_no > 3:
                print(dir, 'DEAD TIME TOO LARGE')
                filter_list.append(dir)
                continue

            
            total_rts += n_rts

            info_data = np.loadtxt(info_path, delimiter='\t', dtype='str')
            HPLC_type = info_data[1,2]

            fl = grad[:,-1]
            times = grad[:,0]
            # print(grad)
            test = True

            index = 0

            try:
                B_ind = np.argmax(np.asarray(B_mobile[1:,:], dtype=float))   
                B_solv = B_mobile[0,B_ind]
                A_ind = np.argmax(np.asarray(A_mobile[1:,:20], dtype=float))
                A_solv = A_mobile[0,A_ind]

            except:
                print('no gradient')
                filter_list.append(dir)
                continue
            inflections = get_inflections(pB, times) ## all the inflection points in the gradient
            # print(inflections)
            # print(inflections, dir)
            t_crit = inflections[1]
            loc = '/home/cmkstien/Desktop/RT_data/filtered_July31'
            # if dir == '0186':
            #     print(grad)
            #     print(inflections)
            #     exit()
            # else:
            #     continue
            ### NEED TO MAP ALL THE A to B onto B to A
            ### Make sure it's going from organic - aqueous

            if len(inflections) <= 3:
                plot_grads(grad, rts, t_crit[0], dir, False, loc)
            else:
                plot_grads(grad, rts, t_crit[0], dir, True, loc)
                # filter_list.append(dir)
                print(dir, 'inflections')

            if grad.shape[1] > 5:
                c = grad[:,-3]
                d = grad[:,-2]
            else: 
                c= np.zeros_like(fl)
                d= np.zeros_like(fl)

            t_max.append(times[-1])

            if not np.all(c == 0) and not np.all(d == 0):
                four_grad += n_rts
                filter_list.append(dir)
                print(dir, 'CD')
                continue
            elif np.all(fl == fl[0]):
                uniform_fl += n_rts
                continue
            else: ## Cases with inconsistent flow rates
                prev = fl[0]
                # print(dir)

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
                        # print(rt, "ABOVE CRITICAL")
                    
                count = 0
                frac = len(rt_list) / len(rts)
                # if len(rt_list) > 0:
                #     print(dir)
                #     print(frac)
                #     for i in rts:
                #         if rt < 3:
                #             count +=1
                #     print(count)
                # if dir in ['0383', '0382', '0384', '0392']:
                #     print(frac, dir, t_crit, len(rt_list))
                #     print(grad, t_crit)
                #     plot_grads(grad, rts, dir, False, loc)
                #     exit()
                # if frac > 0.10 and len(rt_list) > 0:
                #     plot_grads(grad, rts, t_crit, dir, True, loc)
                #     filter_list.append(dir)
                #     # # remove_rts(dir, rt_list)
                #     # threshold_list.append(dir)
                #     # print(dir, frac)                            

                # else:
                #     plot_grads(grad, rts, t_crit, dir, False, loc)
        
                    # if dir not in filter_list:
                    #     filter_list.append(dir)
                        # Add the following code to break out of all loops

                not_uniform_fl += n_rts

                continue
                # print(file, fl, grad)

        with open(r'/home/cmkstien/Desktop/RT_data/filtered_July31/HILIC_filt_list_july31.csv', 'w') as file:
            for item in list(sorted(set(filter_list))):
                # print(item)
                file.write("%s\n" % item)


dir = '/home/cmkstien/Desktop/rt_backup/RepoRT/processed_data/'

gen_filter_list(dir)
    # with open(r'/home/cmkstien/Desktop/RT_data/threshold_list', 'w') as file:
    #     for item in list(set(threshold_list)):
    #         print(item)
    #         file.write("%s\n" % item)
    # print(f"Total RTs: {total_rts}")
    # # print(uniform_fl, not_uniform_fl,  no_grad, four_grad)
    # print(f"Uniform flow rate: {uniform_fl}")
    # print(f"Not Uniform flow rate: {not_uniform_fl}")
    # print(f"No Grad: {no_grad}")
    # print(f"Four Grad: {four_grad}")
    # print('***********Chromatography Type***************')
    # print(f"RP: {rp}")
    # print(f"HILIC: {hilic}")
    # print(f"Normal: {normal}")
    # print(f"Amide: {amide}")
    # print(f"Other: {not_found}")
    # print('**********MOBILEPHASEs**************')
    # print("IGNORE THE A LABEL THESE DESCRIBE BOTH")
    # print(mobile_labels[0], int(mobile_phase_dist[0][0]),'\n')
    # print(mobile_labels[1], int(mobile_phase_dist[0][1]),'\n')
    # print(mobile_labels[2], int(mobile_phase_dist[0][2]),'\n')
    # print(mobile_labels[3], int(mobile_phase_dist[0][3]),'\n')
    # print(mobile_labels[4], int(mobile_phase_dist[0][4]),'\n')
    # print(mobile_labels[5], int(mobile_phase_dist[0][5]),'\n')
    # print(mobile_labels[6], int(mobile_phase_dist[0][6]),'\n')
    # print(mobile_labels[7], int(mobile_phase_dist[0][7]),'\n')

    # print(removed_rts, "RTs removed if IPrOH or ACETONE") ## roughly 4k rts removed
    # print("***************Column Parameters*******************")

    # exit(50