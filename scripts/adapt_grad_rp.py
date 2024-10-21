import numpy as np
import os
import pickle
import matplotlib.pyplot as plt


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


grad_path = '/home/cmkstien/Graphormer_RT/HuanLab/RP_gradient.tsv'
grad_data = np.loadtxt(grad_path, delimiter='\t', dtype='str')

grad = grad_data[1:, :]
grad = np.asarray(grad, dtype=np.float32)

pA = np.asarray(grad[:, 1], dtype=float)
pB = np.asarray(grad[:, 2], dtype=float)

fl = grad[:,-1]
times = grad[:,0]
t_no = 0.33

inflections = get_inflections(pB, times)

inflections = inflections[:-1]

loc = '/home/cmkstien/Desktop/RT_data/filtered_not_ret_Jun24/'
t_crit = 0



if t_crit == 0: 
    t_crit = inflections[-1][0]

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
# if temp == '':
#     temp = '0'
col_fl = fl[0]
print(grad_l)

column_params = ['Waters', 'L1', '100', '1.0', '1.7', '0', '0.15', '0.33', 'RP', 'h2o', 'acn']

column_params.extend(grad_l)
column_params.extend(['0','0' ]) #pH unknown

A_add = ['0.1', '%', '0', '', '0', '', '0', '', '0', '', '0', '', '0', '', '0', '', '0', '', '0', '', '0', '', '0', '', '0', '', '0', '', '0', '']
B_add = ['0.1', '%', '0', '', '0', '', '0', '', '0', '', '0', '', '0', '', '0', '', '0', '', '0', '', '0', '', '0', '', '0', '', '0', '', '0', '']
A_start = '100'
A_end = '5'
B_start = '0'
B_end = '95'
tanaka_params = np.zeros((8))
hsmb_params = np.zeros((7))
column_params.extend([A_start, A_end, B_start, B_end])
column_params.extend(A_add)
column_params.extend(B_add)
column_params.extend(tanaka_params)
column_params.extend(hsmb_params)


print(len(column_params))
method_dict = {}
method_dict['HUAN'] = column_params
with open('/home/cmkstien/Graphormer_RT/HuanLab/RP_huan_column_params.pkl', 'wb') as f:
    pickle.dump(method_dict, f)
print('success')

# print(grad_l)
