import numpy as np


externals = ['0029', '0127', '0275']

## 0029 - H2O to ACN, 47 points (2 inflection points) Waters ACQUITY UPLC BEH Shield RP18
## 0275 - H2O to MEOH 75 points (1 inflection point) Phenomenex Kinetex PS C18
## 0127 - H2O to ACN 93 points, (1 inflection point) Merck Supelco Ascentis Express C18


with open('/home/cmkstien/Desktop/RT_data/rt_data_July31_METLINFILTERONLY.csv', 'r') as f:
    data = f.readlines()
    print(data[1])
    # Create two output files
    external_file = '/home/cmkstien/Desktop/RT_data/Dataset_TrueExternalJuly31/True_external_methods.csv'
    other_file = '/home/cmkstien/Desktop/RT_data/Dataset_TrueExternalJuly31/FullDataset_July31_minux_ex.csv'

    # Iterate through the file
    external_methods = []
    other_methods = []

    for line in data:
        method = line.split(',')[0].strip()

        if method in externals:
            external_methods.append(line)
        else:
            other_methods.append(line)

    # Write external methods to external file
    with open(external_file, 'w') as f:
        f.writelines(external_methods)

    # Write other methods to other file
    with open(other_file, 'w') as f:
        f.writelines(other_methods)

    # Print the first line of each file
    with open(external_file, 'r') as f:
        print(f.readline())

    with open(other_file, 'r') as f:
        print(f.readline())
    exit()