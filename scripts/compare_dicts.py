import pickle

with open('/home/cmkstien/Desktop/RT_data/col_metadata.pickle', 'rb') as handle: ## used for global node hash encodings
    columndict2 = pickle.load(handle) 
with open('/home/cmkstien/Desktop/RT_data/col_metadataJuly30.pickle', 'rb') as handle: ## used for global node hash encodings
    columndict1 = pickle.load(handle) 

keys = list(columndict1.keys())


for i in keys:
    if columndict1[i] != columndict2[i]:
        print(i)
        print(columndict2[i][11:19])
        print(columndict1[i][11:19])
    else:
        print(i, 'same')