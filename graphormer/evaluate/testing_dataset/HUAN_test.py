
from re import L
import numpy as np 
import csv
from rdkit import Chem
import torch
import time

from .featurizing_helpers import *
print("YOURE DEF IN THE RCORRECT FILE")

import itertools

import dgl
import torch
import os

from graphormer.data import register_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import pickle
import gc

companies = ['', 'Waters', 'Thermo', 'Agilent', 'Restek', 'Merck', 'Phenomenex', 'HILICON', 'Other']
USPs = ['', 'L1', 'L10', 'L109', 'L11', 'L43', 'L68', 'L3','L114', 'L112', 'L122', 'Other']
solvs = ['h2o','meoh', 'acn', 'Other']
HPLC_type = ['RP', 'HILIC', 'Other']
lengths = ['0','50', '100','150', '200', '250', 'Other']

def one_hot_lengths(length):
    one_hot = [0] * len(lengths)
    one_hot[lengths.index(length)] = 1
    if length not in lengths:
        one_hot[-1] = 1
    return one_hot

def one_hot_HPLC_type(HPLC_type):
    one_hot = [0] * len(HPLC_type)
    one_hot[HPLC_type.index(HPLC_type)] = 1
    if HPLC_type not in HPLC_type:
        one_hot[-1] = 1
    return one_hot

def one_hot_company(company):
    one_hot = [0] * len(companies)
    one_hot[companies.index(company)] = 1
    if company not in companies:
        one_hot[-1] = 1
    return one_hot

def one_hot_USP(USP):
    one_hot = [0] * len(USPs)
    one_hot[USPs.index(USP)] = 1
    if USP not in USPs:
        one_hot[-1] = 1
    return one_hot

def one_hot_solvent(solvent):
    one_hot = [0] * len(solvs)
    one_hot[solvs.index(solvent)] = 1
    if solvent not in solvs:
        one_hot[-1] = 1
    return one_hot

def featurize_column(column_params, index):

    company = one_hot_company(column_params[0])
    USP = one_hot_USP(column_params[1])
    length = float(column_params[2]) / 250 ## consider mapping these into fixed bins for steps of 50 
    # length = one_hot_lengths(str(int(length)))
    # length = float(length)
    if column_params[3] == '':
        diameter = 0
    else:
        diameter = float(column_params[3]) ## normalizing diameter

    part_size = float(column_params[4])
    temp = float(column_params[5]) / 100 ## normalizing temperature (rethink this maybe)
    fl = float(column_params[6])  ## Double check that fl and col_fl are two different values
    dead = float(column_params[7]) ## dead time - MAYBE REMOVE COLUMN THAT HAS DEAD TIME OF ZERO
    # HPLC_type = one_hot_HPLC_type(column_params[8])

    solv_A = one_hot_solvent(column_params[9])
    solv_B = one_hot_solvent(column_params[10])
    # time_start_B = float(column_params[11]) ## this is always zero, so redundant
    start_B = float(column_params[12]) / 100
    t1 = float(column_params[13])   ## maybe consider normalizing the times as well to the fraction of the gradient ? TODO: 
    B1 = float(column_params[14]) / 100
    t2 = float(column_params[15]) 
    B2 = float(column_params[16]) / 100
    t3 = float(column_params[17]) 
    B3 = float(column_params[18]) / 100

    s1 = (B2 - B1) / (t2 - t1) 
    if t3 == 0 and B3 == 0: ## for cases where there is only 2 infleciton points
        s2 = 0
        s3 = 0
    else:
        s2 = (B3 - B2) / (t3 - t2)
        s3 = (B3 - B1) / (t3 - t1) ## add s3 if s1 and s2 help


    pH_A = float(column_params[19]) / 14 # normalize pH TODO: consider taking log

    if column_params[20] == '':
        pH_B = 0
    else:
        pH_B = float(column_params[20]) / 14  # normalize pH 

    add_A = column_params[25:55]
    add_B = column_params[55:84]
    tanaka_params = column_params[84:92]
    tanaka_params = [2.7 if param == '2.7 spp' else param for param in tanaka_params]
    tanaka_params = [2.7 if param == '2.6 spp' else param for param in tanaka_params]

    tanaka_params = [0 if param == '' else float(param) for param in tanaka_params] ## 

    hsmb_params = column_params[92:]
    hsmb_params = [0 if param == '' else float(param) for param in hsmb_params] ## TODO: add these in

    ##TODO: normalize tanaka params somehow
    ## COARSE NORMALIZATION BASED ON VALUES - coarse normalization didn't help
    kPB = tanaka_params[1] # / 10     
    a_CH2 = tanaka_params[2]# / 2
    a_TO = tanaka_params[3]# / 5
    a_CP = tanaka_params[4]
    a_BP = tanaka_params[5] #/ 2 
    a_BP1 = tanaka_params[6] #/ 2
    # particle_size = tanaka_params[7] #/ 5
    
    tanaka_params = [kPB, a_CH2, a_TO, a_CP, a_BP, a_BP1, part_size]
    # tanaka_params = [param +1 for param in tanaka_params]

    add_A_vals = np.ceil(list(map(float, add_A[::2])))
    add_B_vals = np.ceil(list(map(float, add_B[::2])))


    ## get rid of this to recover values. 
    add_A_vals = np.where(add_A_vals != 0, 1, add_A_vals)
    add_B_vals = np.where(add_B_vals != 0, 1, add_B_vals)


    add_A_units = add_A[1::2]
    add_B_units = add_B[1::2]
    # if index == '0186':
    #     print(len(add_A_vals), len(add_B_vals), len(company), len(USP), len(HPLC_type), len(solv_A), len(solv_B), len(add_A_vals), len(add_B_vals))
    # if index == '0126':
    #     print(len(add_A_vals), len(add_B_vals), len(company), len(USP), len(HPLC_type), len(solv_A), len(solv_B), len(add_A_vals), len(add_B_vals))
    #     exit()

    
    ############## Ablation study - remove all the column metadata
    # diameter = 0
    # part_size = 0
    # dead = 0
    # temp = 0
    # length = 0
    # company = np.zeros_like(company)
    # USP = np.zeros_like(USP)

    ############## Gradient Ablation
    # start_B = 0
    # t1 = 0
    # B1 = 0
    # t2 = 0
    # B2 = 0
    # t3 = 0
    # B3 = 0
    # fl = 0
    # ph_A = 0
    # ph_B = 0
    
    # solv_A = np.zeros_like(solv_A)
    # solv_B = np.zeros_like(solv_B)
    # add_A_vals = np.zeros_like(add_A_vals)
    # add_B_vals = np.zeros_like(add_B_vals)
    ###############################

    float_encodings = [diameter, part_size, start_B, t1, B1, t2, B2, t3, B3, pH_A, pH_B, dead, temp, fl, length] 

    float_encodings += tanaka_params
    float_encodings += hsmb_params

    int_encodings = np.concatenate([[-2],company, USP, solv_A, solv_B, add_A_vals, add_B_vals])
   
    features = np.concatenate((int_encodings, float_encodings))
 
    return features



    
class IRSpectraD(DGLDataset):
    def __init__(self):
        self.mode = ":("
        self.save_path_2 = '/home/weeb/shit/Graphormer/examples/property_prediction/training_dataset/'
        ## atom encodings
        atom_type_onehot = [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]

        formal_charge_onehot =[
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ]

        hybridization_onehot =[
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ]

        is_aromatic_onehot = [
            [0], 
            [1]
        ]

        total_num_H_onehot = [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ]

        explicit_valence_onehot = [
            [0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0], 
            [0, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
        ]

        total_bonds_onehot = [
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0], 
            [0, 0, 0, 0, 0, 0, 1],
        ]

        ## TODO: Add encoding for global node in the atom type

        i = 0
        self.one_hotatom_to_int_keys = []
        self.one_hotatom_to_int_values = []
        self.hash_dictatom = {}
        self.comb_atom = False

        if self.comb_atom: ## if you want to do combinatoric atom hashing
            for x1 in atom_type_onehot:
                for x2 in formal_charge_onehot:
                    for x3 in hybridization_onehot:
                        for x4 in is_aromatic_onehot:
                            for x5 in total_num_H_onehot: 
                                for x6 in explicit_valence_oesnehot:
                                    for x7 in total_bonds_onehot:
                                        key = torch.cat([torch.Tensor(y) for y in [x1, x2, x3, x4, x5, x6, x7]])
                                        self.one_hotatom_to_int_keys += [key]
                                        self.one_hotatom_to_int_values += [i]
                                        i+=1
                                                        
            count = 0
            while count < len(self.one_hotatom_to_int_keys):
                h = str(self.one_hotatom_to_int_keys[count])
                self.hash_dictatom[h] = self.one_hotatom_to_int_values[count]
                count +=1
            

        ## combinatoric bond mapping
        bond_type_onehot = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]


        is_in_ring_onehot = [
            [0], 
            [1]
        ]

        bond_stereo_onehot = [
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0]
        ]

        is_global_node = [
            [0],
            [1]
        ] # 2022-12-19

        i = 0
        self.one_hot_to_int_keys = []
        self.one_hot_to_int_values = []
        self.hash_dict = {}
        for x1 in bond_type_onehot:
            for x3 in is_in_ring_onehot:
                for x4 in bond_stereo_onehot:
                    for x5 in is_global_node:
                        key = torch.cat([torch.Tensor(y) for y in [x1, x3, x4, x5]]) ## cute quick way to 
                        self.one_hot_to_int_keys += [key]
                        self.one_hot_to_int_values += [i]
                        i+=1

        count = 0
        while count < len(self.one_hot_to_int_keys):
            h = str(self.one_hot_to_int_keys[count])
            self.hash_dict[h] = self.one_hot_to_int_values[count]
            count +=1

        self.num_classes = 1801
        super().__init__(name='IR Spectra', save_dir='/home/weeb/shit/Graphormer/examples/property_prediction/training_dataset/') 

    def process(self):
        
        self.graphs = []
        self.labels = []
        self.smiles = []

        
        with open('/home/cmkstien/Graphormer_RT/HuanLab/RP_huan_column_params.pkl', 'rb') as handle: ## used for global node hash encodings
            self.columndict = pickle.load(handle) 

        keys = list(self.columndict.keys())
        index_dict = {}
        # print(keys)
        # exit()


        print("I'm in the right file")
        x = import_data(r'/home/cmkstien/Graphormer_RT/HuanLab/RP_RTs.csv')


        gnode = True ## Turns off global node
        count = 0
        count_hash = 0
        companies = []
        for i in tqdm(x):

            # descriptors = [i[idx] for idx in subset_indices]
            # descriptors = i[4:]
            # # descriptors = np.insert(descriptors, 0, -3)
            # descriptors = np.asarray(descriptors)
            # descriptors[descriptors == 'Broken'] = 0.01

            # if count < 14265:
            #     count+=1
            #     continue
        
            sm = str(i[0]).replace("Q", "#") ## Hashtags break some of our preprocessing scripts so we replace them with Qs to make life easier 
            # print(sm)
            mol = Chem.MolFromSmiles(sm)
            rt = torch.tensor([float(i[1])]) / 1000 ## normalizing retention time
            # print(rt)
            index = i[2]
            col_meta = self.columndict[index]



            # descriptors = np.asarray(descriptors, dtype=np.float16)            
            # # descriptors = np.pad(descriptors, (0, 100 - len(descriptors))) 
            # descriptors = np.nan_to_num(descriptors, nan=0.0)
            # descriptors = torch.tensor(descriptors, dtype=torch.float16)
            column_params = featurize_column(col_meta, index)
            ablate_info = False
            if ablate_info: 
                column_params = np.zeros_like(column_params)
                column_params[0] = col_ind

            
            num_atoms = mol.GetNumAtoms()
            add_self_loop = False
            g = mol_to_bigraph(mol, explicit_hydrogens=False, node_featurizer=GraphormerAtomFeaturizer(), edge_featurizer=CanonicalBondFeaturizer(), add_self_loop=False) ## uses DGL featurization function                
            ###########################################################################
            count1 = 0
            count2 = 0

            unif = []
            unifatom = []

            ### GLOBAL NODE Encodings
            
            while count2 < len(g.ndata['h']): ## getting all the parameters needed for the global node generation
                hatom = g.ndata['h'][count2][:]
                unifatom.append(list(np.asarray(hatom)))
                flength = len(list(hatom))
                count2 += 1

            features_gnode = False ## if you want a second global node


            if gnode:
                src_list = list(np.full(num_atoms, num_atoms)) ## node pairs describing edges in heteograph - see DGL documentation
                dst_list = list(np.arange(num_atoms))
                features = torch.tensor([[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]], dtype=torch.float32)
                total_features = features.repeat(num_atoms, 1)
                g_nm = column_params ## custom encoding for the global node
                # g_nm = global_feat #column_params ## custom encoding for the global node
                unifatom.append(g_nm)
                g.add_nodes(1)
                g.ndata['h'] = torch.tensor(np.asarray(unifatom))
                g.add_edges(src_list, dst_list, {'e': total_features}) ## adding all the edges for the global node

            if features_gnode:
                src_list = list(np.full(num_atoms, num_atoms + 1)) ## increasing the global node index by one (second global node)
                dst_list = list(np.arange(num_atoms)) ## no connection to the other global node
                features = torch.tensor([[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]], dtype=torch.float32)
                total_features = features.repeat(num_atoms, 1)
                g.add_nodes(1)
                g_nm = descriptors ## custom encoding for the global node
                unifatom.append(g_nm)
                g.ndata['h'] = torch.tensor(np.asarray(unifatom))
                g.add_edges(src_list, dst_list, {'e': total_features}) ## adding all the edges for the global node
            if g.edata == {}:
                print("We did it mom - one atom molecule doesn't break things")
            else:
                while count1 < len(g.edata['e']): ## doing this for the column metadata
      
                    h = str(g.edata['e'][count1])
                    unif.append(self.hash_dict[h])
                    count1 += 1
                
                count1 = 0
                g.edata['e'] = torch.transpose(torch.tensor(unif), 0, -1) + 1

            self.graphs.append(g)
            self.labels.append(rt)
            self.smiles.append((sm, index))
            count+=1
            # gc.collect()
            # if count == 500:
            #     break
        
    def __getitem__(self, i):
        # print(i)
        return self.graphs[i], self.labels[i], self.smiles[i]

    def __len__(self):
        return len(self.graphs)

@register_dataset("HUAN_test")
def create_customized_dataset():

    dataset = IRSpectraD()
    num_graphs = len(dataset)

    train = 0.8
    val = 0.1
    test = 0.1

    return {
        "dataset": dataset,
        "train_idx":  np.arange(0, num_graphs),#rand_train_idx,#np.arange(0, 4),#
        "valid_idx": None,#
        "test_idx": None, #
        "source": "dgl" 
    }