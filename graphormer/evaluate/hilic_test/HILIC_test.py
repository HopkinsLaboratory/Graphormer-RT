
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


    kPB = tanaka_params[1] # / 10     
    a_CH2 = tanaka_params[2]# / 2
    a_TO = tanaka_params[3]# / 5
    a_CP = tanaka_params[4]
    a_BP = tanaka_params[5] #/ 2 
    a_BP1 = tanaka_params[6] #/ 2
    
    tanaka_params = [kPB, a_CH2, a_TO, a_CP, a_BP, a_BP1, part_size]

    add_A_vals = np.ceil(list(map(float, add_A[::2]))) 
    add_B_vals = np.ceil(list(map(float, add_B[::2]))) 


    add_A_units = add_A[1::2]
    add_B_units = add_B[1::2]

    float_encodings = [diameter, part_size, start_B, t1, B1, t2, B2, t3, B3, pH_A, pH_B, dead, temp, fl, length] 

    float_encodings += tanaka_params
    float_encodings += hsmb_params
    int_encodings = np.concatenate([[-2],company, USP, solv_A, solv_B, add_A_vals, add_B_vals])

    features = np.concatenate((int_encodings, float_encodings))

    return features

class IRSpectraD(DGLDataset):
    def __init__(self):
        self.mode = ":("

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
        self.process()

    def process(self):
        
        self.graphs = []
        self.labels = []
        self.smiles = []

        print("I'm in the right file")
        with open('../../sample_data/HILIC_metadata.pickle', 'rb') as handle: 
            self.columndict = pickle.load(handle) 

        x = import_data(r'../../sample_data/Finetune_0185_HILIC.csv')
        count = 0
        for i in tqdm(x):
            sm = str(i[1]).replace("Q", "#") ##

            mol = Chem.MolFromSmiles(sm)
            rt = torch.tensor([float(i[2])])  ## normalizing retention time
            # print(rt)
            index = i[0]

            col_meta = self.columndict[index]

            column_params = featurize_column(col_meta, index)


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

            gnode = True
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
            # print(rt)
            if torch.isnan(rt):
                print(rt)
                exit()
            self.smiles.append((sm, index))
            count+=1
            # gc.collect()
            # if count == 1000:
            #     break

    def __getitem__(self, i):
        # print(i)
        return self.graphs[i], self.labels[i], self.smiles[i]

    def __len__(self):
        return len(self.graphs)

@register_dataset("HILIC_a")
def create_customized_dataset():

    dataset = IRSpectraD()
    num_graphs = len(dataset)


    return {
        "dataset": dataset,
        "train_idx":  np.arange(num_graphs),#rand_train_idx,#np.arange(0, 4),#
        "valid_idx": None,
        "test_idx": None,
        "source": "dgl" 
    }