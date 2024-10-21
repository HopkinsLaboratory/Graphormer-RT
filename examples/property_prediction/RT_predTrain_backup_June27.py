
from re import L
import numpy as np 
import csv
from rdkit import Chem
import torch

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


companies = ['', 'Waters', 'Thermo', 'Agilent', 'Restek', 'Merck', 'Phenomenex']
USPs = ['', 'L1', 'L10', 'L109', 'L11', 'L43']
HPLC_type = ['RP', 'Other']
solvs = ['h2o','meoh', 'acn']

def one_hot_HPLC_type(HPLC_type):
    one_hot = [0] * len(HPLC_type)
    one_hot[HPLC_type.index(HPLC_type)] = 1
    return one_hot

def one_hot_company(company):
    one_hot = [0] * len(companies)
    one_hot[companies.index(company)] = 1
    return one_hot

def one_hot_USP(USP):
    one_hot = [0] * len(USPs)
    one_hot[USPs.index(USP)] = 1
    return one_hot

def one_hot_solvent(solvent):
    one_hot = [0] * len(solvs)
    one_hot[solvs.index(solvent)] = 1
    return one_hot

def featurize_column(column_params, index):

    company = one_hot_company(column_params[0])
    USP = one_hot_USP(column_params[1])
    length = float(column_params[2]) // 50 ## consider mapping these into fixed bins for steps of 50
    if column_params[3] == '':
        diameter = 0
    else:
        diameter = float(column_params[3])

    part_size = float(column_params[4])
    temp = float(column_params[5]) / 100 ## normalizing temperature (rethink this maybe)
    fl = float(column_params[6]) ## Double check that fl and col_fl are two different values
    dead = float(column_params[7]) ## dead time - MAYBE REMOVE COLUMN THAT HAS DEAD TIME OF ZERO
    # HPLC_type = one_hot_HPLC_type(column_params[8])

    solv_A = one_hot_solvent(column_params[9])
    solv_B = one_hot_solvent(column_params[10])
    # time_start_B = float(column_params[11]) ## this is always zero, so redundant
    start_B = float(column_params[12]) / 100
    t1 = float(column_params[13]) ## maybe consider normalizing the times as well to the fraction of the gradient ? TODO: 
    B1 = float(column_params[14]) / 100
    t2 = float(column_params[15])
    B2 = float(column_params[16]) / 100
    t3 = float(column_params[17])
    B3 = float(column_params[18]) / 100

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
    ## COARSE NORMALIZATION BASED ON VALUES
    # kPB = tanaka_params[1] / 10     
    # a_CH2 = tanaka_params[2] / 2
    # a_TO = tanaka_params[3] / 5
    # a_CP = tanaka_params[4]
    # a_BP = tanaka_params[5] / 2 
    # a_BP1 = tanaka_params[6] / 2
    # part_size = tanaka_params[7] / 5
    
    # tanaka_params = [kPB, a_CH2, a_TO, a_CP, a_BP, a_BP1, part_size]

    add_A_vals = np.ceil(list(map(float, add_A[::2])))
    add_B_vals = np.ceil(list(map(float, add_B[::2])))

    add_A_units = add_A[1::2]
    add_B_units = add_B[1::2]
    # if index == '0186':
    #     print(len(add_A_vals), len(add_B_vals), len(company), len(USP), len(HPLC_type), len(solv_A), len(solv_B), len(add_A_vals), len(add_B_vals))
    # if index == '0126':
    #     print(len(add_A_vals), len(add_B_vals), len(company), len(USP), len(HPLC_type), len(solv_A), len(solv_B), len(add_A_vals), len(add_B_vals))
    #     exit()
    float_encodings = [length, diameter, part_size, start_B, t1, B1, t2, B2, t3, B3, pH_A, pH_B,] # float encodings
    # float_encodings += tanaka_params
    # float_encodings += hsmb_params

    features = np.concatenate(([-2], company, USP, solv_A, solv_B, add_A_vals, add_B_vals, float_encodings))
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

        print("I'm in the right file")
        x = import_data(r'/home/cmkstien/Desktop/RT_data/split_42/filter_METLIN/42_train.csv')
        x = x[1:] ## removing header        

        with open('/home/cmkstien/Desktop/RT_data/col_metadata.pickle', 'rb') as handle: ## used for global node hash encodings
            self.columndict = pickle.load(handle) 

        # for key in self.columndict.keys():
        #     print(len(self.columndict[key]), key)

        c = 0
        agg = []
        n = 91
        # for i in self.columndict.keys():
        #     if c==0:
        #         c+=1
        #         print(self.columndict[i][n])
        #         continue

        #     agg.append(self.columndict[i][n]) ## TODO: Fix dead times for 
        # print(list(set(agg)))
        # exit()
        # exit()
        
        print("Loading Data and Converting SMILES to DGL graphs")
        count_outliers = 0

        gnode = True ## Turns off global node
        count = 0
        count_hash = 0
        companies = []
        for i in tqdm(x):
            # if count < 14265:
            #     count+=1
            #     continue
        
            sm = str(i[1]).replace("Q", "#") ## Hashtags break some of our preprocessing scripts so we replace them with Qs to make life easier 
            # print(sm)
            rt = torch.tensor([float(i[2])]) / 1000 ## normalizing retention time
            # print(rt)
            index = i[0]
            col_meta = self.columndict[index]
            # if count == 14269:
            #     print(sm, rt)
    
            # print(col_meta)s
            column_params = featurize_column(col_meta, index)

            mol = Chem.MolFromSmiles(sm)
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

            src_list = list(np.full(num_atoms, num_atoms)) ## node pairs describing edges in heteograph - see DGL documentation
            dst_list = list(np.arange(num_atoms))

            features = torch.tensor([[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]], dtype=torch.float32)
            total_features = features.repeat(num_atoms, 1)

            if gnode:
                features = torch.tensor([[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]], dtype=torch.float32)
                total_features = features.repeat(num_atoms, 1)
                g_nm = column_params ## custom encoding for the global node
                unifatom.append(g_nm)
                g.add_nodes(1)

                # try:
                g.ndata['h'] = torch.tensor(np.asarray(unifatom))
                # except:
                #     # print(len(g_nm))
                #     # print(index)
                #     # print(len(unifatom[0]), 'unifatom')
                #     # print(g_nm)
                #     exit()

                g.add_edges(src_list, dst_list, {'e': total_features}) ## adding all the edges for the global node

            if g.edata == {}:
                print("We did it mom - one atom molecule doesn't break things")
            else:
                while count1 < len(g.edata['e']):
      
                    h = str(g.edata['e'][count1])
                    unif.append(self.hash_dict[h])
                    count1 += 1

                g.edata['e'] = torch.transpose(torch.tensor(unif), 0, -1) + 1


            self.graphs.append(g)
            self.labels.append(rt)
            self.smiles.append((sm, index))
            count+=1
            # gc.collect()
            # if count == 100:
            #     break


    def __getitem__(self, i):
        # print(i)
        return self.graphs[i], self.labels[i], self.smiles[i]

    def __len__(self):
        return len(self.graphs)

@register_dataset("RT_Library")
def create_customized_dataset():

    dataset = IRSpectraD()
    num_graphs = len(dataset)

    train = 0.8
    val = 0.1
    test = 0.1

    return {
        "dataset": dataset,
        "train_idx":  np.arange(0, int(num_graphs * train)),#rand_train_idx,#np.arange(0, 4),#
        "valid_idx": np.arange(int(num_graphs * train), int(num_graphs * (train + val))),#np.arange(4,10), #
        "test_idx": None, #
        "source": "dgl" 
    }