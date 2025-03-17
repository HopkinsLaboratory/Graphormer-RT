
from re import L
import numpy as np 
import csv
from dgl.data import QM9 
from rdkit import Chem
from dgllife.utils import BaseBondFeaturizer
import dgllife.utils as d
import torch
import json
import pandas as pd
from functools import partial
from rdkit.Chem import rdmolfiles, rdmolops

import itertools
import os.path as osp

import dgl
from dgl.data import DGLDataset
import torch
import os

from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
import dgl.backend as F
import pickle

from dgl import save_graphs, load_graphs
from dgl.data.utils import makedirs, save_info, load_info
from joblib import delayed, Parallel

try:
    from rdkit import Chem, RDConfig
    from rdkit.Chem import AllChem, ChemicalFeatures

except ImportError:
    pass

import json
import logging

import numpy as np
import pandas as pd
import scipy.stats as st
import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
import mordred
from mordred import Calculator, descriptors


import mordred
import rdkit
from importlib.metadata import version


import warnings

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

def remove_mordred_duplicates(desc_list, RDKit_Descriptors):

    duplicate_indeces = []
    duplicates = []
    cleared_mordred_descriptors = list()
    
    for potential_duplicate in desc_list:
        if potential_duplicate in RDKit_Descriptors:
            
            dup_index = desc_list.index(potential_duplicate)
            duplicate_indeces.append(dup_index)
            duplicates.append(potential_duplicate)
        
        else:
            cleared_mordred_descriptors.append(potential_duplicate)

    return cleared_mordred_descriptors, duplicate_indeces

def get_all_descriptor_names():
        """
        Get available descriptor names for RDKit and mordred physchem features. Custom subset can be used as list of ALL_descriptors.
        """
        cleared_mordred_descriptors = []
        len_mordred = 0
        len_RDKit = 0
        extra_header_list = ["aa", "a_prop", "t_order", "c_ar", "frac_C", "TPSA_2", "f_count", "oh_count", "cl_count", "amideB"]
        mordred_descriptors = Calculator(descriptors, ignore_3D=True).descriptors
        mordred_descriptors = (list(mordred_descriptors))
        for i in range(0, len(mordred_descriptors)):
            cleared_mordred_descriptors.append(str(mordred_descriptors[i]))
                            
        rdkit_descriptors = [x[0] for x in Descriptors._descList]
        rdkit_descriptors.remove('SPS')

        # Features that exist both in RDKit and mordred. Keep RDKit ones.
        cleared_mordred_descriptors, duplicate_indeces = remove_mordred_duplicates(cleared_mordred_descriptors, rdkit_descriptors)
        len_mordred = len(cleared_mordred_descriptors)
        len_RDKit = len(rdkit_descriptors)
        final_descriptors = extra_header_list
        final_descriptors.extend(rdkit_descriptors)
        final_descriptors.extend(cleared_mordred_descriptors)   

    
        return final_descriptors, len_RDKit, len_mordred, duplicate_indeces

# distributions_path = '/home/cmkstien/Graphormer_RT_extra/CDFs/descriptorCDFs.json'
# with open(distributions_path) as dp:
#     distributions = json.load(dp)
# This code, given the list of SMILES strings allows to compute the descriptors in RDKit and mordred for given molecules. 
# SMILES_address and json_address are file locations from which the program pulls data.
# output_address is file location and file format to which the program will save the results.
# normalise - boolean that when positively selected, has all the results be normalized using the cumulative distribution function.
# preserve_broken - boolean, that when positive , has the descriptors that did not pass tolerance check still appear in the final product but with "Broken" for values.
# tolerance_percentage - defines percentage of what is considered an acceptable amount of descriptors that do not work.

all_descriptors_output = get_all_descriptor_names()
descriptor_names = all_descriptors_output[0]
duplicate_indeces = all_descriptors_output[3]


def Calc_Descriptors(mol,  normalise, ):
    def remove_broken_columns(results_frame, descriptor_names, tolerance_percentage):
        to_remove = list()
        columns_removed = 0
        tolerance = np.round(tolerance_percentage/100 * results_frame.shape[0])

        for column_name in descriptor_names:
            single_column = results_frame[column_name].values
            broken_counter = 0
            for val in single_column:
                if val == "Broken":
                    broken_counter += 1
            if broken_counter > tolerance:
                to_remove.append(column_name)
                columns_removed += 1

        for i in to_remove:
            descriptor_names.remove(i)

        results_frame = results_frame.drop(to_remove, axis=1)
        return results_frame

    # 4. Remove Broken elements while retaining the indeces inside of the single column
    def remove_Broken_and_store_positions(single_column):
        broken_indeces = list()
        column_of_values = list()

        for i in range(0, len(single_column)):
            if single_column[i] == "Broken":
                broken_indeces.append(i)     
            else:
                column_of_values.append(np.float64(single_column[i]))

        return column_of_values, broken_indeces

    # Creates a column of "Broken" values and concatonates it to post_cdf_frame
    def add_broken_column(post_cdf_frame, SMILES_data):
        broken_list = ["Broken"] * len(SMILES_data)
        broken_frame = pd.DataFrame(broken_list, index=None, columns=[column_name])
        post_cdf_frame = pd.concat([post_cdf_frame, broken_frame], axis=1)
        return post_cdf_frame

    # This function rebuilds the whole column after cdf was applied to it. If it had entries with "Broken", those get readded at the same position. The final result is 
    # concatenated to the final result frame, post_cdf_frame.
    def rebuild_entry(column_shape, cdf_results, post_cdf_frame):
                array_ind = 0
                for i in range(0, len(column_shape)):
                    if i in broken_indeces:
                        column_shape[i] = 'Broken'

                    else:
                        column_shape[i] = cdf_results[array_ind]
                        array_ind += 1

                column_frame = pd.DataFrame(column_shape, index=None, columns=[column_name])
                post_cdf_frame = pd.concat([post_cdf_frame, column_frame], axis=1)

                return post_cdf_frame

    # Removes the SMILES that have less than 4 heavy atoms and do not contain carbon. Returns the list with those removed.
    
    # Removes the duplicate indeces that overlap between RDKit and mordred. It does it by comparing the full list of descriptors to list of RDKit Descriptors.
    # Returns the cleared list and also the list of duplicates that occured.


    # Creates a list of all descriptor names, for those defined by Cailum, RDKit and mordred. Returns final descriptor list, along with how many RDKit and how many mordred
    # descriptors there were in the final list and also showing any duplicates.


# Cailum's functions below
    def get_total_bond_order(molecule):
        total_order = 0
        for bond in molecule.GetBonds():
            total_order += bond.GetBondTypeAsDouble()
        return total_order

    def carbon_anal(molecule):
        total = 0
        c_count = 0
        c_ar = 0
        c_SP2 = 0
        f_count = 0
        cl_count = 0
        oh_count = 0
        for atom in molecule.GetAtoms():
            if atom.GetAtomicNum() == 6:
                c_count +=1
                if atom.GetIsAromatic():
                    c_ar +=1 
                if atom.GetHybridization() == rdkit.Chem.rdchem.HybridizationType.SP2:
                    c_SP2 += 1
            if atom.GetAtomicNum() == 9:
                f_count +=1
            if atom.GetAtomicNum() == 17:
                cl_count +=1
            if atom.GetAtomicNum() == 8:
                if atom.GetTotalNumHs() == 1:
                    oh_count+=1
            total+=1

        frac_C = c_count / total
        return c_ar, frac_C, f_count, oh_count, cl_count

    def AromaticAtoms(molecule):
        aromatic_atoms = [molecule.GetAtomWithIdx(i).GetIsAromatic() for i in range(molecule.GetNumAtoms())]
        aa_count = []
        for i in aromatic_atoms:
            if i==True:
                aa_count.append(1)
        sum_aa_count = sum(aa_count)
        return sum_aa_count

    # Given a SMILES string, it calculates the descriptors defined by Cailum and returns them as a list.
    def Extra_descriptors_calculator(molecule):
        extra_header_list = ["aa", "a_prop", "t_order", "c_ar", "frac_C", "TPSA_2", "f_count", "oh_count", "cl_count", "amideB"]
        
        c_ar, frac_C,f_count, oh_count, cl_count = carbon_anal(molecule)

        
        ## Manual Calculation of Other Descriptors 
        aa = AromaticAtoms(molecule)  
        ha = Descriptors.HeavyAtomCount(molecule)
        Aromatic_prop = aa/ha
        a_prop = Aromatic_prop
        t_order =get_total_bond_order(molecule)
        c_ar =c_ar
        c_frac = frac_C
        TPSA_2 =Descriptors.TPSA(molecule,includeSandP=True)
        F = f_count
        Oh = oh_count
        Cl = cl_count
        amideB = Chem.rdMolDescriptors.CalcNumAmideBonds(molecule)
        single_entry = [aa, a_prop, t_order, c_ar, c_frac, TPSA_2, F, Oh, Cl, amideB]
        return single_entry

    # Given the molecule (SMILES string) and the list of duplicate_indeces(to know which ones to skip), the function calculates all descriptors, Cailum's, RDKit and mordred
    # for a given SMILES.
    def calculate_descriptors(molecule, duplicate_indeces):
        

        if molecule is None:
            # invalid?
            logging.warning(f'Chem.MolFromSmiles failed smiles="{molecule}"')
            return None

        # Define calculators and calculate descriptors
        extra_desc_results = Extra_descriptors_calculator(SMILES)
        RDKit_results =  rdkit.Chem.Descriptors.CalcMolDescriptors(molecule, missingVal="Broken", silent=False)

        Mordred_calc = Calculator(descriptors, ignore_3D=True)
        mordred_results = Mordred_calc(molecule)

        RDKit_results = list(RDKit_results.values())
        mordred_results = list(mordred_results.values())

        for i in range(0, len(mordred_results)):
                if (type(mordred_results[i]) == mordred.error.Missing or type(mordred_results[i]) == mordred.error.Error):
                    mordred_results[i] = np.nan

        mordred_cleared_result = []
        
    # 4. Remove columns that are duplicates of RDKit descriptors in Mordred
        for i in range(0,len(mordred_results)):
            if i in duplicate_indeces:
                pass
            else:
                mordred_cleared_result.append(mordred_results[i])
        
        final_output = extra_desc_results
        final_output.extend(RDKit_results)
        final_output.extend(mordred_cleared_result)

        for i in range(0, len(final_output)):
            if np.isfinite(final_output[i]):
                final_output[i] = np.float64(final_output[i])

            else:
                final_output[i] = "Broken"

        return final_output

    # Function takes a frame of results that were previously calculated, along with list of descriptor names and tolerance percentage that defines the percentage of molecules
    # for which getting "Broken" value is acceptable. It then determines which columns are broken, removes them from results_frame and returns the modified results_frame.
    # Main Code Structure:

    # 1. Declare end result frame
    results_frame = list()

    # 3. Convert to molecule object and calculate all descriptors. Add result to main frame.
    i = 0
    SMILES_data = [mol]
    SMILES_frame = SMILES_data.copy()
    SMILES_frame = pd.DataFrame(SMILES_frame, index=None, columns=["SMILES"])
    for SMILES in SMILES_data:
        if i % 1000 == 0:
            print(i)
        results_frame.append(calculate_descriptors(SMILES, duplicate_indeces = []))
        i+=1

    

    # Initial result of calculation of all descriptors. If preserve_broken = True and normalise = False, this is final output.
    results_frame = pd.DataFrame(results_frame, columns = descriptor_names)
    results_frame = pd.concat([SMILES_frame, results_frame], axis=1)

    # Normalization starts here
    if normalise == True:
        post_cdf_frame = SMILES_frame.copy()

        # Those columns have had natural log (ln) applied to their fit and also need ln applied before cdf function
        columns_to_ln = ["VR1_A", "VR2_A", "Ipc"]
        # Those columns are already normalized and can be skipped on the cdf function
        pre_normalized_columns = ["Lipinski", "GhoseFilter"]
        
        for column_name in descriptor_names:
            
            # Skip for columns that are already normalized, concatenating them to final result frame
            if column_name in pre_normalized_columns:
                column_frame = pd.DataFrame(results_frame[column_name].values, index=None, columns=[column_name])
                post_cdf_frame = pd.concat([post_cdf_frame, column_frame], axis=1)

            # Go through distributions and apply cdf function to correspodning data. Concatenate results to final frame
            elif column_name in distributions.keys():
                
                try:
                    current_column = results_frame[column_name].values
                    column_shape = list(np.zeros(len(current_column)))

                    # Create a separate frame for numerical data to which cdf will be applied.
                    # Also, preserve "Broken" indeces, so that the final data frame can properly be reassembled.
                    to_cdf, broken_indeces = remove_Broken_and_store_positions(current_column)
                
                # Apply the CDFs
                # obtain cdf function parameters
                    fit = distributions[column_name]
                    dist = getattr(st, fit[0])
                    arg = fit[1][:-2]
                    loc = fit[1][-2] 
                    scale = fit[1][-1]  
                    minV = np.float64(fit[2])
                    maxV = np.float64(fit[3])

                    to_cdf = np.array(to_cdf)

                    # Extra operation for columns that need ln to work properly
                    if column_name in columns_to_ln:
                        to_cdf = np.log(to_cdf)

                    cdf_results = dist.cdf(np.clip(to_cdf, minV, maxV), loc=loc, scale=scale, *arg)
                    
                    # Rebuild the column structure and concatenate to the final frame
                    post_cdf_frame = rebuild_entry(column_shape, cdf_results, post_cdf_frame)
            

                except:
                    post_cdf_frame = add_broken_column(post_cdf_frame, SMILES_data)
                    print("ERROR happened")
            
                    # Adding a column composed of all "Broken" for cases that did not work
            else:
                post_cdf_frame = add_broken_column(post_cdf_frame, SMILES_data)
                    
        results_frame = post_cdf_frame
        features = results_frame.iloc[0].values


    return features



## Featurization functions from DGL, some custom, some modified. Was easier to store locally

def atom_group(atom, return_one_hot=True, unknown_group=None):
    """
    Get the group number (column number) of an RDKit atom object in the periodic table.

    Parameters:
        atom (rdkit.Chem.Atom): The RDKit atom object.
        return_one_hot (bool): Whether to return the group as a one-hot encoding.
        unknown_group (int or list or None): The encoding to return for atoms with unknown groups.

    Returns:
        int or list or None: The group number of the atom as an integer (if return_one_hot is False),
                             or a one-hot encoding of the group as a list (if return_one_hot is True),
                             or the value provided in unknown_group if the group is not found.
    """
    # Get the atomic number of the atom
    atomic_number = atom.GetAtomicNum()

    # Map atomic numbers to group numbers (column numbers) in the periodic table
    atomic_number_to_group = {
        1: 1, 2: 18,
        3: 1, 4: 14, 5: 15, 6: 16, 7: 17,
        8: 18, 9: 17, 10: 18, 11: 1, 12: 2,
        13: 13, 14: 14, 15: 15, 16: 16, 17: 17,
        18: 18, 19: 1, 20: 2,
        # Add halogens:
        53: 17,  
        35: 17,  
        9: 17,   # Fluorine (F) is in group 17
        # Continue mapping atomic numbers to groups as needed
    }

    # Get the group based on the atomic number
    group = atomic_number_to_group.get(atomic_number, None)

    if return_one_hot:
        num_groups = 18  # Assuming there are 18 groups in the periodic table

        # Encode group as one-hot
        one_hot_group = [0] * num_groups
        if group is not None:
            one_hot_group[group - 1] = 1

        return one_hot_group if group is not None else unknown_group
    else:
        return [group] if group is not None else unknown_group




def atom_period(atom, return_one_hot=True, unknown_period=None):
    """
    Get the period (row number) of an RDKit atom object in the periodic table.

    Parameters:
        atom (rdkit.Chem.Atom): The RDKit atom object.
        return_one_hot (bool): Whether to return the period as a one-hot encoding.
        unknown_period (int or list or None): The encoding to return for atoms with unknown periods.

    Returns:
        int or list or None: The period (row number) of the atom as an integer (if return_one_hot is False),
                             or a one-hot encoding of the period as a list (if return_one_hot is True),
                             or the value provided in unknown_period if the period is not found.
    """
    # Get the atomic number of the atom
    atomic_number = atom.GetAtomicNum()

    # Map atomic numbers to periods (row numbers) in the periodic table
    atomic_number_to_period = {
        1: 1, 2: 1,
        3: 2, 4: 2, 5: 2, 6: 2, 7: 2,
        8: 2, 9: 2, 10: 2, 11: 3, 12: 3,
        13: 3, 14: 3, 15: 3, 16: 3, 17: 2,  # Chlorine (Cl) is in period 2 (change from the previous version)
        18: 3, 19: 4, 20: 4,
        # Add halogens:
        53: 5,  # Iodine (I) is in period 5
        35: 3,  # Bromine (Br) is in period 3
        9: 2,   # Fluorine (F) is in period 2
        # Continue mapping atomic numbers to periods as needed
    }

    # Get the period based on the atomic number
    period = atomic_number_to_period.get(atomic_number, None)
    if return_one_hot:
        num_periods = 7  # Assuming there are 7 periods in the periodic table
        one_hot = [0] * num_periods
        if period is not None:
            one_hot[period - 1] = 1
        return one_hot if period is not None else unknown_period
    else:
        return [period] if period is not None else unknown_period


def atom_mass(atom, coef=0.01):
    """Get the mass of an atom and scale it.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    coef : float
        The mass will be multiplied by ``coef``.

    Returns
    -------
    list
        List containing one float only.
    """
    return [atom.GetMass() * coef]

def atom_explicit_valence_one_hot(atom, allowable_set=None, encode_unknown=False):
    
    allowable_set = list(range(0, 7))
    # print(atom.GetExplicitValence())
    return d.one_hot_encoding(atom.GetExplicitValence(), allowable_set, encode_unknown)


def atom_type_one_hot(atom, allowable_set=None, encode_unknown=False):

    allowable_set = ['H', 'C', 'N', 'O', 'F','Si', 'P', 'S', 'Cl', 'Br', 'I']
    return d.one_hot_encoding(atom.GetSymbol(), allowable_set, encode_unknown)

def atom_is_aromatic_one_hot(atom, allowable_set=None, encode_unknown=False):
    
    if allowable_set is None:
        allowable_set = [False, True]
    val = atom.GetIsAromatic()
    if val:
        return [0]
    else:
        return [1] 

def atom_hybridization_one_hot(atom, allowable_set=None, encode_unknown=False):

    allowable_set = [
                    Chem.rdchem.HybridizationType.S, 
                    Chem.rdchem.HybridizationType.SP,
                    Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3,                       
                    Chem.rdchem.HybridizationType.SP3D,
                    Chem.rdchem.HybridizationType.SP3D2 
                    ]
    # print(atom.GetHybridization())
    return d.one_hot_encoding(atom.GetHybridization(), allowable_set, encode_unknown)


def atom_formal_charge_one_hot(atom, allowable_set=None, encode_unknown=False):
    if allowable_set is None:
        allowable_set = list(range(-2, 4))
    return d.one_hot_encoding(atom.GetFormalCharge(), allowable_set, encode_unknown)


def construct_bigraph_from_mol(mol, add_self_loop=False): ## modified edge to bigraph to enable adding of global node 
  
    g = dgl.graph(([], []), idtype=torch.int32)

    # Add nodes
    num_atoms = mol.GetNumAtoms()
    g.add_nodes(num_atoms)

    # Add edges
    src_list = []
    dst_list = []
    num_bonds = mol.GetNumBonds()
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        src_list.extend([u, v])
        dst_list.extend([v, u])
    # for i in range(num_atoms):
    #     src_list.append(num_atoms+1)
    #     drc_list.append(i)

    if add_self_loop:
        nodes = g.nodes().tolist()
        src_list.extend(nodes)
        dst_list.extend(nodes)

    g.add_edges(torch.IntTensor(src_list), torch.IntTensor(dst_list))

    return g




def smiles_to_bigraph(smiles, add_self_loop=False,
                      node_featurizer=None,
                      edge_featurizer=None,
                      canonical_atom_order=True,
                      explicit_hydrogens=False,
                      num_virtual_nodes=0):
    mol = Chem.MolFromSmiles(smiles)
    num_atoms = mol.GetNumAtoms()
    src = []
    dst = []
    for i in range(num_atoms):
        for j in range(num_atoms):
            if i != j or add_self_loop:
                src.append(i)
                dst.append(j)

    g = dgl.graph((torch.IntTensor(src), torch.IntTensor(dst)), idtype=torch.int32)

    return g

def atom_partial_charge(atom):
   
    gasteiger_charge = atom.GetProp('_GasteigerCharge')
    if gasteiger_charge in ['-nan', 'nan', '-inf', 'inf']:
        gasteiger_charge = 0
    return [float(gasteiger_charge)]

featurizer_funcs = ['one_hot_encoding',
           'atom_type_one_hot',
           'atomic_number_one_hot',
           'atomic_number',
           'atom_degree_one_hot',
           'atom_degree',
           'atom_total_degree_one_hot',
           'atom_total_degree',
           'atom_explicit_valence_one_hot',
           'atom_explicit_valence',
           'atom_implicit_valence_one_hot',
           'atom_implicit_valence',
           'atom_hybridization_one_hot',
           'atom_total_num_H_one_hot',
           'atom_total_num_H',
           'atom_formal_charge_one_hot',
           'atom_formal_charge',
           'atom_num_radical_electrons_one_hot',
           'atom_num_radical_electrons',
           'atom_is_aromatic_one_hot',
           'atom_is_aromatic',
           'atom_is_in_ring_one_hot',
           'atom_is_in_ring',
           'atom_chiral_tag_one_hot',
           'atom_chirality_type_one_hot',
           'atom_mass',
           'atom_is_chiral_center',
           'ConcatFeaturizer',
           'BaseAtomFeaturizer',
           'CanonicalAtomFeaturizer',
           'WeaveAtomFeaturizer',
           'PretrainAtomFeaturizer',
           'AttentiveFPAtomFeaturizer',
           'PAGTNAtomFeaturizer',
           'bond_type_one_hot',
           'bond_is_conjugated_one_hot',
           'bond_is_conjugated',
           'bond_is_in_ring_one_hot',
           'bond_is_in_ring',
           'bond_stereo_one_hot',
           'bond_direction_one_hot',
           'BaseBondFeaturizer',
           'CanonicalBondFeaturizer',
           'WeaveEdgeFeaturizer',
           'PretrainBondFeaturizer',
           'AttentiveFPBondFeaturizer',
           'PAGTNEdgeFeaturizer']


class BaseAtomFeaturizer(object):
    def __init__(self, featurizer_funcs, feat_sizes=None):
        self.featurizer_funcs = featurizer_funcs
        if feat_sizes is None:
            feat_sizes = dict()
        self._feat_sizes = feat_sizes

    def feat_size(self, feat_name=None):
        """Get the feature size for ``feat_name``.

        When there is only one feature, users do not need to provide ``feat_name``.

        Parameters
        ----------
        feat_name : str
            Feature for query.

        Returns
        -------
        int
            Feature size for the feature with name ``feat_name``. Default to None.
        """
        if feat_name is None:
            assert len(self.featurizer_funcs) == 1, \
                'feat_name should be provided if there are more than one features'
            feat_name = list(self.featurizer_funcs.keys())[0]

        if feat_name not in self.featurizer_funcs:
            return ValueError('Expect feat_name to be in {}, got {}'.format(
                list(self.featurizer_funcs.keys()), feat_name))

        if feat_name not in self._feat_sizes:
            atom = Chem.MolFromSmiles('C').GetAtomWithIdx(0)
            self._feat_sizes[feat_name] = len(self.featurizer_funcs[feat_name](atom))

        return self._feat_sizes[feat_name]

    def __call__(self, mol):
        """Featurize all atoms in a molecule.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule instance.

        Returns
        -------
        dict
            For each function in self.featurizer_funcs with the key ``k``, store the computed
            feature under the key ``k``. Each feature is a tensor of dtype float32 and shape
            (N, M), where N is the number of atoms in the molecule.
        """
        num_atoms = mol.GetNumAtoms()
        atom_features = defaultdict(list)

        # Compute features for each atom
        for i in range(num_atoms):
            atom = mol.GetAtomWithIdx(i)
            for feat_name, feat_func in self.featurizer_funcs.items():
                atom_features[feat_name].append(feat_func(atom))

        # Stack the features and convert them to float arrays
        processed_features = dict()
        for feat_name, feat_list in atom_features.items():
            feat = np.stack(feat_list)
            processed_features[feat_name] = F.zerocopy_from_numpy(feat.astype(np.float32))

        return processed_features


def atom_total_bonds(atom, allowable_set=None, encode_unknown=False):
    # print("IM USING THIS FUNCTION")
    mol = atom.GetOwningMol()
    id = atom.GetIdx()
    count = 0
    num_bonds = mol.GetNumBonds()
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        if u == id or v == id:
            count += 1

    allowable_set = list(range(0,7))
    # print(count, "YOU HAVE THIS MANY BONDS")
    return d.one_hot_encoding(count, allowable_set, encode_unknown)

class ConcatFeaturizer(object):

    def __init__(self, func_list):
        self.func_list = func_list


    def __call__(self, x):
        return list(itertools.chain.from_iterable(
            [func(x) for func in self.func_list]))

def is_global_node(is_gnode:bool): # 2022-12-19
    return [0]

class CanonicalBondFeaturizer(BaseBondFeaturizer):
    def __init__(self, bond_data_field='e', self_loop=False):
        super(CanonicalBondFeaturizer, self).__init__(
            featurizer_funcs={bond_data_field: ConcatFeaturizer(
                [d.bond_type_one_hot,
                #  d.bond_is_conjugated, ## we realized this encoding was redundanct based on bond type
                 d.bond_is_in_ring,
                 d.bond_stereo_one_hot,
                 is_global_node]

            )}, self_loop=self_loop)


class GraphormerAtomFeaturizer(object):
    def __init__(self, atom_data_field='h', atom_types=None, chiral_types=None,
                 hybridization_types=None):
        super(GraphormerAtomFeaturizer, self).__init__()

        self._atom_data_field = atom_data_field

        if atom_types is None:
            atom_types = ['C', 'N', 'O', 'F','Si', 'P', 'S', 'Cl', 'Br', 'I']
        self._atom_types = atom_types

        if chiral_types is None:
            chiral_types = [Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW]
        self._chiral_types = chiral_types

        if hybridization_types is None:
            hybridization_types = [Chem.rdchem.HybridizationType.SP,
                                   Chem.rdchem.HybridizationType.SP2,
                                   Chem.rdchem.HybridizationType.SP3]
        self._hybridization_types = hybridization_types

    
        self._featurizer = ConcatFeaturizer([
            # d.atomic_number, ## this is the set of atom features, can be commented/removed. Make sure to change input value in graphormer_layers.py if you do
            atom_type_one_hot,
            atom_formal_charge_one_hot, 
            atom_hybridization_one_hot, 
            atom_is_aromatic_one_hot,
            d.atom_total_num_H_one_hot, 
            atom_explicit_valence_one_hot,
            atom_total_bonds,

            atom_partial_charge, 
            atom_mass, 
            # atom_group,
            # atom_period,

            ])

    def feat_size(self):
        """Get the feature size.

        Returns
        -------
        int
            Feature size.
        """
        mol = Chem.MolFromSmiles('C')
        feats = self(mol)[self._atom_data_field]

        return feats.shape[-1]

    def __call__(self, mol):
        """Featurizes the input molecule.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule instance.

        Returns
        -------
        dict
            Mapping atom_data_field as specified in the input argument to the atom
            features, which is a float32 tensor of shape (N, M), N is the number of
            atoms and M is the feature size.
        """
        atom_features = []

        AllChem.ComputeGasteigerCharges(mol)
        num_atoms = mol.GetNumAtoms()

        # Get information for donor and acceptor
        fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
        mol_featurizer = ChemicalFeatures.BuildFeatureFactory(fdef_name)
        mol_feats = mol_featurizer.GetFeaturesForMol(mol)

        # Get a symmetrized smallest set of smallest rings
        # Following the practice from Chainer Chemistry (https://github.com/chainer/
        # chainer-chemistry/blob/da2507b38f903a8ee333e487d422ba6dcec49b05/chainer_chemistry/
        # dataset/preprocessors/weavenet_preprocessor.py)
        sssr = Chem.GetSymmSSSR(mol)
        for i in range(num_atoms):
            atom = mol.GetAtomWithIdx(i)
            # Features that can be computed directly from RDKit atom instances, which is a list
            feats = self._featurizer(atom) ## 18 here is to pad
            extend = 90 - len(feats) ## 1779 ## 67 for only column stuff
            feats.extend([0]*extend) ## 38 with tanaka and hsmbd


            atom_features.append(feats)
        atom_features = np.stack(atom_features)
  

        return {self._atom_data_field: F.zerocopy_from_numpy(atom_features.astype(np.float32))}

def mol_to_graph(mol, graph_constructor, node_featurizer, edge_featurizer,
                 canonical_atom_order, explicit_hydrogens=False, num_virtual_nodes=0): ## modified mol_to_graph -> needed for global node

    if mol is None:
        print('Invalid mol found')
        return None

    # Whether to have hydrogen atoms as explicit nodes
    if explicit_hydrogens:
        mol = Chem.AddHs(mol)

    if canonical_atom_order:
        new_order = rdmolfiles.CanonicalRankAtoms(mol)
        mol = rdmolops.RenumberAtoms(mol, new_order)
    g = graph_constructor(mol)

    if node_featurizer is not None:
        g.ndata.update(node_featurizer(mol))

    if edge_featurizer is not None:
        g.edata.update(edge_featurizer(mol))

    if num_virtual_nodes > 0:
        num_real_nodes = g.num_nodes()
        real_nodes = list(range(num_real_nodes))
        g.add_nodes(num_virtual_nodes)

        # Change Topology
        virtual_src = []
        virtual_dst = []
        for count in range(num_virtual_nodes):
            virtual_node = num_real_nodes + count
            virtual_node_copy = [virtual_node] * num_real_nodes
            virtual_src.extend(real_nodes)
            virtual_src.extend(virtual_node_copy)
            virtual_dst.extend(virtual_node_copy)
            virtual_dst.extend(real_nodes)
        g.add_edges(virtual_src, virtual_dst)

        for nk, nv in g.ndata.items():
            nv = torch.cat([nv, torch.zeros(g.num_nodes(), 1)], dim=1)
            nv[-num_virtual_nodes:, -1] = 1
            g.ndata[nk] = nv

        for ek, ev in g.edata.items():
            ev = torch.cat([ev, torch.zeros(g.num_edges(), 1)], dim=1)
            ev[-num_virtual_nodes * num_real_nodes * 2:, -1] = 1
            g.edata[ek] = ev

    return g

def mol_to_bigraph(mol, add_self_loop=False,
                   node_featurizer=None,
                   edge_featurizer=None,
                   canonical_atom_order=False, ## I changed this
                   explicit_hydrogens=False,
                   num_virtual_nodes=0):

    
    return mol_to_graph(mol, partial(construct_bigraph_from_mol, add_self_loop=add_self_loop),
                        node_featurizer, edge_featurizer,
                        canonical_atom_order, explicit_hydrogens, num_virtual_nodes=0)



def import_smiles(file):
    with open(file,'r') as rf:
        r=csv.reader(rf)
        # next(r)
        smiles=[]
        for row in r:
            smiles.append(row[0])
        return smiles


def import_data(file):
    with open(file,'r',  encoding='latin-1') as rf:
        r=csv.reader(rf)
        # next(r)
        data=[]
        for row in r:
            data.append(row)
        return data