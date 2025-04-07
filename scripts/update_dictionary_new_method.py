import pickle
import os
import numpy as np

## STATE DICT = THESE ARE THE CURRENTLY INITIAZLED EMBEDDINGS FOR ONE HOT ENCODED VALUES. 
## spelling and case matter here, if not the code will use the 'Other' embedding 
## float values theoretically have no limit, empty values are set to 0
companies = ['', 'Waters', 'Thermo', 'Agilent', 'Restek', 'Merck', 'Phenomenex', 'HILICON','GL','Advanced', 'Other']
USPs = ['', 'L1', 'L10', 'L109', 'L11', 'L43', 'L68', 'L3','L114', 'L112', 'L122', 'L7', 'L10', 'Other']
solvs = ['h2o','meoh', 'acn', 'iproh', 'Other']
HPLCs = ['RP', 'HILIC', 'PFP', 'Other']



method_name = 'test_method'

## This data is for RepoRT method 0135 as an example
company_name = 'Waters'
usp_code = 'L1'
col_length = 100 ## 0 for non-defined float values
col_innerdiam = 2.1
col_part_size = 1.7
temp = 0 ##
col_fl = 0.10000000149011612
col_dead = 1.1025 ## calculated by RepoRT
HPLC_type = 'RP'
A_solv = 'h2o'
B_solv = 'meoh'
## Gradient inflection points
time1 = 0.0
grad1 = 5.0
time2 = 5.0
grad2 = 5.0
time3 = 25.0
grad3 = 100.0
time4 = 25.0
grad4 = 100.0
A_pH = 3
B_pH = 3
A_start = 5
A_end = 0
B_start = 5
B_end = 0
## Additive concentrations and units (see re)
eluent_A_formic = 0.1
eluent_A_formic_unit = '%'
eluent_A_acetic = 0
eluent_A_acetic_unit =  ''
eluent_A_trifluoroacetic = 0
eluent_A_trifluoroacetic_unit = ''
eluent_A_phosphor = 0
eluent_A_phosphor_unit = ''
eluent_A_nh4ac = 0
eluent_A_nh4ac_unit = ''
eluent_A_nh4form = 0
eluent_A_nh4form_unit = ''
eluent_A_nh4carb = 0
eluent_A_nh4carb_unit = ''
eluent_A_nh4bicarb = 0
eluent_A_nh4bicarb_unit = ''
eluent_A_nh4f = 0
eluent_A_nh4f_unit = ''
eluent_A_nh4oh = 0
eluent_A_nh4oh_unit = ''
eluent_A_trieth = 0
eluent_A_trieth_unit = ''
eluent_A_triprop = 0
eluent_A_triprop_unit = ''
eluent_A_tribut = 0
eluent_A_tribut_unit = ''
eluent_A_nndimethylhex = 0
eluent_A_nndimethylhex_unit = '' 
eluent_A_medronic = 0
eluent_A_medronic_unit = ''
eluent_B_formic = 0.1
eluent_B_formic_unit = '%'
eluent_B_acetic = 0
eluent_B_acetic_unit = ''
eluent_B_trifluoroacetic = 0
eluent_B_trifluoroacetic_unit = ''
eluent_B_phosphor = 0
eluent_B_phosphor_unit = ''
eluent_B_nh4ac = 0
eluent_B_nh4ac_unit = ''
eluent_B_nh4form = 0
eluent_B_nh4form_unit = ''
eluent_B_nh4carb = 0
eluent_B_nh4carb_unit = ''
eluent_B_nh4bicarb = 0
eluent_B_nh4bicarb_unit = ''
eluent_B_nh4f = 0
eluent_B_nh4f_unit = ''
eluent_B_nh4oh = 0
eluent_B_nh4oh_unit = ''
eluent_B_trieth = 0
eluent_B_trieth_unit = ''
eluent_B_triprop = 0
eluent_B_triprop_unit = ''
eluent_B_tribut = 0
eluent_B_tribut_unit = ''
eluent_B_nndimethylhex = 0
eluent_B_nndimethylhex_unit = '' 
eluent_B_medronic = 0
eluent_B_medronic_unit = ''

## Tanaka parameters (calculated by RepoRT)
kPB = 3.99
alpha_CH2 = 1.47
alpha_T_O = 1.38
alpha_C_P = 0.37
alpha_B_P = 0.23
alpha_B_P1 = 0.09
## HSMB Parameters (calcualted by RepoRT)
particle_size = 1.7
pore_size = 130
H = 1.0
S_star = 0.028
A = -0.366
B = 0.007
C_pH_28 = 0.142
C_pH_7 = 0.088
EB_ret_factor = 6.4

column_params = [company_name, usp_code, col_length, col_innerdiam, col_part_size, temp, col_fl, col_dead, HPLC_type, A_solv, B_solv, time1, grad1, time2, grad2, time3, grad3, time4, grad4, \
                 A_pH, B_pH, A_start, A_end, B_start, B_end, eluent_A_formic, eluent_A_formic_unit, eluent_A_acetic, eluent_A_acetic_unit, eluent_A_trifluoroacetic, eluent_A_trifluoroacetic_unit, eluent_A_phosphor, eluent_A_phosphor_unit, eluent_A_nh4ac, eluent_A_nh4ac_unit, eluent_A_nh4form, eluent_A_nh4form_unit, eluent_A_nh4carb, eluent_A_nh4carb_unit, eluent_A_nh4bicarb, eluent_A_nh4bicarb_unit, eluent_A_nh4f, eluent_A_nh4f_unit, eluent_A_nh4oh, eluent_A_nh4oh_unit, eluent_A_trieth, eluent_A_trieth_unit, eluent_A_triprop, eluent_A_triprop_unit, eluent_A_tribut, eluent_A_tribut_unit, eluent_A_nndimethylhex, eluent_A_nndimethylhex_unit, eluent_A_medronic, eluent_A_medronic_unit, eluent_B_formic, eluent_B_formic_unit, eluent_B_acetic, eluent_B_acetic_unit, eluent_B_trifluoroacetic, eluent_B_trifluoroacetic_unit, eluent_B_phosphor, eluent_B_phosphor_unit,eluent_B_nh4ac ,eluent_B_nh4ac_unit ,eluent_B_nh4form ,eluent_B_nh4form_unit ,eluent_B_nh4carb ,eluent_B_nh4carb_unit ,eluent_B_nh4bicarb ,eluent_B_nh4bicarb_unit ,eluent_B_nh4f ,eluent_B_nh4f_unit ,eluent_B_nh4oh ,eluent_B_nh4oh_unit ,eluent_B_trieth ,eluent_B_trieth_unit ,eluent_B_triprop ,eluent_B_triprop_unit ,eluent_B_tribut ,eluent_B_tribut_unit ,eluent_B_nndimethylhex ,eluent_B_nndimethylhex_unit ,eluent_B_medronic ,eluent_B_medronic_unit, \
                 kPB ,alpha_CH2 ,alpha_T_O ,alpha_C_P ,alpha_B_P , alpha_B_P1 ,particle_size ,pore_size ,H ,S_star ,A ,B ,C_pH_28 ,C_pH_7 ,EB_ret_factor]

dict_path = '../sample_data/RP_metadata.pickle'
with open(dict_path, 'rb') as f:
    data = pickle.load(f)
header = data['1']
assert len(header) == len(column_params), f"Header length {len(header)} does not match column params length {len(column_params)}"

data[method_name] = column_params

with open(dict_path + '_updated', 'wb') as f:
    pickle.dump(data, f)