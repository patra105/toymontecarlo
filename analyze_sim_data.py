import numpy as np
import vector
import matplotlib.pyplot as plt
import mplhep as hep
import csv
import os, re, math, time
from dataset import DataSet
from bin_dataset import BinDataSet
from figagg import FigAgg

fig_agg = FigAgg()

##############################
### load and sort the data ###
##############################
data_dir = 'simulation_npz_data/'

npz_data_files = os.listdir(data_dir)

'''data file naming scheme --> mWR_2000_mN_1900_1000000events_sim_data.npz'''

mass_points_list = []

for data_file_name in npz_data_files:
    mWR_inds = re.search(r'mWR_\d+', data_file_name)
    mN_inds = re.search(r'mN_\d+', data_file_name)
    num_events_inds = re.search(r'\d+e', data_file_name)

    mWR = np.int64(data_file_name[mWR_inds.start()+4:mWR_inds.end()])
    mN  = np.int64(data_file_name[ mN_inds.start()+3: mN_inds.end()])
    num_events = np.int64(data_file_name[num_events_inds.start():num_events_inds.end()-1])

    mWR_in = False

    for i in range(len(mass_points_list)):
        mass_point = mass_points_list[i]

        if mass_point[0] == mWR:
            mWR_in = True
            point_ind = i

    if mWR_in == False:
        mass_points_list.append([mWR, [mN]])
    else:
        mass_points_list[point_ind][1].append(mN)

#print(f'mass_points_list: {mass_points_list}')

'''mass_points_list --> [[3200, [1, 3000, 800]], [2000, [1900, 1, 800]], [1200, [1100, 1, 200]]] (before sorting)'''

mWR_list = []

for i in range(len(mass_points_list)):
    mass_points = mass_points_list[i]

    mWR = mass_points[0]

    mWR_list.append(mWR)

mWR_list.sort()

mass_points_list_mWR_sort = []

for i in range(len(mWR_list)):
    mWR = mWR_list[i]

    for j in range(len(mass_points_list)):
        mass_points = mass_points_list[j]

        if mass_points[0] == mWR:
            mass_points_list_mWR_sort.append(mass_points)

#print(f'mass_points_list_mWR_sort: {mass_points_list_mWR_sort}')

'''mass_points_list_mWR_sort --> [[1200, [1100, 1, 200]], [2000, [1900, 1, 800]], [3200, [1, 3000, 800]]] (sorted by mWR)'''

for i in range(len(mass_points_list_mWR_sort)):
    mass_points_list_mWR_sort[i][1].sort()

mass_points_list_mN_sort = mass_points_list_mWR_sort

#print(f'mass_points_list_mN_sort: {mass_points_list_mN_sort}')

'''mass_points_list_mN_sort --> [[1200, [1, 200, 1100]], [2000, [1, 800, 1900]], [3200, [1, 800, 3000]]] (sorted by mWR and mN)'''

mass_points_flat = []

for i in range(len(mass_points_list_mN_sort)):
    mass_points = mass_points_list_mN_sort[i]

    for j in range(len(mass_points[1])):
        mass_points_flat.append([mass_points[0], mass_points[1][j]])

#print(f'mass_points_flat: {mass_points_flat}')

'''mass_points_flat --> [[1200, 1], [1200, 200], [1200, 1100], [2000, 1], [2000, 800], [2000, 1900], [3200, 1], [3200, 800], [3200, 3000]]'''

if len(mass_points_flat) != len(npz_data_files):
    print('>!>>>>>!!>>>>>>!>>>>!!!>> len(mass_points_flat) not equal to len(npz_data_files) <<<<!!!<<<<<<!<<<<!!<<')

sorted_npz_data_files = np.ones(len(npz_data_files)).tolist()

for data_file_name in npz_data_files:
    mWR_inds = re.search(r'mWR_\d+', data_file_name)
    mN_inds = re.search(r'mN_\d+', data_file_name)
    num_events_inds = re.search(r'\d+e', data_file_name)

    mWR = np.int64(data_file_name[mWR_inds.start()+4:mWR_inds.end()])
    mN  = np.int64(data_file_name[ mN_inds.start()+3: mN_inds.end()])
    num_events = np.int64(data_file_name[num_events_inds.start():num_events_inds.end()-1])

    mass_point = [mWR, mN]

    ind = mass_points_flat.index(mass_point)

    sorted_npz_data_files[ind] = data_file_name

#print(f'sorted_npz_data_files: {sorted_npz_data_files}')
##############################
##############################

def convert_to_4D_p_array(four_vec_array):

    dtype = [('E', float), ('px', float), ('py', float), ('pz', float)]

    struc_four_vec_array = four_vec_array.astype(float).view(dtype).reshape(-1)
    p_array_4D = struc_four_vec_array.view(vector.MomentumNumpy4D)

    return p_array_4D

def get_thetas(p_data_4D): # p_data_4D is a vector.backends.numpy.MomentumNumpy4D object (list of 4-momentum vectors)

    px_vals = p_data_4D.px # 1d flat numpy array of px values of each 4-momentum vector
    py_vals = p_data_4D.py
    thetas = np.arctan2(py_vals, px_vals) # theta is angle between transverse momentum vector and postive x-axis

    return thetas

def combine_into_4D_array(E_array, px_array, py_array, pz_array): # Thanks, ChatGPT!!!
    """
    Combine four 2D numpy arrays (E, px, py, pz) into a structured
    MomentumNumpy4D array of shape (N, M).
    """
    import numpy as np
    import vector

    # Stack components into shape (N, M, 4)
    stacked = np.stack([E_array, px_array, py_array, pz_array], axis=-1)

    # Prepare dtype for structured array
    dtype = [('E', float), ('px', float), ('py', float), ('pz', float)]

    # Flatten, convert, then reshape back
    struc = stacked.reshape(-1, 4).astype(float).view(dtype).reshape(-1)
    vecs = struc.view(vector.MomentumNumpy4D)

    return vecs.reshape(stacked.shape[:-1])

def scale_momentum(p_array_4D, scale_array):

    if scale_array.ndim == 1: # ith element of scale_array is scaling factor for 3-momentum of ith 4-vector
        N = len(p_array_4D)

        if len(scale_array) != N:
            raise ValueError(f'length of scale array ({len(scale_array)}) is not equal to length of 4-vector array ({N})')

        mass_array = p_array_4D.mass

        px_array = p_array_4D.px
        py_array = p_array_4D.py
        pz_array = p_array_4D.pz

        px_scaled_array = np.multiply(px_array, scale_array)
        py_scaled_array = np.multiply(py_array, scale_array)
        pz_scaled_array = np.multiply(pz_array, scale_array)

        p_scaled_array = np.sqrt(np.power(px_scaled_array, 2) + np.power(py_scaled_array, 2) + np.power(pz_scaled_array, 2))

        E_new_array = np.sqrt(np.power(p_scaled_array, 2) + np.power(mass_array, 2))

        p_array_4D_scaled = vector.obj(E = E_new_array, px = px_scaled_array, py = py_scaled_array, pz = pz_scaled_array)

    if scale_array.ndim == 2: # ith row of scale_array contains all M scaling factors by which to scale the ith 4-vector's 3-momentum
        N = len(p_array_4D)
        M = len(scale_array[0])

        ones_array = np.ones([N,M])

        mass_array = p_array_4D.mass

        px_array = p_array_4D.px
        py_array = p_array_4D.py
        pz_array = p_array_4D.pz

        mass_array_2d = mass_array[:, None]
        mass_array_full = mass_array_2d*ones_array

        px_array_2d = px_array[:, None]
        py_array_2d = py_array[:, None]
        pz_array_2d = pz_array[:, None]

        px_scaled_array = px_array_2d*scale_array
        py_scaled_array = py_array_2d*scale_array
        pz_scaled_array = pz_array_2d*scale_array

        p_scaled_array = np.sqrt(np.power(px_scaled_array, 2) + np.power(py_scaled_array, 2) + np.power(pz_scaled_array, 2))

        E_new_array = np.sqrt(np.power(p_scaled_array, 2) + np.power(mass_array_full, 2))

        p_array_4D_scaled = combine_into_4D_array(E_new_array, px_scaled_array, py_scaled_array, pz_scaled_array)

    return p_array_4D_scaled

def get_JER(p_array_4D):

    S = 0.92
    C = 0.04

    pt_array = p_array_4D.pt

    JER_array = np.multiply(pt_array, np.sqrt(np.divide(S**2, pt_array) + C**2))

    return JER_array

def analyze_data(data_dir, data_file_name, fig_agg, **kwargs):

    #####################################
    ### set the values for the kwargs ###
    #####################################
    keyword_vals_dict = {
                         'make_pt_figs': True,
                         'make_mass_figs': True,
                         'make_eta_figs': True,
                         'make_phi_figs': True,
                         'make_smear_frac_figs': True,
                         'make_lW_pt_aligned_figs': True,
                         'make_jet_p_correction_fac_figs': True,
                         'make_scaled_jet_var_figs': True,
                         'analyze_tails': True,
                         'plot_tail_bounds': True
                        }

    keyword_vals_dict.update(kwargs)

    keyword_strings_tuple = (
                             'make_pt_figs',
                             'make_mass_figs',
                             'make_eta_figs',
                             'make_phi_figs',
                             'make_smear_frac_figs',
                             'make_lW_pt_aligned_figs',
                             'make_jet_p_correction_fac_figs',
                             'make_scaled_jet_var_figs',
                             'analyze_tails',
                             'plot_tail_bounds'
                            )

    for keyword in keyword_vals_dict.keys():
        if keyword not in keyword_strings_tuple:
            raise ValueError(f'Unrecognized keyword --> {keyword} <-- has been passed to analyze_data()')
    
    keyword_vals_tuple = tuple(keyword_vals_dict[keyword_string] for keyword_string in keyword_strings_tuple)

    make_pt_figs = keyword_vals_tuple[0]
    make_mass_figs = keyword_vals_tuple[1]
    make_eta_figs = keyword_vals_tuple[2]
    make_phi_figs = keyword_vals_tuple[3]
    make_smear_frac_figs = keyword_vals_tuple[4]
    make_lW_pt_aligned_figs = keyword_vals_tuple[5]
    make_jet_p_correction_fac_figs = keyword_vals_tuple[6]
    make_scaled_jet_var_figs = keyword_vals_tuple[7]
    analyze_tails = keyword_vals_tuple[8]
    plot_tail_bounds = keyword_vals_tuple[9]
    #####################################
    #####################################

    ############################
    ### load simulation data ###
    ############################
    # data file naming scheme --> mWR_2000_mN_1900_1000000events_sim_data.npz

    mWR_inds = re.search(r'mWR_\d+', data_file_name)
    mN_inds = re.search(r'mN_\d+', data_file_name)
    num_events_inds = re.search(r'\d+e', data_file_name)

    mWR = np.int64(data_file_name[mWR_inds.start()+4:mWR_inds.end()])
    mN  = np.int64(data_file_name[ mN_inds.start()+3: mN_inds.end()])
    num_events = np.int64(data_file_name[num_events_inds.start():num_events_inds.end()-1])

    data = np.load(data_dir + data_file_name)

    lW_data         = data['lW_data']
    lN_data         = data['lN_data']
    j1_data         = data['j1_data']
    j2_data         = data['j2_data']
    lW_data_smeared = data['lW_data_smeared']
    lN_data_smeared = data['lN_data_smeared']
    j1_data_smeared = data['j1_data_smeared']
    j2_data_smeared = data['j2_data_smeared']

#    print(f'\nlW_data:\n{lW_data}')
#    print(f'\nlN_data:\n{lN_data}')
#    print(f'\nj1_data:\n{j1_data}')
#    print(f'\nj2_data:\n{j2_data}')
#    print('-------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n\n')
    ############################
    ############################

    ###############################
    ### analyze simulation data ###
    ###############################
    lW_4D_p_data         = convert_to_4D_p_array(lW_data)
    lN_4D_p_data         = convert_to_4D_p_array(lN_data)
    j1_4D_p_data         = convert_to_4D_p_array(j1_data)
    j2_4D_p_data         = convert_to_4D_p_array(j2_data)
    lW_4D_p_data_smeared = convert_to_4D_p_array(lW_data_smeared)
    lN_4D_p_data_smeared = convert_to_4D_p_array(lN_data_smeared)
    j1_4D_p_data_smeared = convert_to_4D_p_array(j1_data_smeared)
    j2_4D_p_data_smeared = convert_to_4D_p_array(j2_data_smeared)
#    print(f'--------------------------------------------------{type(j2_4D_p_data_smeared)}-------------------------------------------------------')

    lWlN_4D_p_data             = lW_4D_p_data         + lN_4D_p_data                                                       # dilepton data
    j1j2_4D_p_data             =                                               j1_4D_p_data         + j2_4D_p_data         # dijet data
    lNj1j2_4D_p_data           =                        lN_4D_p_data         + j1_4D_p_data         + j2_4D_p_data         # threeobject data
    lWlNj1j2_4D_p_data         = lW_4D_p_data         + lN_4D_p_data         + j1_4D_p_data         + j2_4D_p_data         # fourobject data
    lWlN_4D_p_data_smeared     = lW_4D_p_data_smeared + lN_4D_p_data_smeared                                               # dilepton data smeared
    j1j2_4D_p_data_smeared     =                                               j1_4D_p_data_smeared + j2_4D_p_data_smeared # dijet data smeared
    lNj1j2_4D_p_data_smeared   =                        lN_4D_p_data_smeared + j1_4D_p_data_smeared + j2_4D_p_data_smeared # threeobject data smeared
    lWlNj1j2_4D_p_data_smeared = lW_4D_p_data_smeared + lN_4D_p_data_smeared + j1_4D_p_data_smeared + j2_4D_p_data_smeared # fourobject data smeared


    lW_pt_data               =               lW_4D_p_data.pt
    lN_pt_data               =               lN_4D_p_data.pt
    j1_pt_data               =               j1_4D_p_data.pt
    j2_pt_data               =               j2_4D_p_data.pt
    j1j2_pt_data             =             j1j2_4D_p_data.pt
    lNj1j2_pt_data           =           lNj1j2_4D_p_data.pt
    lWlNj1j2_pt_data         =         lWlNj1j2_4D_p_data.pt
    lW_pt_data_smeared       =       lW_4D_p_data_smeared.pt
    lN_pt_data_smeared       =       lN_4D_p_data_smeared.pt
    j1_pt_data_smeared       =       j1_4D_p_data_smeared.pt
    j2_pt_data_smeared       =       j2_4D_p_data_smeared.pt
    j1j2_pt_data_smeared     =     j1j2_4D_p_data_smeared.pt
    lNj1j2_pt_data_smeared   =   lNj1j2_4D_p_data_smeared.pt
    lWlNj1j2_pt_data_smeared = lWlNj1j2_4D_p_data_smeared.pt

    lWlN_mass_data             =             lWlN_4D_p_data.mass
    j1j2_mass_data             =             j1j2_4D_p_data.mass
    lNj1j2_mass_data           =           lNj1j2_4D_p_data.mass
    lWlNj1j2_mass_data         =         lWlNj1j2_4D_p_data.mass
    lWlN_mass_data_smeared     =     lWlN_4D_p_data_smeared.mass
    j1j2_mass_data_smeared     =     j1j2_4D_p_data_smeared.mass
    lNj1j2_mass_data_smeared   =   lNj1j2_4D_p_data_smeared.mass
    lWlNj1j2_mass_data_smeared = lWlNj1j2_4D_p_data_smeared.mass

    lW_eta_data         =         lW_4D_p_data.eta
    lN_eta_data         =         lN_4D_p_data.eta
    j1_eta_data         =         j1_4D_p_data.eta
    j2_eta_data         =         j2_4D_p_data.eta
    lW_eta_data_smeared = lW_4D_p_data_smeared.eta
    lN_eta_data_smeared = lN_4D_p_data_smeared.eta
    j1_eta_data_smeared = j1_4D_p_data_smeared.eta
    j2_eta_data_smeared = j2_4D_p_data_smeared.eta

    lW_phi_data         =         lW_4D_p_data.phi
    lN_phi_data         =         lN_4D_p_data.phi
    j1_phi_data         =         j1_4D_p_data.phi
    j2_phi_data         =         j2_4D_p_data.phi
    lW_phi_data_smeared = lW_4D_p_data_smeared.phi
    lN_phi_data_smeared = lN_4D_p_data_smeared.phi
    j1_phi_data_smeared = j1_4D_p_data_smeared.phi
    j2_phi_data_smeared = j2_4D_p_data_smeared.phi

    lW_e_data         =         lW_4D_p_data.e
    lN_e_data         =         lN_4D_p_data.e
    j1_e_data         =         j1_4D_p_data.e
    j2_e_data         =         j2_4D_p_data.e
    lW_e_data_smeared = lW_4D_p_data_smeared.e
    lN_e_data_smeared = lN_4D_p_data_smeared.e
    j1_e_data_smeared = j1_4D_p_data_smeared.e
    j2_e_data_smeared = j2_4D_p_data_smeared.e
    lW_e_data_smear_frac = np.divide(lW_e_data_smeared, lW_e_data) - 1
    lN_e_data_smear_frac = np.divide(lN_e_data_smeared, lN_e_data) - 1
    j1_e_data_smear_frac = np.divide(j1_e_data_smeared, j1_e_data) - 1
    j2_e_data_smear_frac = np.divide(j2_e_data_smeared, j2_e_data) - 1
    ###############################
    ###############################

    ##################################
    ### make lW pt aligned vectors ###
    ##################################
    thetas = get_thetas(lW_4D_p_data)

    lW_4D_p_data_lW_align         = lW_4D_p_data.rotateZ(-1*thetas)
    lN_4D_p_data_lW_align         = lN_4D_p_data.rotateZ(-1*thetas)
    j1_4D_p_data_lW_align         = j1_4D_p_data.rotateZ(-1*thetas)
    j2_4D_p_data_lW_align         = j2_4D_p_data.rotateZ(-1*thetas)
    lW_4D_p_data_smeared_lW_align = lW_4D_p_data_smeared.rotateZ(-1*thetas)
    lN_4D_p_data_smeared_lW_align = lN_4D_p_data_smeared.rotateZ(-1*thetas)
    j1_4D_p_data_smeared_lW_align = j1_4D_p_data_smeared.rotateZ(-1*thetas)
    j2_4D_p_data_smeared_lW_align = j2_4D_p_data_smeared.rotateZ(-1*thetas)

    lNj1j2_4D_p_data_lW_align         = lN_4D_p_data_lW_align         + j1_4D_p_data_lW_align         + j2_4D_p_data_lW_align # threeobject data
    lNj1j2_4D_p_data_smeared_lW_align = lN_4D_p_data_smeared_lW_align + j1_4D_p_data_smeared_lW_align + j2_4D_p_data_smeared_lW_align

    lWlNj1j2_4D_p_data_lW_align         = lW_4D_p_data_lW_align         + lN_4D_p_data_lW_align         + j1_4D_p_data_lW_align         + j2_4D_p_data_lW_align # fourobject data
    lWlNj1j2_4D_p_data_smeared_lW_align = lW_4D_p_data_smeared_lW_align + lN_4D_p_data_smeared_lW_align + j1_4D_p_data_smeared_lW_align + j2_4D_p_data_smeared_lW_align
    ##################################
    ##################################

    #############################################
    ### make variables for lW aligned vectors ###
    #############################################
    j1_pt_data_smeared_lW_align = j1_4D_p_data_lW_align.pt
    j2_pt_data_smeared_lW_align = j2_4D_p_data_lW_align.pt

    lW_p_par_data = lW_4D_p_data_lW_align.px # momentum component parallel to lW pt vector
    lN_p_par_data = lN_4D_p_data_lW_align.px
    j1_p_par_data = j1_4D_p_data_lW_align.px
    j2_p_par_data = j2_4D_p_data_lW_align.px
    lNj1j2_p_par_data = lNj1j2_4D_p_data_lW_align.px
    lWlNj1j2_p_par_data = lWlNj1j2_4D_p_data_lW_align.px

    lW_p_perp_data = lW_4D_p_data_lW_align.py # momentum component perpendicular to lW pt vector
    lN_p_perp_data = lN_4D_p_data_lW_align.py
    j1_p_perp_data = j1_4D_p_data_lW_align.py
    j2_p_perp_data = j2_4D_p_data_lW_align.py
    lNj1j2_p_perp_data = lNj1j2_4D_p_data_lW_align.py
    lWlNj1j2_p_perp_data = lWlNj1j2_4D_p_data_lW_align.py

    lW_p_par_to_pt_data = np.divide(lW_p_par_data, lW_pt_data)
    lN_p_par_to_pt_data = np.divide(lN_p_par_data, lN_pt_data)
    j1_p_par_to_pt_data = np.divide(j1_p_par_data, j1_pt_data)
    j2_p_par_to_pt_data = np.divide(j2_p_par_data, j2_pt_data)
    lNj1j2_p_par_to_pt_data = np.divide(lNj1j2_p_par_data, lNj1j2_pt_data)
#    lWlNj1j2_p_par_to_pt_data = np.divide(lWlNj1j2_p_par_data, lWlNj1j2_pt_data)

    lW_p_perp_to_pt_data = np.divide(lW_p_perp_data, lW_pt_data)
    lN_p_perp_to_pt_data = np.divide(lN_p_perp_data, lN_pt_data)
    j1_p_perp_to_pt_data = np.divide(j1_p_perp_data, j1_pt_data)
    j2_p_perp_to_pt_data = np.divide(j2_p_perp_data, j2_pt_data)
    lNj1j2_p_perp_to_pt_data = np.divide(lNj1j2_p_perp_data, lNj1j2_pt_data)
#    lWlNj1j2_p_perp_to_pt_data = np.divide(lWlNj1j2_p_perp_data, lWlNj1j2_pt_data)

    lW_p_perp_to_par_data = np.divide(lW_p_perp_data, lW_p_par_data)
    lN_p_perp_to_par_data = np.divide(lN_p_perp_data, lN_p_par_data)
    j1_p_perp_to_par_data = np.divide(j1_p_perp_data, j1_p_par_data)
    j2_p_perp_to_par_data = np.divide(j2_p_perp_data, j2_p_par_data)
    lNj1j2_p_perp_to_par_data = np.divide(lNj1j2_p_perp_data, lNj1j2_p_par_data)
#    lWlNj1j2_p_perp_to_par_data = np.divide(lWlNj1j2_p_perp_data, lWlNj1j2_p_par_data)


    lW_p_par_data_smeared = lW_4D_p_data_smeared_lW_align.px # momentum component parallel to lW pt vector
    lN_p_par_data_smeared = lN_4D_p_data_smeared_lW_align.px
    j1_p_par_data_smeared = j1_4D_p_data_smeared_lW_align.px
    j2_p_par_data_smeared = j2_4D_p_data_smeared_lW_align.px
    lNj1j2_p_par_data_smeared = lNj1j2_4D_p_data_smeared_lW_align.px
    lWlNj1j2_p_par_data_smeared = lWlNj1j2_4D_p_data_smeared_lW_align.px

    lW_p_perp_data_smeared = lW_4D_p_data_smeared_lW_align.py # momentum component perpendicular to lW pt vector
    lN_p_perp_data_smeared = lN_4D_p_data_smeared_lW_align.py
    j1_p_perp_data_smeared = j1_4D_p_data_smeared_lW_align.py
    j2_p_perp_data_smeared = j2_4D_p_data_smeared_lW_align.py
    lNj1j2_p_perp_data_smeared = lNj1j2_4D_p_data_smeared_lW_align.py
    lWlNj1j2_p_perp_data_smeared = lWlNj1j2_4D_p_data_smeared_lW_align.py

    lW_p_par_to_pt_data_smeared = np.divide(lW_p_par_data_smeared, lW_pt_data_smeared)
    lN_p_par_to_pt_data_smeared = np.divide(lN_p_par_data_smeared, lN_pt_data_smeared)
    j1_p_par_to_pt_data_smeared = np.divide(j1_p_par_data_smeared, j1_pt_data_smeared)
    j2_p_par_to_pt_data_smeared = np.divide(j2_p_par_data_smeared, j2_pt_data_smeared)
    lNj1j2_p_par_to_pt_data_smeared = np.divide(lNj1j2_p_par_data_smeared, lNj1j2_pt_data_smeared)
    lWlNj1j2_p_par_to_pt_data_smeared = np.divide(lWlNj1j2_p_par_data_smeared, lWlNj1j2_pt_data_smeared)

    lW_p_perp_to_pt_data_smeared = np.divide(lW_p_perp_data_smeared, lW_pt_data_smeared)
    lN_p_perp_to_pt_data_smeared = np.divide(lN_p_perp_data_smeared, lN_pt_data_smeared)
    j1_p_perp_to_pt_data_smeared = np.divide(j1_p_perp_data_smeared, j1_pt_data_smeared)
    j2_p_perp_to_pt_data_smeared = np.divide(j2_p_perp_data_smeared, j2_pt_data_smeared)
    lNj1j2_p_perp_to_pt_data_smeared = np.divide(lNj1j2_p_perp_data_smeared, lNj1j2_pt_data_smeared)
    lWlNj1j2_p_perp_to_pt_data_smeared = np.divide(lWlNj1j2_p_perp_data_smeared, lWlNj1j2_pt_data_smeared)

    lW_p_perp_to_par_data_smeared = np.divide(lW_p_perp_data_smeared, lW_p_par_data_smeared)
    lN_p_perp_to_par_data_smeared = np.divide(lN_p_perp_data_smeared, lN_p_par_data_smeared)
    j1_p_perp_to_par_data_smeared = np.divide(j1_p_perp_data_smeared, j1_p_par_data_smeared)
    j2_p_perp_to_par_data_smeared = np.divide(j2_p_perp_data_smeared, j2_p_par_data_smeared)
    lNj1j2_p_perp_to_par_data_smeared = np.divide(lNj1j2_p_perp_data_smeared, lNj1j2_p_par_data_smeared)
    lWlNj1j2_p_perp_to_par_data_smeared = np.divide(lWlNj1j2_p_perp_data_smeared, lWlNj1j2_p_par_data_smeared)

    lWlNj1j2_mass_lW_align = lWlNj1j2_4D_p_data_lW_align.mass
    lWlNj1j2_mass_smeared_lW_align = lWlNj1j2_4D_p_data_smeared_lW_align.mass
    #############################################
    #############################################

    ###########################################
    ### making jet momentum scaling factors ###
    ###########################################
    lowest_s_1 = 0
    highest_s_1 = 3

    s_1 = np.linspace(lowest_s_1, highest_s_1, 101)

    M = len(s_1)
    N = len(lWlNj1j2_mass_smeared_lW_align)

    s_1_2d = s_1[None, :]

    lW_p_par_data_smeared_2d = lW_p_par_data_smeared[:, None]
    lN_p_par_data_smeared_2d = lN_p_par_data_smeared[:, None]
    j1_p_par_data_smeared_2d = j1_p_par_data_smeared[:, None]

    j2_p_par_data_smeared_recip = np.divide(1, j2_p_par_data_smeared)
    j2_p_par_data_smeared_recip_2d = j2_p_par_data_smeared_recip[:, None]

    s_2 = -(lW_p_par_data_smeared_2d + lN_p_par_data_smeared_2d + s_1_2d*j1_p_par_data_smeared_2d)*j2_p_par_data_smeared_recip_2d

    '''                  s_2 is an N x M array, where N is the number of events for the current (mWR, mN) point and M is the length of the s_1 list
    s_2 =
    [
     [s_2(s_1[0]), s_2(s_1[1]), s_2(s_1[2]), ..., s_2(s_1[M-1])] <-- event 0
     [s_2(s_1[0]), s_2(s_1[1]), s_2(s_1[2]), ..., s_2(s_1[M-1])] <-- event 1
     [s_2(s_1[0]), s_2(s_1[1]), s_2(s_1[2]), ..., s_2(s_1[M-1])] <-- event 2
                                    .
                                    .
                                    .
     [s_2(s_1[0]), s_2(s_1[1]), s_2(s_1[2]), ..., s_2(s_1[M-1])] <-- event N-1
    ]
    '''
#    print('\n'*8)
#    print(f'N = {N}')
#    print(f'M = {M}')
#    print(f'\ns_2:\n{s_2}')
#    print(f'len(s_2): {len(s_2)}')
#    print(f'len(s_2[0]): {len(s_2[0])}')
#    print(f'len(s_2[32]): {len(s_2[32])}')
#    print('\n'*8)
    ###########################################
    ###########################################

    ################################
    ### make scaling chi^2 array ###
    ################################
    ones_array_2d = np.ones([N, M])

    s_1_2d = s_1_2d*ones_array_2d # make s_1_2d into an N x M array

    j1_4D_p_data_smeared_scaled = scale_momentum(j1_4D_p_data_smeared_lW_align, s_1_2d)
    j2_4D_p_data_smeared_scaled = scale_momentum(j2_4D_p_data_smeared_lW_align, s_2)

    '''                  j1_4D_p_data_smeared_scaled is an N x M array, where N is the number of events for the current (mWR, mN) point and M is the length of the s_1 list
                         Here, * means multiplication of the 3-momentum and subsequent correction of the energy
    j1_4D_p_data_smeared_scaled =
    [
     [j1*s_1[0], j1*s_1[1], j1*s_1[2], ..., j1*s_1[M-1]] <-- event 0
     [j1*s_1[0], j1*s_1[1], j1*s_1[2], ..., j1*s_1[M-1]] <-- event 1
     [j1*s_1[0], j1*s_1[1], j1*s_1[2], ..., j1*s_1[M-1]] <-- event 2
                                    .
                                    .
                                    .
     [j1*s_1[0], j1*s_1[1], j1*s_1[2], ..., j1*s_1[M-1]] <-- event N-1
    ]
    '''

    '''                  j2_4D_p_data_smeared_scaled is an N x M array, where N is the number of events for the current (mWR, mN) point and M is the length of the s_1 list
                         Here, * means multiplication of the 3-momentum and subsequent correction of the energy
    j2_4D_p_data_smeared_scaled =
    [
     [j2*s_2[0], j2*s_2[1], j2*s_2[2], ..., j2*s_2[M-1]] <-- event 0
     [j2*s_2[0], j2*s_2[1], j2*s_2[2], ..., j2*s_2[M-1]] <-- event 1
     [j2*s_2[0], j2*s_2[1], j2*s_2[2], ..., j2*s_2[M-1]] <-- event 2
                                    .
                                    .
                                    .
     [j2*s_2[0], j2*s_2[1], j2*s_2[2], ..., j2*s_2[M-1]] <-- event N-1
    ]
    '''

    j1_pt_data_smeared_2d = j1_pt_data_smeared_lW_align[:, None] # pre-scaling pt of the smeared jets
    j2_pt_data_smeared_2d = j2_pt_data_smeared_lW_align[:, None]
    
    j1_pt_data_smeared_scaled = j1_4D_p_data_smeared_scaled.pt # post-scaling pt of the smeared jets
    j2_pt_data_smeared_scaled = j2_4D_p_data_smeared_scaled.pt

    j1_JER_array = get_JER(j1_4D_p_data_smeared_lW_align) # JER of the pre-scaled jets
    j2_JER_array = get_JER(j2_4D_p_data_smeared_lW_align)

    j1_JER_array_recip = np.divide(1, j1_JER_array)
    j2_JER_array_recip = np.divide(1, j2_JER_array)

    j1_JER_array_recip_2d = j1_JER_array_recip[:, None]
    j2_JER_array_recip_2d = j2_JER_array_recip[:, None]

    j1_chi_sq_array = np.power(np.multiply(j1_pt_data_smeared_scaled - j1_pt_data_smeared_2d, j1_JER_array_recip_2d), 2)
    j2_chi_sq_array = np.power(np.multiply(j2_pt_data_smeared_scaled - j2_pt_data_smeared_2d, j2_JER_array_recip_2d), 2)

    chi_sq_array = j1_chi_sq_array + j2_chi_sq_array

    '''                  chi_sq_array is an N x M array, where N is the number of events for the current (mWR, mN) point and M is the length of the s_1 list
    chi_sq_array =
    [
     [chi_sq(s_1[0],s_2[0][0]),   chi_sq(s_1[1],s_2[0][1]),   chi_sq(s_1[2],s_2[0][2]),   ..., chi_sq(s_1[M-1],s_2[0][M-1])] <-- event 0
     [chi_sq(s_1[0],s_2[1][0]),   chi_sq(s_1[1],s_2[1][1]),   chi_sq(s_1[2],s_2[1][2]),   ..., chi_sq(s_1[M-1],s_2[1][M-1])] <-- event 1
     [chi_sq(s_1[0],s_2[2][0]),   chi_sq(s_1[1],s_2[2][1]),   chi_sq(s_1[2],s_2[2][2]),   ..., chi_sq(s_1[M-1],s_2[2][M-1])] <-- event 2
                                    .
                                    .
                                    .
     [chi_sq(s_1[0],s_2[N-1][0]), chi_sq(s_1[1],s_2[N-1][1]), chi_sq(s_1[2],s_2[N-1][2]), ..., chi_sq(s_1[M-1],s_2[N-1][M-1])] <-- event N-1
    ]
    '''

    min_chi_sq_array = np.min(chi_sq_array, axis = 1) # flat 1d array of length N where each element is the minimum chi^2(s_1, s_2) for that event
    print(f'----------------------------------------------len(min_chi_sq_array): {len(min_chi_sq_array)}--------------------------------------------------')

    min_chi_sq_inds = np.argmin(chi_sq_array, axis=1) # flat 1d array of length N where each element is the index of the minimum chi^2(s_1, s_2) for that event

    best_s_1_array = s_1_2d[np.arange(s_1_2d.shape[0]), min_chi_sq_inds]
    best_s_2_array = s_2[np.arange(s_2.shape[0]), min_chi_sq_inds]

    low_mask = (best_s_1_array == lowest_s_1)
    low_s_1_array = best_s_1_array[low_mask]

    high_mask = (best_s_1_array == highest_s_1)
    high_s_1_array = best_s_1_array[high_mask]

    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> low_s_1_array (len: {len(low_s_1_array)}):\n{low_s_1_array}')
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> high_s_1_array (len: {len(high_s_1_array)}):\n{high_s_1_array}')
    ################################
    ################################

    #######################
    ### scaling results ###
    #######################
    j1_4D_p_data_smeared_scaled_best = j1_4D_p_data_smeared_scaled[np.arange(j1_4D_p_data_smeared_scaled.shape[0]), min_chi_sq_inds]
    j2_4D_p_data_smeared_scaled_best = j2_4D_p_data_smeared_scaled[np.arange(j2_4D_p_data_smeared_scaled.shape[0]), min_chi_sq_inds]

    lNj1j2_4D_p_data_smeared_lW_align_scaled   =                                 lN_4D_p_data_smeared_lW_align + j1_4D_p_data_smeared_scaled_best + j2_4D_p_data_smeared_scaled_best
    lWlNj1j2_4D_p_data_smeared_lW_align_scaled = lW_4D_p_data_smeared_lW_align + lN_4D_p_data_smeared_lW_align + j1_4D_p_data_smeared_scaled_best + j2_4D_p_data_smeared_scaled_best

    lNj1j2_p_par_data_smeared_scaled    =   lNj1j2_4D_p_data_smeared_lW_align_scaled.px
    lWlNj1j2_p_par_data_smeared_scaled  = lWlNj1j2_4D_p_data_smeared_lW_align_scaled.px
    lWlNj1j2_p_perp_data_smeared_scaled = lWlNj1j2_4D_p_data_smeared_lW_align_scaled.py

    lNj1j2_scaled_p_par_to_lW_p_par = np.divide(lNj1j2_p_par_data_smeared_scaled, lW_p_par_data_smeared)
    lNj1j2_old_p_par_to_lW_p_par = np.divide(lNj1j2_p_par_data_smeared, lW_p_par_data_smeared) # pre-scaling 3 obj p_par over lW p_par

    lNj1j2_mass_smeared_scaled = lNj1j2_4D_p_data_smeared_lW_align_scaled.mass
    lWlNj1j2_mass_smeared_scaled = lWlNj1j2_4D_p_data_smeared_lW_align_scaled.mass
    #######################
    #######################

    ############################################
    ### filtering data into analysis regions ###
    ############################################
#    lWlN_SR_mass_mask = lWlN_mass_data >= 400
#    lWlN_DYCR_mass_mask = (lWlN_mass_data >= 60) & (lWlN_mass_data <= 150)
#
#    lW_pt_SR_data       =       lW_pt_data[lWlN_SR_mass_mask]
#    lN_pt_SR_data       =       lN_pt_data[lWlN_SR_mass_mask]
#    j1_pt_SR_data       =       j1_pt_data[lWlN_SR_mass_mask]
#    j2_pt_SR_data       =       j2_pt_data[lWlN_SR_mass_mask]
#    j1j2_pt_SR_data     =     j1j2_pt_data[lWlN_SR_mass_mask]
#    lNj1j2_pt_SR_data   =   lNj1j2_pt_data[lWlN_SR_mass_mask]
#    lWlNj1j2_pt_SR_data = lWlNj1j2_pt_data[lWlN_SR_mass_mask]
#
#    j1j2_mass_SR_data     =     j1j2_mass_data[lWlN_SR_mass_mask]
#    lNj1j2_mass_SR_data   =   lNj1j2_mass_data[lWlN_SR_mass_mask]
#    lWlNj1j2_mass_SR_data = lWlNj1j2_mass_data[lWlN_SR_mass_mask]
#
#    lW_eta_SR_data = lW_eta_data[lWlN_SR_mass_mask]
#    lN_eta_SR_data = lN_eta_data[lWlN_SR_mass_mask]
#    j1_eta_SR_data = j1_eta_data[lWlN_SR_mass_mask]
#    j2_eta_SR_data = j2_eta_data[lWlN_SR_mass_mask]
#
#    lW_phi_SR_data = lW_phi_data[lWlN_SR_mass_mask]
#    lN_phi_SR_data = lN_phi_data[lWlN_SR_mass_mask]
#    j1_phi_SR_data = j1_phi_data[lWlN_SR_mass_mask]
#    j2_phi_SR_data = j2_phi_data[lWlN_SR_mass_mask]
#
#
#    lW_pt_DYCR_data       =       lW_pt_data[lWlN_DYCR_mass_mask]
#    lN_pt_DYCR_data       =       lN_pt_data[lWlN_DYCR_mass_mask]
#    j1_pt_DYCR_data       =       j1_pt_data[lWlN_DYCR_mass_mask]
#    j2_pt_DYCR_data       =       j2_pt_data[lWlN_DYCR_mass_mask]
#    j1j2_pt_DYCR_data     =     j1j2_pt_data[lWlN_DYCR_mass_mask]
#    lNj1j2_pt_DYCR_data   =   lNj1j2_pt_data[lWlN_DYCR_mass_mask]
#    lWlNj1j2_pt_DYCR_data = lWlNj1j2_pt_data[lWlN_DYCR_mass_mask]
#
#    j1j2_mass_DYCR_data     =     j1j2_mass_data[lWlN_DYCR_mass_mask]
#    lNj1j2_mass_DYCR_data   =   lNj1j2_mass_data[lWlN_DYCR_mass_mask]
#    lWlNj1j2_mass_DYCR_data = lWlNj1j2_mass_data[lWlN_DYCR_mass_mask]
#
#    lW_eta_DYCR_data = lW_eta_data[lWlN_DYCR_mass_mask]
#    lN_eta_DYCR_data = lN_eta_data[lWlN_DYCR_mass_mask]
#    j1_eta_DYCR_data = j1_eta_data[lWlN_DYCR_mass_mask]
#    j2_eta_DYCR_data = j2_eta_data[lWlN_DYCR_mass_mask]
#
#    lW_phi_DYCR_data = lW_phi_data[lWlN_DYCR_mass_mask]
#    lN_phi_DYCR_data = lN_phi_data[lWlN_DYCR_mass_mask]
#    j1_phi_DYCR_data = j1_phi_data[lWlN_DYCR_mass_mask]
#    j2_phi_DYCR_data = j2_phi_data[lWlN_DYCR_mass_mask]
    ############################################
    ############################################

    ############################
    ### make DataSet objects ###
    ############################
    dataset_lW_pt               = DataSet(datalist_1 =       lW_pt_data)
    dataset_lN_pt               = DataSet(datalist_1 =       lN_pt_data)
    dataset_j1_pt               = DataSet(datalist_1 =       j1_pt_data)
    dataset_j2_pt               = DataSet(datalist_1 =       j2_pt_data)
    dataset_j1j2_pt             = DataSet(datalist_1 =     j1j2_pt_data)
    dataset_lNj1j2_pt           = DataSet(datalist_1 =   lNj1j2_pt_data)
    dataset_lWlNj1j2_pt         = DataSet(datalist_1 = lWlNj1j2_pt_data)
    dataset_lW_pt_smeared       = DataSet(datalist_1 =       lW_pt_data_smeared)
    dataset_lN_pt_smeared       = DataSet(datalist_1 =       lN_pt_data_smeared)
    dataset_j1_pt_smeared       = DataSet(datalist_1 =       j1_pt_data_smeared)
    dataset_j2_pt_smeared       = DataSet(datalist_1 =       j2_pt_data_smeared)
    dataset_j1j2_pt_smeared     = DataSet(datalist_1 =     j1j2_pt_data_smeared)
    dataset_lNj1j2_pt_smeared   = DataSet(datalist_1 =   lNj1j2_pt_data_smeared)
    dataset_lWlNj1j2_pt_smeared = DataSet(datalist_1 = lWlNj1j2_pt_data_smeared)

    dataset_j1j2_mass             = DataSet(datalist_1 =     j1j2_mass_data)
    dataset_lNj1j2_mass           = DataSet(datalist_1 =   lNj1j2_mass_data)
    dataset_lWlNj1j2_mass         = DataSet(datalist_1 = lWlNj1j2_mass_data)
    dataset_j1j2_mass_smeared     = DataSet(datalist_1 =     j1j2_mass_data_smeared)
    dataset_lNj1j2_mass_smeared   = DataSet(datalist_1 =   lNj1j2_mass_data_smeared)
    dataset_lWlNj1j2_mass_smeared = DataSet(datalist_1 = lWlNj1j2_mass_data_smeared)

    dataset_lW_eta         = DataSet(datalist_1 = lW_eta_data)
    dataset_lN_eta         = DataSet(datalist_1 = lN_eta_data)
    dataset_j1_eta         = DataSet(datalist_1 = j1_eta_data)
    dataset_j2_eta         = DataSet(datalist_1 = j2_eta_data)
    dataset_lW_eta_smeared = DataSet(datalist_1 = lW_eta_data_smeared)
    dataset_lN_eta_smeared = DataSet(datalist_1 = lN_eta_data_smeared)
    dataset_j1_eta_smeared = DataSet(datalist_1 = j1_eta_data_smeared)
    dataset_j2_eta_smeared = DataSet(datalist_1 = j2_eta_data_smeared)

    dataset_lW_phi         = DataSet(datalist_1 = lW_phi_data)
    dataset_lN_phi         = DataSet(datalist_1 = lN_phi_data)
    dataset_j1_phi         = DataSet(datalist_1 = j1_phi_data)
    dataset_j2_phi         = DataSet(datalist_1 = j2_phi_data)
    dataset_lW_phi_smeared = DataSet(datalist_1 = lW_phi_data_smeared)
    dataset_lN_phi_smeared = DataSet(datalist_1 = lN_phi_data_smeared)
    dataset_j1_phi_smeared = DataSet(datalist_1 = j1_phi_data_smeared)
    dataset_j2_phi_smeared = DataSet(datalist_1 = j2_phi_data_smeared)

    dataset_lW_e_smear_frac = DataSet(datalist_1 = lW_e_data_smear_frac)
    dataset_lN_e_smear_frac = DataSet(datalist_1 = lN_e_data_smear_frac)
    dataset_j1_e_smear_frac = DataSet(datalist_1 = j1_e_data_smear_frac)
    dataset_j2_e_smear_frac = DataSet(datalist_1 = j2_e_data_smear_frac)


    dataset_lW_p_par       = DataSet(datalist_1 = lW_p_par_data)
    dataset_lN_p_par       = DataSet(datalist_1 = lN_p_par_data)
    dataset_j1_p_par       = DataSet(datalist_1 = j1_p_par_data)
    dataset_j2_p_par       = DataSet(datalist_1 = j2_p_par_data)
    dataset_lNj1j2_p_par   = DataSet(datalist_1 = lNj1j2_p_par_data)
    dataset_lWlNj1j2_p_par = DataSet(datalist_1 = lWlNj1j2_p_par_data)

    dataset_lW_p_perp       = DataSet(datalist_1 = lW_p_perp_data)
    dataset_lN_p_perp       = DataSet(datalist_1 = lN_p_perp_data)
    dataset_j1_p_perp       = DataSet(datalist_1 = j1_p_perp_data)
    dataset_j2_p_perp       = DataSet(datalist_1 = j2_p_perp_data)
    dataset_lNj1j2_p_perp   = DataSet(datalist_1 = lNj1j2_p_perp_data)
    dataset_lWlNj1j2_p_perp = DataSet(datalist_1 = lWlNj1j2_p_perp_data)

    dataset_lW_p_par_to_pt       = DataSet(datalist_1 = lW_p_par_to_pt_data)
    dataset_lN_p_par_to_pt       = DataSet(datalist_1 = lN_p_par_to_pt_data)
    dataset_j1_p_par_to_pt       = DataSet(datalist_1 = j1_p_par_to_pt_data)
    dataset_j2_p_par_to_pt       = DataSet(datalist_1 = j2_p_par_to_pt_data)
    dataset_lNj1j2_p_par_to_pt   = DataSet(datalist_1 = lNj1j2_p_par_to_pt_data)
#    dataset_lWlNj1j2_p_par_to_pt = DataSet(datalist_1 = lWlNj1j2_p_par_to_pt_data)

    dataset_lW_p_perp_to_pt       = DataSet(datalist_1 = lW_p_perp_to_pt_data)
    dataset_lN_p_perp_to_pt       = DataSet(datalist_1 = lN_p_perp_to_pt_data)
    dataset_j1_p_perp_to_pt       = DataSet(datalist_1 = j1_p_perp_to_pt_data)
    dataset_j2_p_perp_to_pt       = DataSet(datalist_1 = j2_p_perp_to_pt_data)
    dataset_lNj1j2_p_perp_to_pt   = DataSet(datalist_1 = lNj1j2_p_perp_to_pt_data)
#    dataset_lWlNj1j2_p_perp_to_pt = DataSet(datalist_1 = lWlNj1j2_p_perp_to_pt_data)

    dataset_lW_p_perp_to_par       = DataSet(datalist_1 = lW_p_perp_to_par_data)
    dataset_lN_p_perp_to_par       = DataSet(datalist_1 = lN_p_perp_to_par_data)
    dataset_j1_p_perp_to_par       = DataSet(datalist_1 = j1_p_perp_to_par_data)
    dataset_j2_p_perp_to_par       = DataSet(datalist_1 = j2_p_perp_to_par_data)
    dataset_lNj1j2_p_perp_to_par   = DataSet(datalist_1 = lNj1j2_p_perp_to_par_data)
#    dataset_lWlNj1j2_p_perp_to_par = DataSet(datalist_1 = lWlNj1j2_p_perp_to_par_data)


    dataset_lW_p_par_smeared       = DataSet(datalist_1 = lW_p_par_data_smeared)
    dataset_lN_p_par_smeared       = DataSet(datalist_1 = lN_p_par_data_smeared)
    dataset_j1_p_par_smeared       = DataSet(datalist_1 = j1_p_par_data_smeared)
    dataset_j2_p_par_smeared       = DataSet(datalist_1 = j2_p_par_data_smeared)
    dataset_lNj1j2_p_par_smeared   = DataSet(datalist_1 = lNj1j2_p_par_data_smeared)
    dataset_lWlNj1j2_p_par_smeared = DataSet(datalist_1 = lWlNj1j2_p_par_data_smeared)

    dataset_lW_p_perp_smeared       = DataSet(datalist_1 = lW_p_perp_data_smeared)
    dataset_lN_p_perp_smeared       = DataSet(datalist_1 = lN_p_perp_data_smeared)
    dataset_j1_p_perp_smeared       = DataSet(datalist_1 = j1_p_perp_data_smeared)
    dataset_j2_p_perp_smeared       = DataSet(datalist_1 = j2_p_perp_data_smeared)
    dataset_lNj1j2_p_perp_smeared   = DataSet(datalist_1 = lNj1j2_p_perp_data_smeared)
    dataset_lWlNj1j2_p_perp_smeared = DataSet(datalist_1 = lWlNj1j2_p_perp_data_smeared)

    dataset_lW_p_par_to_pt_smeared       = DataSet(datalist_1 = lW_p_par_to_pt_data_smeared)
    dataset_lN_p_par_to_pt_smeared       = DataSet(datalist_1 = lN_p_par_to_pt_data_smeared)
    dataset_j1_p_par_to_pt_smeared       = DataSet(datalist_1 = j1_p_par_to_pt_data_smeared)
    dataset_j2_p_par_to_pt_smeared       = DataSet(datalist_1 = j2_p_par_to_pt_data_smeared)
    dataset_lNj1j2_p_par_to_pt_smeared   = DataSet(datalist_1 = lNj1j2_p_par_to_pt_data_smeared)
    dataset_lWlNj1j2_p_par_to_pt_smeared = DataSet(datalist_1 = lWlNj1j2_p_par_to_pt_data_smeared)

    dataset_lW_p_perp_to_pt_smeared       = DataSet(datalist_1 = lW_p_perp_to_pt_data_smeared)
    dataset_lN_p_perp_to_pt_smeared       = DataSet(datalist_1 = lN_p_perp_to_pt_data_smeared)
    dataset_j1_p_perp_to_pt_smeared       = DataSet(datalist_1 = j1_p_perp_to_pt_data_smeared)
    dataset_j2_p_perp_to_pt_smeared       = DataSet(datalist_1 = j2_p_perp_to_pt_data_smeared)
    dataset_lNj1j2_p_perp_to_pt_smeared   = DataSet(datalist_1 = lNj1j2_p_perp_to_pt_data_smeared)
    dataset_lWlNj1j2_p_perp_to_pt_smeared = DataSet(datalist_1 = lWlNj1j2_p_perp_to_pt_data_smeared)

    dataset_lW_p_perp_to_par_smeared       = DataSet(datalist_1 = lW_p_perp_to_par_data_smeared)
    dataset_lN_p_perp_to_par_smeared       = DataSet(datalist_1 = lN_p_perp_to_par_data_smeared)
    dataset_j1_p_perp_to_par_smeared       = DataSet(datalist_1 = j1_p_perp_to_par_data_smeared)
    dataset_j2_p_perp_to_par_smeared       = DataSet(datalist_1 = j2_p_perp_to_par_data_smeared)
    dataset_lNj1j2_p_perp_to_par_smeared   = DataSet(datalist_1 = lNj1j2_p_perp_to_par_data_smeared)
    dataset_lWlNj1j2_p_perp_to_par_smeared = DataSet(datalist_1 = lWlNj1j2_p_perp_to_par_data_smeared)

    dataset_lWlNj1j2_mass_lW_align         = DataSet(datalist_1 = lWlNj1j2_mass_lW_align)
    dataset_lWlNj1j2_mass_smeared_lW_align = DataSet(datalist_1 = lWlNj1j2_mass_smeared_lW_align)


    dataset_event_0_jet_facs = DataSet(
                                       datalist_1 = s_1,
                                       datalist_2 = s_2[0]
                                      )
    dataset_event_0_chi_sq = DataSet(
                                     datalist_1 = s_1,
                                     datalist_2 = chi_sq_array[0]
                                    )
    dataset_best_s_1_array = DataSet(datalist_1 = best_s_1_array)
    dataset_best_s_2_array = DataSet(datalist_1 = best_s_2_array)

    dataset_lNj1j2_scaled_p_par_to_lW_p_par = DataSet(datalist_1 = lNj1j2_scaled_p_par_to_lW_p_par)
    dataset_lNj1j2_old_p_par_to_lW_p_par    = DataSet(datalist_1 = lNj1j2_old_p_par_to_lW_p_par)
    dataset_lWlNj1j2_p_par_smeared_scaled   = DataSet(datalist_1 = lWlNj1j2_p_par_data_smeared_scaled)
    dataset_lWlNj1j2_p_perp_smeared_scaled  = DataSet(datalist_1 = lWlNj1j2_p_perp_data_smeared_scaled)
    dataset_lNj1j2_mass_smeared_scaled      = DataSet(datalist_1 = lNj1j2_mass_smeared_scaled)
    dataset_lWlNj1j2_mass_smeared_scaled    = DataSet(datalist_1 = lWlNj1j2_mass_smeared_scaled)
    ############################
    ############################

    #################################################
    ### set axis limits and bin widths for masses ###
    #################################################
    if mWR == 1200:
        lo_3obj_mass_x_bound = 0
        hi_3obj_mass_x_bound = 20000
        lo_3obj_mass_y_bound = 1
        hi_3obj_mass_y_bound = 10**8

        lo_4obj_mass_x_bound = 0
        hi_4obj_mass_x_bound = 20000
        lo_4obj_mass_y_bound = 1
        hi_4obj_mass_y_bound = 10**8

        bin_width_3obj = 10
        bin_width_4obj = 10

    if mWR == 2000:
        lo_3obj_mass_x_bound = 0
        hi_3obj_mass_x_bound = 20000
        lo_3obj_mass_y_bound = 1
        hi_3obj_mass_y_bound = 10**8

        lo_4obj_mass_x_bound = 0
        hi_4obj_mass_x_bound = 20000
        lo_4obj_mass_y_bound = 1
        hi_4obj_mass_y_bound = 10**8

        bin_width_3obj = 10
        bin_width_4obj = 10

    if mWR == 3200:
        lo_3obj_mass_x_bound = 0
        hi_3obj_mass_x_bound = 20000
        lo_3obj_mass_y_bound = 1
        hi_3obj_mass_y_bound = 10**8

        lo_4obj_mass_x_bound = 0
        hi_4obj_mass_x_bound = 20000
        lo_4obj_mass_y_bound = 1
        hi_4obj_mass_y_bound = 10**8

        bin_width_3obj = 10
        bin_width_4obj = 10
    #################################################
    #################################################
    
    if make_pt_figs == True:
        print(f'\nplotting pt figs for (mWR, mN) = ({mWR}, {mN})\n')
        ##################
        ### pt figures ###
        ##################
        dataset_lW_pt.make_1d_hist(
                                   fig_dict = fig_agg.get_fig_dict(),
                                   figname = 'mWR_' + str(mWR) + '_' + 'lW_pt', #<-- VARIABLE NAME
                                   axname = 'ax_a',
                                   savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lW_pt.png', #<-- VARIABLE NAME,
                                   label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                   legend_loc = [1, 1],
                                   data = dataset_lW_pt.get_datalist(1), #<-- VARIABLE NAME
                                   bin_width = 50,
                                   x_bounds = [0, 2000],
                                   y_bounds = [10**0, 10**8],
                                   xscale = 'linear',
                                   yscale = 'log',
                                   xlabel = r'$l_{W}$ $p_{T}$ [GeV]', #<-- VARIABLE NAME
                                   ylabel = 'events / 50 GeV',
                                   title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                   annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                   overwrite = False
                                  )
        dataset_lN_pt.make_1d_hist(
                                   fig_dict = fig_agg.get_fig_dict(),
                                   figname = 'mWR_' + str(mWR) + '_' + 'lN_pt', #<-- VARIABLE NAME
                                   axname = 'ax_a',
                                   savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lN_pt.png', #<-- VARIABLE NAME
                                   label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                   legend_loc = [1, 1],
                                   data = dataset_lN_pt.get_datalist(1), #<-- VARIABLE NAME
                                   bin_width = 50,
                                   x_bounds = [0, 2000],
                                   y_bounds = [10**0, 10**8],
                                   xscale = 'linear',
                                   yscale = 'log',
                                   xlabel = r'$l_{N}$ $p_{T}$ [GeV]', #<-- VARIABLE NAME
                                   ylabel = 'events / 50 GeV',
                                   title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                   annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                   overwrite = False
                                  )
        dataset_j1_pt.make_1d_hist(
                                   fig_dict = fig_agg.get_fig_dict(),
                                   figname = 'mWR_' + str(mWR) + '_' + 'j1_pt', #<-- VARIABLE NAME
                                   axname = 'ax_a',
                                   savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_j1_pt.png', #<-- VARIABLE NAME,
                                   label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                   legend_loc = [1, 1],
                                   data = dataset_j1_pt.get_datalist(1), #<-- VARIABLE NAME
                                   bin_width = 50,
                                   x_bounds = [0, 2000],
                                   y_bounds = [10**0, 10**8],
                                   xscale = 'linear',
                                   yscale = 'log',
                                   xlabel = r'$j_{1}$ $p_{T}$ [GeV]', #<-- VARIABLE NAME
                                   ylabel = 'events / 50 GeV',
                                   title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                   annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                   overwrite = False
                                  )
        dataset_j2_pt.make_1d_hist(
                                   fig_dict = fig_agg.get_fig_dict(),
                                   figname = 'mWR_' + str(mWR) + '_' + 'j2_pt', #<-- VARIABLE NAME
                                   axname = 'ax_a',
                                   savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_j2_pt.png', #<-- VARIABLE NAME,
                                   label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                   legend_loc = [1, 1],
                                   data = dataset_j2_pt.get_datalist(1), #<-- VARIABLE NAME
                                   bin_width = 50,
                                   x_bounds = [0, 2000],
                                   y_bounds = [10**0, 10**8],
                                   xscale = 'linear',
                                   yscale = 'log',
                                   xlabel = r'$j_{2}$ $p_{T}$ [GeV]', #<-- VARIABLE NAME
                                   ylabel = 'events / 50 GeV',
                                   title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                   annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                   overwrite = False
                                  )
        dataset_j1j2_pt.make_1d_hist(
                                     fig_dict = fig_agg.get_fig_dict(),
                                     figname = 'mWR_' + str(mWR) + '_' + 'j1j2_pt', #<-- VARIABLE NAME
                                     axname = 'ax_a',
                                     savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_j1j2_pt.png', #<-- VARIABLE NAME,
                                     label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                     legend_loc = [1, 1],
                                     data = dataset_j1j2_pt.get_datalist(1), #<-- VARIABLE NAME
                                     bin_width = 50,
                                     x_bounds = [0, 2000],
                                     y_bounds = [10**0, 10**8],
                                     xscale = 'linear',
                                     yscale = 'log',
                                     xlabel = r'$j_{1}j_{2}$ $p_{T}$ [GeV]', #<-- VARIABLE NAME
                                     ylabel = 'events / 50 GeV',
                                     title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                     annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                     overwrite = False
                                    )
        dataset_lNj1j2_pt.make_1d_hist(
                                       fig_dict = fig_agg.get_fig_dict(),
                                       figname = 'mWR_' + str(mWR) + '_' + 'lNj1j2_pt', #<-- VARIABLE NAME
                                       axname = 'ax_a',
                                       savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lNj1j2_pt.png', #<-- VARIABLE NAME,
                                       label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                       legend_loc = [1, 1],
                                       data = dataset_lNj1j2_pt.get_datalist(1), #<-- VARIABLE NAME
                                       bin_width = 50,
                                       x_bounds = [0, 2000],
                                       y_bounds = [10**0, 10**8],
                                       xscale = 'linear',
                                       yscale = 'log',
                                       xlabel = r'$l_{N}j_{1}j_{2}$ $p_{T}$ [GeV]', #<-- VARIABLE NAME
                                       ylabel = 'events / 50 GeV',
                                       title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                       annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                       overwrite = False
                                      )
        dataset_lWlNj1j2_pt.make_1d_hist(
                                         fig_dict = fig_agg.get_fig_dict(),
                                         figname = 'mWR_' + str(mWR) + '_' + 'lWlNj1j2_pt', #<-- VARIABLE NAME
                                         axname = 'ax_a',
                                         savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lWlNj1j2_pt.png', #<-- VARIABLE NAME,
                                         label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                         legend_loc = [1, 1],
                                         data = dataset_lWlNj1j2_pt.get_datalist(1), #<-- VARIABLE NAME
                                         bin_width = 10,
                                         x_bounds = [0, 500],
                                         y_bounds = [10**0, 10**8],
                                         xscale = 'linear',
                                         yscale = 'log',
                                         xlabel = r'$l_{W}l_{N}j_{1}j_{2}$ $p_{T}$ [GeV]', #<-- VARIABLE NAME
                                         ylabel = 'events / 10 GeV',
                                         title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                         annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                         overwrite = False
                                        )
        dataset_lW_pt_smeared.make_1d_hist(
                                   fig_dict = fig_agg.get_fig_dict(),
                                   figname = 'mWR_' + str(mWR) + '_' + 'lW_pt_smeared', #<-- VARIABLE NAME
                                   axname = 'ax_a',
                                   savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lW_pt_smeared.png', #<-- VARIABLE NAME,
                                   label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                   legend_loc = [1, 1],
                                   data = dataset_lW_pt_smeared.get_datalist(1), #<-- VARIABLE NAME
                                   bin_width = 50,
                                   x_bounds = [0, 2000],
                                   y_bounds = [10**0, 10**8],
                                   xscale = 'linear',
                                   yscale = 'log',
                                   xlabel = r'$l_{W}$ $p_{T}$ [GeV]', #<-- VARIABLE NAME
                                   ylabel = 'events / 50 GeV',
                                   title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                   annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                   overwrite = False
                                  )
        dataset_lN_pt_smeared.make_1d_hist(
                                   fig_dict = fig_agg.get_fig_dict(),
                                   figname = 'mWR_' + str(mWR) + '_' + 'lN_pt_smeared', #<-- VARIABLE NAME
                                   axname = 'ax_a',
                                   savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lN_pt_smeared.png', #<-- VARIABLE NAME
                                   label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                   legend_loc = [1, 1],
                                   data = dataset_lN_pt_smeared.get_datalist(1), #<-- VARIABLE NAME
                                   bin_width = 50,
                                   x_bounds = [0, 2000],
                                   y_bounds = [10**0, 10**8],
                                   xscale = 'linear',
                                   yscale = 'log',
                                   xlabel = r'$l_{N}$ $p_{T}$ [GeV]', #<-- VARIABLE NAME
                                   ylabel = 'events / 50 GeV',
                                   title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                   annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                   overwrite = False
                                  )
        dataset_j1_pt_smeared.make_1d_hist(
                                   fig_dict = fig_agg.get_fig_dict(),
                                   figname = 'mWR_' + str(mWR) + '_' + 'j1_pt_smeared', #<-- VARIABLE NAME
                                   axname = 'ax_a',
                                   savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_j1_pt_smeared.png', #<-- VARIABLE NAME,
                                   label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                   legend_loc = [1, 1],
                                   data = dataset_j1_pt_smeared.get_datalist(1), #<-- VARIABLE NAME
                                   bin_width = 50,
                                   x_bounds = [0, 2000],
                                   y_bounds = [10**0, 10**8],
                                   xscale = 'linear',
                                   yscale = 'log',
                                   xlabel = r'$j_{1}$ $p_{T}$ [GeV]', #<-- VARIABLE NAME
                                   ylabel = 'events / 50 GeV',
                                   title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                   annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                   overwrite = False
                                  )
        dataset_j2_pt_smeared.make_1d_hist(
                                   fig_dict = fig_agg.get_fig_dict(),
                                   figname = 'mWR_' + str(mWR) + '_' + 'j2_pt_smeared', #<-- VARIABLE NAME
                                   axname = 'ax_a',
                                   savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_j2_pt_smeared.png', #<-- VARIABLE NAME,
                                   label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                   legend_loc = [1, 1],
                                   data = dataset_j2_pt_smeared.get_datalist(1), #<-- VARIABLE NAME
                                   bin_width = 50,
                                   x_bounds = [0, 2000],
                                   y_bounds = [10**0, 10**8],
                                   xscale = 'linear',
                                   yscale = 'log',
                                   xlabel = r'$j_{2}$ $p_{T}$ [GeV]', #<-- VARIABLE NAME
                                   ylabel = 'events / 50 GeV',
                                   title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                   annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                   overwrite = False
                                  )
        dataset_j1j2_pt_smeared.make_1d_hist(
                                     fig_dict = fig_agg.get_fig_dict(),
                                     figname = 'mWR_' + str(mWR) + '_' + 'j1j2_pt_smeared', #<-- VARIABLE NAME
                                     axname = 'ax_a',
                                     savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_j1j2_pt_smeared.png', #<-- VARIABLE NAME,
                                     label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                     legend_loc = [1, 1],
                                     data = dataset_j1j2_pt_smeared.get_datalist(1), #<-- VARIABLE NAME
                                     bin_width = 50,
                                     x_bounds = [0, 2000],
                                     y_bounds = [10**0, 10**8],
                                     xscale = 'linear',
                                     yscale = 'log',
                                     xlabel = r'$j_{1}j_{2}$ $p_{T}$ [GeV]', #<-- VARIABLE NAME
                                     ylabel = 'events / 50 GeV',
                                     title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                     annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                     overwrite = False
                                    )
        dataset_lNj1j2_pt_smeared.make_1d_hist(
                                       fig_dict = fig_agg.get_fig_dict(),
                                       figname = 'mWR_' + str(mWR) + '_' + 'lNj1j2_pt_smeared', #<-- VARIABLE NAME
                                       axname = 'ax_a',
                                       savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lNj1j2_pt_smeared.png', #<-- VARIABLE NAME,
                                       label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                       legend_loc = [1, 1],
                                       data = dataset_lNj1j2_pt_smeared.get_datalist(1), #<-- VARIABLE NAME
                                       bin_width = 50,
                                       x_bounds = [0, 2000],
                                       y_bounds = [10**0, 10**8],
                                       xscale = 'linear',
                                       yscale = 'log',
                                       xlabel = r'$l_{N}j_{1}j_{2}$ $p_{T}$ [GeV]', #<-- VARIABLE NAME
                                       ylabel = 'events / 50 GeV',
                                       title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                       annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                       overwrite = False
                                      )
        dataset_lWlNj1j2_pt_smeared.make_1d_hist(
                                         fig_dict = fig_agg.get_fig_dict(),
                                         figname = 'mWR_' + str(mWR) + '_' + 'lWlNj1j2_pt_smeared', #<-- VARIABLE NAME
                                         axname = 'ax_a',
                                         savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lWlNj1j2_pt_smeared.png', #<-- VARIABLE NAME,
                                         label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                         legend_loc = [1, 1],
                                         data = dataset_lWlNj1j2_pt_smeared.get_datalist(1), #<-- VARIABLE NAME
                                         bin_width = 10,
                                         x_bounds = [0, 500],
                                         y_bounds = [10**0, 10**8],
                                         xscale = 'linear',
                                         yscale = 'log',
                                         xlabel = r'$l_{W}l_{N}j_{1}j_{2}$ $p_{T}$ [GeV]', #<-- VARIABLE NAME
                                         ylabel = 'events / 10 GeV',
                                         title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                         annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                         overwrite = False
                                        )
        ##################
        ##################

    if make_mass_figs == True:
        print(f'\nplotting mass figs for (mWR, mN) = ({mWR}, {mN})\n')
        ####################
        ### mass figures ###
        ####################
#        dataset_j1j2_mass.make_1d_hist(
#                                       fig_dict = fig_agg.get_fig_dict(),
#                                       figname = 'mWR_' + str(mWR) + '_' + 'j1j2_mass', #<-- VARIABLE NAME
#                                       axname = 'ax_a',
#                                       savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_j1j2_mass.png', #<-- VARIABLE NAME,
#                                       label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
#                                       legend_loc = [1, 1],
#                                       data = dataset_j1j2_mass.get_datalist(1), #<-- VARIABLE NAME
#                                       bin_width = 50,
#                                       x_bounds = [0, 3000],
#                                       y_bounds = [10**0, 10**8],
#                                       xscale = 'linear',
#                                       yscale = 'log',
#                                       xlabel = r'$j_{1}j_{2}$ mass [GeV]', #<-- VARIABLE NAME
#                                       ylabel = 'events / 50 GeV',
#                                       title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
#                                       annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
#                                       overwrite = False
#                                      )
#        dataset_lNj1j2_mass.make_1d_hist(
#                                         fig_dict = fig_agg.get_fig_dict(),
#                                         figname = 'mWR_' + str(mWR) + '_' + 'lNj1j2_mass', #<-- VARIABLE NAME
#                                         axname = 'ax_a',
#                                         savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lNj1j2_mass.png', #<-- VARIABLE NAME,
#                                         label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
#                                         legend_loc = [1, 1],
#                                         data = dataset_lNj1j2_mass.get_datalist(1), #<-- VARIABLE NAME
#                                         bin_width = 50,
#                                         x_bounds = [0, 4000],
#                                         y_bounds = [10**0, 10**8],
#                                         xscale = 'linear',
#                                         yscale = 'log',
#                                         xlabel = r'$l_{N}j_{1}j_{2}$ mass [GeV]', #<-- VARIABLE NAME
#                                         ylabel = 'events / 50 GeV',
#                                         title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
#                                         annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
#                                         overwrite = False
#                                        )
#        dataset_lWlNj1j2_mass.make_1d_hist(
#                                           fig_dict = fig_agg.get_fig_dict(),
#                                           figname = 'mWR_' + str(mWR) + '_' + 'lWlNj1j2_mass', #<-- VARIABLE NAME
#                                           axname = 'ax_a',
#                                           savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lWlNj1j2_mass.png', #<-- VARIABLE NAME,
#                                           label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
#                                           legend_loc = [1, 1],
#                                           data = dataset_lWlNj1j2_mass.get_datalist(1), #<-- VARIABLE NAME
#                                           bin_width = 50,
#                                           x_bounds = [0, 4500],
#                                           y_bounds = [10**0, 10**8],
#                                           xscale = 'linear',
#                                           yscale = 'log',
#                                           xlabel = r'$l_{W}l_{N}j_{1}j_{2}$ mass [GeV]', #<-- VARIABLE NAME
#                                           ylabel = 'events / 50 GeV',
#                                           title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
#                                           annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
#                                           overwrite = False
#                                          )
#        dataset_j1j2_mass_smeared.make_1d_hist(
#                                       fig_dict = fig_agg.get_fig_dict(),
#                                       figname = 'mWR_' + str(mWR) + '_' + 'j1j2_mass_smeared', #<-- VARIABLE NAME
#                                       axname = 'ax_a',
#                                       savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_j1j2_mass_smeared.png', #<-- VARIABLE NAME,
#                                       label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
#                                       legend_loc = [1, 1],
#                                       data = dataset_j1j2_mass_smeared.get_datalist(1), #<-- VARIABLE NAME
#                                       bin_width = 50,
#                                       x_bounds = [0, 3000],
#                                       y_bounds = [10**0, 10**8],
#                                       xscale = 'linear',
#                                       yscale = 'log',
#                                       xlabel = r'$j_{1}j_{2}$ mass [GeV]', #<-- VARIABLE NAME
#                                       ylabel = 'events / 50 GeV',
#                                       title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
#                                       annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
#                                       overwrite = False
#                                      )
        lNj1j2_mass_smeared_bin_data = dataset_lNj1j2_mass_smeared.make_1d_hist(
                                                                                fig_dict = fig_agg.get_fig_dict(),
                                                                                figname = 'mWR_' + str(mWR) + '_' + 'lNj1j2_mass_smeared', #<-- VARIABLE NAME
                                                                                axname = 'ax_a',
                                                                                savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lNj1j2_mass_smeared.png', #<-- VARIABLE NAME,
                                                                                label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                                                                legend_loc = [1, 1],
                                                                                data = dataset_lNj1j2_mass_smeared.get_datalist(1), #<-- VARIABLE NAME
                                                                                bin_width = bin_width_3obj,
                                                                                x_bounds = [lo_3obj_mass_x_bound, hi_3obj_mass_x_bound],
                                                                                y_bounds = [lo_3obj_mass_y_bound, hi_3obj_mass_y_bound],
                                                                                xscale = 'linear',
                                                                                yscale = 'log',
                                                                                xlabel = r'$l_{N}j_{1}j_{2}$ mass [GeV]', #<-- VARIABLE NAME
                                                                                ylabel = f'events / {bin_width_3obj} GeV',
                                                                                title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                                                                annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                                                                overwrite = False
                                                                               )
        lWlNj1j2_mass_smeared_bin_data = dataset_lWlNj1j2_mass_smeared.make_1d_hist(
                                                                                    fig_dict = fig_agg.get_fig_dict(),
                                                                                    figname = 'mWR_' + str(mWR) + '_' + 'lWlNj1j2_mass_smeared', #<-- VARIABLE NAME
                                                                                    axname = 'ax_a',
                                                                                    savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lWlNj1j2_mass_smeared.png', #<-- VARIABLE NAME,
                                                                                    label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                                                                    legend_loc = [1, 1],
                                                                                    data = dataset_lWlNj1j2_mass_smeared.get_datalist(1), #<-- VARIABLE NAME
                                                                                    bin_width = bin_width_4obj,
                                                                                    x_bounds = [lo_4obj_mass_x_bound, hi_4obj_mass_x_bound],
                                                                                    y_bounds = [lo_4obj_mass_y_bound, hi_4obj_mass_y_bound],
                                                                                    xscale = 'linear',
                                                                                    yscale = 'log',
                                                                                    xlabel = r'$l_{W}l_{N}j_{1}j_{2}$ mass [GeV]', #<-- VARIABLE NAME
                                                                                    ylabel = f'events / {bin_width_4obj} GeV',
                                                                                    title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                                                                    annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                                                                    overwrite = False
                                                                                   )
        ####################
        ####################

    if make_eta_figs == True:
        print(f'\nplotting eta figs for (mWR, mN) = ({mWR}, {mN})\n')
        ###################
        ### eta figures ###
        ###################
        dataset_lW_eta.make_1d_hist(
                                    fig_dict = fig_agg.get_fig_dict(),
                                    figname = 'mWR_' + str(mWR) + '_' + 'lW_eta', #<-- VARIABLE NAME
                                    axname = 'ax_a',
                                    savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lW_eta.png', #<-- VARIABLE NAME,
                                    label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                    legend_loc = [1, 1],
                                    data = dataset_lW_eta.get_datalist(1), #<-- VARIABLE NAME
                                    bin_width = 0.1,
                                    x_bounds = [-4, 4],
                                    y_bounds = [10**0, 10**8],
                                    xscale = 'linear',
                                    yscale = 'log',
                                    xlabel = r'$l_{W}$ $\eta$', #<-- VARIABLE NAME
                                    ylabel = 'events / 0.1',
                                    title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                    annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                    overwrite = False
                                   )
        dataset_lN_eta.make_1d_hist(
                                    fig_dict = fig_agg.get_fig_dict(),
                                    figname = 'mWR_' + str(mWR) + '_' + 'lN_eta', #<-- VARIABLE NAME
                                    axname = 'ax_a',
                                    savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lN_eta.png', #<-- VARIABLE NAME,
                                    label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                    legend_loc = [1, 1],
                                    data = dataset_lN_eta.get_datalist(1), #<-- VARIABLE NAME
                                    bin_width = 0.1,
                                    x_bounds = [-4, 4],
                                    y_bounds = [10**0, 10**8],
                                    xscale = 'linear',
                                    yscale = 'log',
                                    xlabel = r'$l_{N}$ $\eta$', #<-- VARIABLE NAME
                                    ylabel = 'events / 0.1',
                                    title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                    annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                    overwrite = False
                                   )
        dataset_j1_eta.make_1d_hist(
                                    fig_dict = fig_agg.get_fig_dict(),
                                    figname = 'mWR_' + str(mWR) + '_' + 'j1_eta', #<-- VARIABLE NAME
                                    axname = 'ax_a',
                                    savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_j1_eta.png', #<-- VARIABLE NAME,
                                    label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                    legend_loc = [1, 1],
                                    data = dataset_j1_eta.get_datalist(1), #<-- VARIABLE NAME
                                    bin_width = 0.1,
                                    x_bounds = [-4, 4],
                                    y_bounds = [10**0, 10**8],
                                    xscale = 'linear',
                                    yscale = 'log',
                                    xlabel = r'$j_{1}$ $\eta$', #<-- VARIABLE NAME
                                    ylabel = 'events / 0.1',
                                    title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                    annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                    overwrite = False
                                   )
        dataset_j2_eta.make_1d_hist(
                                    fig_dict = fig_agg.get_fig_dict(),
                                    figname = 'mWR_' + str(mWR) + '_' + 'j2_eta', #<-- VARIABLE NAME
                                    axname = 'ax_a',
                                    savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_j2_eta.png', #<-- VARIABLE NAME,
                                    label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                    legend_loc = [1, 1],
                                    data = dataset_j2_eta.get_datalist(1), #<-- VARIABLE NAME
                                    bin_width = 0.1,
                                    x_bounds = [-4, 4],
                                    y_bounds = [10**0, 10**8],
                                    xscale = 'linear',
                                    yscale = 'log',
                                    xlabel = r'$j_{2}$ $\eta$', #<-- VARIABLE NAME
                                    ylabel = 'events / 0.1',
                                    title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                    annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                    overwrite = False
                                   )
        dataset_lW_eta_smeared.make_1d_hist(
                                    fig_dict = fig_agg.get_fig_dict(),
                                    figname = 'mWR_' + str(mWR) + '_' + 'lW_eta_smeared', #<-- VARIABLE NAME
                                    axname = 'ax_a',
                                    savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lW_eta_smeared.png', #<-- VARIABLE NAME,
                                    label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                    legend_loc = [1, 1],
                                    data = dataset_lW_eta_smeared.get_datalist(1), #<-- VARIABLE NAME
                                    bin_width = 0.1,
                                    x_bounds = [-4, 4],
                                    y_bounds = [10**0, 10**8],
                                    xscale = 'linear',
                                    yscale = 'log',
                                    xlabel = r'$l_{W}$ $\eta$', #<-- VARIABLE NAME
                                    ylabel = 'events / 0.1',
                                    title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                    annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                    overwrite = False
                                   )
        dataset_lN_eta_smeared.make_1d_hist(
                                    fig_dict = fig_agg.get_fig_dict(),
                                    figname = 'mWR_' + str(mWR) + '_' + 'lN_eta_smeared', #<-- VARIABLE NAME
                                    axname = 'ax_a',
                                    savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lN_eta_smeared.png', #<-- VARIABLE NAME,
                                    label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                    legend_loc = [1, 1],
                                    data = dataset_lN_eta_smeared.get_datalist(1), #<-- VARIABLE NAME
                                    bin_width = 0.1,
                                    x_bounds = [-4, 4],
                                    y_bounds = [10**0, 10**8],
                                    xscale = 'linear',
                                    yscale = 'log',
                                    xlabel = r'$l_{N}$ $\eta$', #<-- VARIABLE NAME
                                    ylabel = 'events / 0.1',
                                    title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                    annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                    overwrite = False
                                   )
        dataset_j1_eta_smeared.make_1d_hist(
                                    fig_dict = fig_agg.get_fig_dict(),
                                    figname = 'mWR_' + str(mWR) + '_' + 'j1_eta_smeared', #<-- VARIABLE NAME
                                    axname = 'ax_a',
                                    savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_j1_eta_smeared.png', #<-- VARIABLE NAME,
                                    label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                    legend_loc = [1, 1],
                                    data = dataset_j1_eta_smeared.get_datalist(1), #<-- VARIABLE NAME
                                    bin_width = 0.1,
                                    x_bounds = [-4, 4],
                                    y_bounds = [10**0, 10**8],
                                    xscale = 'linear',
                                    yscale = 'log',
                                    xlabel = r'$j_{1}$ $\eta$', #<-- VARIABLE NAME
                                    ylabel = 'events / 0.1',
                                    title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                    annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                    overwrite = False
                                   )
        dataset_j2_eta_smeared.make_1d_hist(
                                    fig_dict = fig_agg.get_fig_dict(),
                                    figname = 'mWR_' + str(mWR) + '_' + 'j2_eta_smeared', #<-- VARIABLE NAME
                                    axname = 'ax_a',
                                    savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_j2_eta_smeared.png', #<-- VARIABLE NAME,
                                    label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                    legend_loc = [1, 1],
                                    data = dataset_j2_eta_smeared.get_datalist(1), #<-- VARIABLE NAME
                                    bin_width = 0.1,
                                    x_bounds = [-4, 4],
                                    y_bounds = [10**0, 10**8],
                                    xscale = 'linear',
                                    yscale = 'log',
                                    xlabel = r'$j_{2}$ $\eta$', #<-- VARIABLE NAME
                                    ylabel = 'events / 0.1',
                                    title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                    annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                    overwrite = False
                                   )
        ###################
        ###################

    if make_phi_figs == True:
        print(f'\nplotting phi figs for (mWR, mN) = ({mWR}, {mN})\n')
        ###################
        ### phi figures ###
        ###################
        dataset_lW_phi.make_1d_hist(
                                    fig_dict = fig_agg.get_fig_dict(),
                                    figname = 'mWR_' + str(mWR) + '_' + 'lW_phi', #<-- VARIABLE NAME
                                    axname = 'ax_a',
                                    savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lW_phi.png', #<-- VARIABLE NAME,
                                    label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                    legend_loc = [1, 1],
                                    data = dataset_lW_phi.get_datalist(1), #<-- VARIABLE NAME
                                    bin_width = 0.1,
                                    x_bounds = [-3.2, 3.2],
                                    y_bounds = [10**0, 10**8],
                                    xscale = 'linear',
                                    yscale = 'log',
                                    xlabel = r'$l_{W}$ $\phi$', #<-- VARIABLE NAME
                                    ylabel = 'events / 0.1',
                                    title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                    annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                    overwrite = False
                                   )
        dataset_lN_phi.make_1d_hist(
                                    fig_dict = fig_agg.get_fig_dict(),
                                    figname = 'mWR_' + str(mWR) + '_' + 'lN_phi', #<-- VARIABLE NAME
                                    axname = 'ax_a',
                                    savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lN_phi.png', #<-- VARIABLE NAME,
                                    label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                    legend_loc = [1, 1],
                                    data = dataset_lN_phi.get_datalist(1), #<-- VARIABLE NAME
                                    bin_width = 0.1,
                                    x_bounds = [-3.2, 3.2],
                                    y_bounds = [10**0, 10**8],
                                    xscale = 'linear',
                                    yscale = 'log',
                                    xlabel = r'$l_{N}$ $\phi$', #<-- VARIABLE NAME
                                    ylabel = 'events / 0.1',
                                    title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                    annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                    overwrite = False
                                   )
        dataset_j1_phi.make_1d_hist(
                                    fig_dict = fig_agg.get_fig_dict(),
                                    figname = 'mWR_' + str(mWR) + '_' + 'j1_phi', #<-- VARIABLE NAME
                                    axname = 'ax_a',
                                    savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_j1_phi.png', #<-- VARIABLE NAME,
                                    label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                    legend_loc = [1, 1],
                                    data = dataset_j1_phi.get_datalist(1), #<-- VARIABLE NAME
                                    bin_width = 0.1,
                                    x_bounds = [-3.2, 3.2],
                                    y_bounds = [10**0, 10**8],
                                    xscale = 'linear',
                                    yscale = 'log',
                                    xlabel = r'$j_{1}$ $\phi$', #<-- VARIABLE NAME
                                    ylabel = 'events / 0.1',
                                    title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                    annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                    overwrite = False
                                   )
        dataset_j2_phi.make_1d_hist(
                                    fig_dict = fig_agg.get_fig_dict(),
                                    figname = 'mWR_' + str(mWR) + '_' + 'j2_phi', #<-- VARIABLE NAME
                                    axname = 'ax_a',
                                    savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_j2_phi.png', #<-- VARIABLE NAME,
                                    label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                    legend_loc = [1, 1],
                                    data = dataset_j2_phi.get_datalist(1), #<-- VARIABLE NAME
                                    bin_width = 0.1,
                                    x_bounds = [-3.2, 3.2],
                                    y_bounds = [10**0, 10**8],
                                    xscale = 'linear',
                                    yscale = 'log',
                                    xlabel = r'$j_{2}$ $\phi$', #<-- VARIABLE NAME
                                    ylabel = 'events / 0.1',
                                    title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                    annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                    overwrite = False
                                   )
        dataset_lW_phi_smeared.make_1d_hist(
                                    fig_dict = fig_agg.get_fig_dict(),
                                    figname = 'mWR_' + str(mWR) + '_' + 'lW_phi_smeared', #<-- VARIABLE NAME
                                    axname = 'ax_a',
                                    savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lW_phi_smeared.png', #<-- VARIABLE NAME,
                                    label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                    legend_loc = [1, 1],
                                    data = dataset_lW_phi_smeared.get_datalist(1), #<-- VARIABLE NAME
                                    bin_width = 0.1,
                                    x_bounds = [-3.2, 3.2],
                                    y_bounds = [10**0, 10**8],
                                    xscale = 'linear',
                                    yscale = 'log',
                                    xlabel = r'$l_{W}$ $\phi$', #<-- VARIABLE NAME
                                    ylabel = 'events / 0.1',
                                    title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                    annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                    overwrite = False
                                   )
        dataset_lN_phi_smeared.make_1d_hist(
                                    fig_dict = fig_agg.get_fig_dict(),
                                    figname = 'mWR_' + str(mWR) + '_' + 'lN_phi_smeared', #<-- VARIABLE NAME
                                    axname = 'ax_a',
                                    savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lN_phi_smeared.png', #<-- VARIABLE NAME,
                                    label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                    legend_loc = [1, 1],
                                    data = dataset_lN_phi_smeared.get_datalist(1), #<-- VARIABLE NAME
                                    bin_width = 0.1,
                                    x_bounds = [-3.2, 3.2],
                                    y_bounds = [10**0, 10**8],
                                    xscale = 'linear',
                                    yscale = 'log',
                                    xlabel = r'$l_{N}$ $\phi$', #<-- VARIABLE NAME
                                    ylabel = 'events / 0.1',
                                    title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                    annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                    overwrite = False
                                   )
        dataset_j1_phi_smeared.make_1d_hist(
                                    fig_dict = fig_agg.get_fig_dict(),
                                    figname = 'mWR_' + str(mWR) + '_' + 'j1_phi_smeared', #<-- VARIABLE NAME
                                    axname = 'ax_a',
                                    savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_j1_phi_smeared.png', #<-- VARIABLE NAME,
                                    label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                    legend_loc = [1, 1],
                                    data = dataset_j1_phi_smeared.get_datalist(1), #<-- VARIABLE NAME
                                    bin_width = 0.1,
                                    x_bounds = [-3.2, 3.2],
                                    y_bounds = [10**0, 10**8],
                                    xscale = 'linear',
                                    yscale = 'log',
                                    xlabel = r'$j_{1}$ $\phi$', #<-- VARIABLE NAME
                                    ylabel = 'events / 0.1',
                                    title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                    annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                    overwrite = False
                                   )
        dataset_j2_phi_smeared.make_1d_hist(
                                    fig_dict = fig_agg.get_fig_dict(),
                                    figname = 'mWR_' + str(mWR) + '_' + 'j2_phi_smeared', #<-- VARIABLE NAME
                                    axname = 'ax_a',
                                    savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_j2_phi_smeared.png', #<-- VARIABLE NAME,
                                    label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                    legend_loc = [1, 1],
                                    data = dataset_j2_phi_smeared.get_datalist(1), #<-- VARIABLE NAME
                                    bin_width = 0.1,
                                    x_bounds = [-3.2, 3.2],
                                    y_bounds = [10**0, 10**8],
                                    xscale = 'linear',
                                    yscale = 'log',
                                    xlabel = r'$j_{2}$ $\phi$', #<-- VARIABLE NAME
                                    ylabel = 'events / 0.1',
                                    title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                    annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                    overwrite = False
                                   )
        ###################
        ###################

    if make_smear_frac_figs == True:
        print(f'\nplotting smear frac figs for (mWR, mN) = ({mWR}, {mN})\n')
        ##########################
        ### smear frac figures ###
        ##########################
        dataset_lW_e_smear_frac.make_1d_hist(
                                             fig_dict = fig_agg.get_fig_dict(),
                                             figname = 'mWR_' + str(mWR) + '_' + 'lW_e_smear_frac', #<-- VARIABLE NAME
                                             axname = 'ax_a',
                                             savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lW_e_smear_frac.png', #<-- VARIABLE NAME,
                                             label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                             legend_loc = [1, 1],
                                             data = dataset_lW_e_smear_frac.get_datalist(1), #<-- VARIABLE NAME
                                             bin_width = 0.1,
                                             x_bounds = [-2, 8],
                                             y_bounds = [10**0, 10**8],
                                             xscale = 'linear',
                                             yscale = 'log',
                                             xlabel = r'$l_{W}$ $\frac{smeared \ energy}{unsmeared \ energy}$ - 1', #<-- VARIABLE NAME
                                             ylabel = 'events / 0.1',
                                             title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                             annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                             overwrite = False
                                            )
        dataset_lN_e_smear_frac.make_1d_hist(
                                             fig_dict = fig_agg.get_fig_dict(),
                                             figname = 'mWR_' + str(mWR) + '_' + 'lN_e_smear_frac', #<-- VARIABLE NAME
                                             axname = 'ax_a',
                                             savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lN_e_smear_frac.png', #<-- VARIABLE NAME,
                                             label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                             legend_loc = [1, 1],
                                             data = dataset_lN_e_smear_frac.get_datalist(1), #<-- VARIABLE NAME
                                             bin_width = 0.1,
                                             x_bounds = [-2, 8],
                                             y_bounds = [10**0, 10**8],
                                             xscale = 'linear',
                                             yscale = 'log',
                                             xlabel = r'$l_{N}$ $\frac{smeared \ energy}{unsmeared \ energy}$ - 1', #<-- VARIABLE NAME
                                             ylabel = 'events / 0.1',
                                             title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                             annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                             overwrite = False
                                            )
        j1_e_smear_frac_bin_counts, j1_e_smear_frac_bin_edges = dataset_j1_e_smear_frac.make_1d_hist(
                                                                                                     fig_dict = fig_agg.get_fig_dict(),
                                                                                                     figname = 'mWR_' + str(mWR) + '_' + 'j1_e_smear_frac', #<-- VARIABLE NAME
                                                                                                     axname = 'ax_a',
                                                                                                     savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_j1_e_smear_frac.png', #<-- VAR
                                                                                                     label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                                                                                     legend_loc = [1, 1],
                                                                                                     data = dataset_j1_e_smear_frac.get_datalist(1), #<-- VARIABLE NAME
                                                                                                     bin_width = 0.01,
                                                                                                     x_bounds = [-2, 2],
                                                                                                     y_bounds = [10**0, 10**8],
                                                                                                     xscale = 'linear',
                                                                                                     yscale = 'log',
                                                                                                     xlabel = r'$j_{1}$ $\frac{smeared \ energy}{unsmeared \ energy}$ - 1', #<-- VARIABLE NAME
                                                                                                     ylabel = 'events / 0.01',
                                                                                                     title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                                                                                     annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                                                                                     overwrite = False
                                                                                                    )
        j2_e_smear_frac_bin_counts, j2_e_smear_frac_bin_edges = dataset_j2_e_smear_frac.make_1d_hist(
                                                                                                     fig_dict = fig_agg.get_fig_dict(),
                                                                                                     figname = 'mWR_' + str(mWR) + '_' + 'j2_e_smear_frac', #<-- VARIABLE NAME
                                                                                                     axname = 'ax_a',
                                                                                                     savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_j2_e_smear_frac.png', #<-- VAR
                                                                                                     label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                                                                                     legend_loc = [1, 1],
                                                                                                     data = dataset_j2_e_smear_frac.get_datalist(1), #<-- VARIABLE NAME
                                                                                                     bin_width = 0.01,
                                                                                                     x_bounds = [-2, 2],
                                                                                                     y_bounds = [10**0, 10**8],
                                                                                                     xscale = 'linear',
                                                                                                     yscale = 'log',
                                                                                                     xlabel = r'$j_{2}$ $\frac{smeared \ energy}{unsmeared \ energy}$ - 1', #<-- VARIABLE NAME
                                                                                                     ylabel = 'events / 0.01',
                                                                                                     title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                                                                                     annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                                                                                     overwrite = False
                                                                                                    )
        ##########################
        ##########################

    if make_lW_pt_aligned_figs == True:
        print(f'\nplotting lW pt aligned figs for (mWR, mN) = ({mWR}, {mN})\n')
        #############################
        ### lW pt aligned figures ###
        #############################
        dataset_lW_p_par.make_1d_hist(
                                             fig_dict = fig_agg.get_fig_dict(),
                                             figname = 'mWR_' + str(mWR) + '_' + 'lW_p_par', #<-- VARIABLE NAME
                                             axname = 'ax_a',
                                             savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lW_p_par.png', #<-- VARIABLE NAME,
                                             label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                             legend_loc = [1, 1],
                                             data = dataset_lW_p_par.get_datalist(1), #<-- VARIABLE NAME
                                             bin_width = 50,
                                             x_bounds = [-2*10**3, 2*10**3],
                                             y_bounds = [10**0, 10**8],
                                             xscale = 'linear',
                                             yscale = 'log',
                                             xlabel = r'$l_W$ $p_{||}$', #<-- VARIABLE NAME
                                             ylabel = 'events / 50 GeV',
                                             title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                             annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                             overwrite = False
                                            )
        dataset_lN_p_par.make_1d_hist(
                                             fig_dict = fig_agg.get_fig_dict(),
                                             figname = 'mWR_' + str(mWR) + '_' + 'lN_p_par', #<-- VARIABLE NAME
                                             axname = 'ax_a',
                                             savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lN_p_par.png', #<-- VARIABLE NAME,
                                             label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                             legend_loc = [1, 1],
                                             data = dataset_lN_p_par.get_datalist(1), #<-- VARIABLE NAME
                                             bin_width = 50,
                                             x_bounds = [-2*10**3, 2*10**3],
                                             y_bounds = [10**0, 10**8],
                                             xscale = 'linear',
                                             yscale = 'log',
                                             xlabel = r'$l_N$ $p_{||}$', #<-- VARIABLE NAME
                                             ylabel = 'events / 50 GeV',
                                             title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                             annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                             overwrite = False
                                            )
        dataset_lNj1j2_p_par.make_1d_hist(
                                             fig_dict = fig_agg.get_fig_dict(),
                                             figname = 'mWR_' + str(mWR) + '_' + 'lNj1j2_p_par', #<-- VARIABLE NAME
                                             axname = 'ax_a',
                                             savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lNj1j2_p_par.png', #<-- VARIABLE NAME,
                                             label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                             legend_loc = [1, 1],
                                             data = dataset_lNj1j2_p_par.get_datalist(1), #<-- VARIABLE NAME
                                             bin_width = 50,
                                             x_bounds = [-2*10**3, 2*10**3],
                                             y_bounds = [10**0, 10**8],
                                             xscale = 'linear',
                                             yscale = 'log',
                                             xlabel = r'$l_N j_1 j_2$ $p_{||}$', #<-- VARIABLE NAME
                                             ylabel = 'events / 50 GeV',
                                             title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                             annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                             overwrite = False
                                            )
        dataset_lWlNj1j2_p_par.make_1d_hist(
                                             fig_dict = fig_agg.get_fig_dict(),
                                             figname = 'mWR_' + str(mWR) + '_' + 'lWlNj1j2_p_par', #<-- VARIABLE NAME
                                             axname = 'ax_a',
                                             savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lWlNj1j2_p_par.png', #<-- VARIABLE NAME,
                                             label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                             legend_loc = [1, 1],
                                             data = dataset_lWlNj1j2_p_par.get_datalist(1), #<-- VARIABLE NAME
                                             bin_width = 10,
                                             x_bounds = [-5*10**2, 5*10**2],
                                             y_bounds = [10**0, 10**8],
                                             xscale = 'linear',
                                             yscale = 'log',
                                             xlabel = r'$l_W l_N j_1 j_2$ $p_{||}$', #<-- VARIABLE NAME
                                             ylabel = 'events / 10 GeV',
                                             title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                             annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                             overwrite = False
                                            )
        dataset_lW_p_perp.make_1d_hist(
                                             fig_dict = fig_agg.get_fig_dict(),
                                             figname = 'mWR_' + str(mWR) + '_' + 'lW_p_perp', #<-- VARIABLE NAME
                                             axname = 'ax_a',
                                             savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lW_p_perp.png', #<-- VARIABLE NAME,
                                             label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                             legend_loc = [1, 1],
                                             data = dataset_lW_p_perp.get_datalist(1), #<-- VARIABLE NAME
                                             bin_width = 0.01,
                                             x_bounds = [-0.5, 0.5],
                                             y_bounds = [10**0, 10**8],
                                             xscale = 'linear',
                                             yscale = 'log',
                                             xlabel = r'$l_W$ $p_{\perp}$', #<-- VARIABLE NAME
                                             ylabel = 'events / 0.01 GeV',
                                             title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                             annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                             overwrite = False
                                            )
        dataset_lN_p_perp.make_1d_hist(
                                             fig_dict = fig_agg.get_fig_dict(),
                                             figname = 'mWR_' + str(mWR) + '_' + 'lN_p_perp', #<-- VARIABLE NAME
                                             axname = 'ax_a',
                                             savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lN_p_perp.png', #<-- VARIABLE NAME,
                                             label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                             legend_loc = [1, 1],
                                             data = dataset_lN_p_perp.get_datalist(1), #<-- VARIABLE NAME
                                             bin_width = 50,
                                             x_bounds = [-1*10**3, 1*10**3],
                                             y_bounds = [10**0, 10**8],
                                             xscale = 'linear',
                                             yscale = 'log',
                                             xlabel = r'$l_N$ $p_{\perp}$', #<-- VARIABLE NAME
                                             ylabel = 'events / 50 GeV',
                                             title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                             annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                             overwrite = False
                                            )
        dataset_lNj1j2_p_perp.make_1d_hist(
                                             fig_dict = fig_agg.get_fig_dict(),
                                             figname = 'mWR_' + str(mWR) + '_' + 'lNj1j2_p_perp', #<-- VARIABLE NAME
                                             axname = 'ax_a',
                                             savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lNj1j2_p_perp.png', #<-- VARIABLE NAME,
                                             label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                             legend_loc = [1, 1],
                                             data = dataset_lNj1j2_p_perp.get_datalist(1), #<-- VARIABLE NAME
                                             bin_width = 10,
                                             x_bounds = [-5*10**2, 5*10**2],
                                             y_bounds = [10**0, 10**8],
                                             xscale = 'linear',
                                             yscale = 'log',
                                             xlabel = r'$l_N j_1 j_2$ $p_{\perp}$', #<-- VARIABLE NAME
                                             ylabel = 'events / 10 GeV',
                                             title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                             annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                             overwrite = False
                                            )
        dataset_lWlNj1j2_p_perp.make_1d_hist(
                                             fig_dict = fig_agg.get_fig_dict(),
                                             figname = 'mWR_' + str(mWR) + '_' + 'lWlNj1j2_p_perp', #<-- VARIABLE NAME
                                             axname = 'ax_a',
                                             savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lWlNj1j2_p_perp.png', #<-- VARIABLE NAME,
                                             label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                             legend_loc = [1, 1],
                                             data = dataset_lWlNj1j2_p_perp.get_datalist(1), #<-- VARIABLE NAME
                                             bin_width = 10,
                                             x_bounds = [-5*10**2, 5*10**2],
                                             y_bounds = [10**0, 10**8],
                                             xscale = 'linear',
                                             yscale = 'log',
                                             xlabel = r'$l_W l_N j_1 j_2$ $p_{\perp}$', #<-- VARIABLE NAME
                                             ylabel = 'events / 10 GeV',
                                             title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                             annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                             overwrite = False
                                            )
        dataset_lW_p_par_to_pt.make_1d_hist(
                                             fig_dict = fig_agg.get_fig_dict(),
                                             figname = 'mWR_' + str(mWR) + '_' + 'lW_p_par_to_pt', #<-- VARIABLE NAME
                                             axname = 'ax_a',
                                             savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lW_p_par_to_pt.png', #<-- VARIABLE NAME,
                                             label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                             legend_loc = [1, 1],
                                             data = dataset_lW_p_par_to_pt.get_datalist(1), #<-- VARIABLE NAME
                                             bin_width = 0.01,
                                             x_bounds = [-1, 1],
                                             y_bounds = [10**0, 10**8],
                                             xscale = 'linear',
                                             yscale = 'log',
                                             xlabel = r'$l_W$ $\frac{p_{||}}{p_t}$', #<-- VARIABLE NAME
                                             ylabel = 'events / 0.01',
                                             title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                             annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                             overwrite = False
                                            )
        dataset_lN_p_par_to_pt.make_1d_hist(
                                             fig_dict = fig_agg.get_fig_dict(),
                                             figname = 'mWR_' + str(mWR) + '_' + 'lN_p_par_to_pt', #<-- VARIABLE NAME
                                             axname = 'ax_a',
                                             savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lN_p_par_to_pt.png', #<-- VARIABLE NAME,
                                             label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                             legend_loc = [1, 1],
                                             data = dataset_lN_p_par_to_pt.get_datalist(1), #<-- VARIABLE NAME
                                             bin_width = 0.01,
                                             x_bounds = [-1, 1],
                                             y_bounds = [10**0, 10**8],
                                             xscale = 'linear',
                                             yscale = 'log',
                                             xlabel = r'$l_N$ $\frac{p_{||}}{p_t}$', #<-- VARIABLE NAME
                                             ylabel = 'events / 0.01',
                                             title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                             annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                             overwrite = False
                                            )
        dataset_lNj1j2_p_par_to_pt.make_1d_hist(
                                             fig_dict = fig_agg.get_fig_dict(),
                                             figname = 'mWR_' + str(mWR) + '_' + 'lNj1j2_p_par_to_pt', #<-- VARIABLE NAME
                                             axname = 'ax_a',
                                             savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lNj1j2_p_par_to_pt.png', #<-- VARIABLE NAME,
                                             label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                             legend_loc = [1, 1],
                                             data = dataset_lNj1j2_p_par_to_pt.get_datalist(1), #<-- VARIABLE NAME
                                             bin_width = 0.01,
                                             x_bounds = [-1, 1],
                                             y_bounds = [10**0, 10**8],
                                             xscale = 'linear',
                                             yscale = 'log',
                                             xlabel = r'$l_N j_1 j_2$ $\frac{p_{||}}{p_t}$', #<-- VARIABLE NAME
                                             ylabel = 'events / 0.01',
                                             title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                             annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                             overwrite = False
                                            )
        dataset_lW_p_perp_to_pt.make_1d_hist(
                                             fig_dict = fig_agg.get_fig_dict(),
                                             figname = 'mWR_' + str(mWR) + '_' + 'lW_p_perp_to_pt', #<-- VARIABLE NAME
                                             axname = 'ax_a',
                                             savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lW_p_perp_to_pt.png', #<-- VARIABLE NAME,
                                             label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                             legend_loc = [1, 1],
                                             data = dataset_lW_p_perp_to_pt.get_datalist(1), #<-- VARIABLE NAME
                                             bin_width = 0.01,
                                             x_bounds = [-1, 1],
                                             y_bounds = [10**0, 10**8],
                                             xscale = 'linear',
                                             yscale = 'log',
                                             xlabel = r'$l_W$ $\frac{p_{\perp}}{p_t}$', #<-- VARIABLE NAME
                                             ylabel = 'events / 0.01',
                                             title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                             annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                             overwrite = False
                                            )
        dataset_lN_p_perp_to_pt.make_1d_hist(
                                             fig_dict = fig_agg.get_fig_dict(),
                                             figname = 'mWR_' + str(mWR) + '_' + 'lN_p_perp_to_pt', #<-- VARIABLE NAME
                                             axname = 'ax_a',
                                             savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lN_p_perp_to_pt.png', #<-- VARIABLE NAME,
                                             label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                             legend_loc = [1, 1],
                                             data = dataset_lN_p_perp_to_pt.get_datalist(1), #<-- VARIABLE NAME
                                             bin_width = 0.01,
                                             x_bounds = [-1, 1],
                                             y_bounds = [10**0, 10**8],
                                             xscale = 'linear',
                                             yscale = 'log',
                                             xlabel = r'$l_N$ $\frac{p_{\perp}}{p_t}$', #<-- VARIABLE NAME
                                             ylabel = 'events / 0.01',
                                             title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                             annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                             overwrite = False
                                            )
        dataset_lNj1j2_p_perp_to_pt.make_1d_hist(
                                             fig_dict = fig_agg.get_fig_dict(),
                                             figname = 'mWR_' + str(mWR) + '_' + 'lNj1j2_p_perp_to_pt', #<-- VARIABLE NAME
                                             axname = 'ax_a',
                                             savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lNj1j2_p_perp_to_pt.png', #<-- VARIABLE NAME,
                                             label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                             legend_loc = [1, 1],
                                             data = dataset_lNj1j2_p_perp_to_pt.get_datalist(1), #<-- VARIABLE NAME
                                             bin_width = 0.01,
                                             x_bounds = [-1, 1],
                                             y_bounds = [10**0, 10**8],
                                             xscale = 'linear',
                                             yscale = 'log',
                                             xlabel = r'$l_N j_1 j_2$ $\frac{p_{\perp}}{p_t}$', #<-- VARIABLE NAME
                                             ylabel = 'events / 0.01',
                                             title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                             annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                             overwrite = False
                                            )
        dataset_lW_p_perp_to_par.make_1d_hist(
                                             fig_dict = fig_agg.get_fig_dict(),
                                             figname = 'mWR_' + str(mWR) + '_' + 'lW_p_perp_to_par', #<-- VARIABLE NAME
                                             axname = 'ax_a',
                                             savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lW_p_perp_to_par.png', #<-- VARIABLE NAME,
                                             label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                             legend_loc = [1, 1],
                                             data = dataset_lW_p_perp_to_par.get_datalist(1), #<-- VARIABLE NAME
                                             bin_width = 0.01,
                                             x_bounds = [-1, 1],
                                             y_bounds = [10**0, 10**8],
                                             xscale = 'linear',
                                             yscale = 'log',
                                             xlabel = r'$l_W$ $\frac{p_{\perp}}{p_{||}}$', #<-- VARIABLE NAME
                                             ylabel = 'events / 0.01',
                                             title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                             annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                             overwrite = False
                                            )
        dataset_lN_p_perp_to_par.make_1d_hist(
                                             fig_dict = fig_agg.get_fig_dict(),
                                             figname = 'mWR_' + str(mWR) + '_' + 'lN_p_perp_to_par', #<-- VARIABLE NAME
                                             axname = 'ax_a',
                                             savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lN_p_perp_to_par.png', #<-- VARIABLE NAME,
                                             label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                             legend_loc = [1, 1],
                                             data = dataset_lN_p_perp_to_par.get_datalist(1), #<-- VARIABLE NAME
                                             bin_width = 0.01,
                                             x_bounds = [-1, 1],
                                             y_bounds = [10**0, 10**8],
                                             xscale = 'linear',
                                             yscale = 'log',
                                             xlabel = r'$l_N$ $\frac{p_{\perp}}{p_{||}}$', #<-- VARIABLE NAME
                                             ylabel = 'events / 0.01',
                                             title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                             annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                             overwrite = False
                                            )
        dataset_lNj1j2_p_perp_to_par.make_1d_hist(
                                             fig_dict = fig_agg.get_fig_dict(),
                                             figname = 'mWR_' + str(mWR) + '_' + 'lNj1j2_p_perp_to_par', #<-- VARIABLE NAME
                                             axname = 'ax_a',
                                             savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lNj1j2_p_perp_to_par.png', #<-- VARIABLE NAME,
                                             label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                             legend_loc = [1, 1],
                                             data = dataset_lNj1j2_p_perp_to_par.get_datalist(1), #<-- VARIABLE NAME
                                             bin_width = 0.01,
                                             x_bounds = [-1, 1],
                                             y_bounds = [10**0, 10**8],
                                             xscale = 'linear',
                                             yscale = 'log',
                                             xlabel = r'$l_N j_1 j_2$ $\frac{p_{\perp}}{p_{||}}$', #<-- VARIABLE NAME
                                             ylabel = 'events / 0.01',
                                             title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                             annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                             overwrite = False
                                            )

        dataset_lW_p_par_smeared.make_1d_hist(
                                             fig_dict = fig_agg.get_fig_dict(),
                                             figname = 'mWR_' + str(mWR) + '_' + 'lW_p_par_smeared', #<-- VARIABLE NAME
                                             axname = 'ax_a',
                                             savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lW_p_par_smeared.png', #<-- VARIABLE NAME,
                                             label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                             legend_loc = [1, 1],
                                             data = dataset_lW_p_par_smeared.get_datalist(1), #<-- VARIABLE NAME
                                             bin_width = 50,
                                             x_bounds = [-2*10**3, 2*10**3],
                                             y_bounds = [10**0, 10**8],
                                             xscale = 'linear',
                                             yscale = 'log',
                                             xlabel = r'$l_W$ $p_{||}$', #<-- VARIABLE NAME
                                             ylabel = 'events / 50 GeV',
                                             title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                             annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                             overwrite = False
                                            )
        dataset_lN_p_par_smeared.make_1d_hist(
                                             fig_dict = fig_agg.get_fig_dict(),
                                             figname = 'mWR_' + str(mWR) + '_' + 'lN_p_par_smeared', #<-- VARIABLE NAME
                                             axname = 'ax_a',
                                             savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lN_p_par_smeared.png', #<-- VARIABLE NAME,
                                             label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                             legend_loc = [1, 1],
                                             data = dataset_lN_p_par_smeared.get_datalist(1), #<-- VARIABLE NAME
                                             bin_width = 50,
                                             x_bounds = [-2*10**3, 2*10**3],
                                             y_bounds = [10**0, 10**8],
                                             xscale = 'linear',
                                             yscale = 'log',
                                             xlabel = r'$l_N$ $p_{||}$', #<-- VARIABLE NAME
                                             ylabel = 'events / 50 GeV',
                                             title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                             annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                             overwrite = False
                                            )
        dataset_lNj1j2_p_par_smeared.make_1d_hist(
                                             fig_dict = fig_agg.get_fig_dict(),
                                             figname = 'mWR_' + str(mWR) + '_' + 'lNj1j2_p_par_smeared', #<-- VARIABLE NAME
                                             axname = 'ax_a',
                                             savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lNj1j2_p_par_smeared.png', #<-- VARIABLE NAME,
                                             label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                             legend_loc = [1, 1],
                                             data = dataset_lNj1j2_p_par_smeared.get_datalist(1), #<-- VARIABLE NAME
                                             bin_width = 50,
                                             x_bounds = [-2*10**3, 2*10**3],
                                             y_bounds = [10**0, 10**8],
                                             xscale = 'linear',
                                             yscale = 'log',
                                             xlabel = r'$l_N j_1 j_2$ $p_{||}$', #<-- VARIABLE NAME
                                             ylabel = 'events / 50 GeV',
                                             title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                             annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                             overwrite = False
                                            )
        dataset_lWlNj1j2_p_par_smeared.make_1d_hist(
                                             fig_dict = fig_agg.get_fig_dict(),
                                             figname = 'mWR_' + str(mWR) + '_' + 'lWlNj1j2_p_par_smeared', #<-- VARIABLE NAME
                                             axname = 'ax_a',
                                             savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lWlNj1j2_p_par_smeared.png', #<-- VARIABLE NAME,
                                             label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                             legend_loc = [1, 1],
                                             data = dataset_lWlNj1j2_p_par_smeared.get_datalist(1), #<-- VARIABLE NAME
                                             bin_width = 10,
                                             x_bounds = [-5*10**2, 5*10**2],
                                             y_bounds = [10**0, 10**8],
                                             xscale = 'linear',
                                             yscale = 'log',
                                             xlabel = r'$l_W l_N j_1 j_2$ $p_{||}$', #<-- VARIABLE NAME
                                             ylabel = 'events / 10 GeV',
                                             title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                             annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                             overwrite = False
                                            )
        dataset_lW_p_perp_smeared.make_1d_hist(
                                             fig_dict = fig_agg.get_fig_dict(),
                                             figname = 'mWR_' + str(mWR) + '_' + 'lW_p_perp_smeared', #<-- VARIABLE NAME
                                             axname = 'ax_a',
                                             savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lW_p_perp_smeared.png', #<-- VARIABLE NAME,
                                             label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                             legend_loc = [1, 1],
                                             data = dataset_lW_p_perp_smeared.get_datalist(1), #<-- VARIABLE NAME
                                             bin_width = 0.01,
                                             x_bounds = [-0.5, 0.5],
                                             y_bounds = [10**0, 10**8],
                                             xscale = 'linear',
                                             yscale = 'log',
                                             xlabel = r'$l_W$ $p_{\perp}$', #<-- VARIABLE NAME
                                             ylabel = 'events / 0.01 GeV',
                                             title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                             annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                             overwrite = False
                                            )
        dataset_lN_p_perp_smeared.make_1d_hist(
                                             fig_dict = fig_agg.get_fig_dict(),
                                             figname = 'mWR_' + str(mWR) + '_' + 'lN_p_perp_smeared', #<-- VARIABLE NAME
                                             axname = 'ax_a',
                                             savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lN_p_perp_smeared.png', #<-- VARIABLE NAME,
                                             label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                             legend_loc = [1, 1],
                                             data = dataset_lN_p_perp_smeared.get_datalist(1), #<-- VARIABLE NAME
                                             bin_width = 50,
                                             x_bounds = [-1*10**3, 1*10**3],
                                             y_bounds = [10**0, 10**8],
                                             xscale = 'linear',
                                             yscale = 'log',
                                             xlabel = r'$l_N$ $p_{\perp}$', #<-- VARIABLE NAME
                                             ylabel = 'events / 50 GeV',
                                             title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                             annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                             overwrite = False
                                            )
        dataset_lNj1j2_p_perp_smeared.make_1d_hist(
                                             fig_dict = fig_agg.get_fig_dict(),
                                             figname = 'mWR_' + str(mWR) + '_' + 'lNj1j2_p_perp_smeared', #<-- VARIABLE NAME
                                             axname = 'ax_a',
                                             savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lNj1j2_p_perp_smeared.png', #<-- VARIABLE NAME,
                                             label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                             legend_loc = [1, 1],
                                             data = dataset_lNj1j2_p_perp_smeared.get_datalist(1), #<-- VARIABLE NAME
                                             bin_width = 10,
                                             x_bounds = [-5*10**2, 5*10**2],
                                             y_bounds = [10**0, 10**8],
                                             xscale = 'linear',
                                             yscale = 'log',
                                             xlabel = r'$l_N j_1 j_2$ $p_{\perp}$', #<-- VARIABLE NAME
                                             ylabel = 'events / 10 GeV',
                                             title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                             annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                             overwrite = False
                                            )
        dataset_lWlNj1j2_p_perp_smeared.make_1d_hist(
                                             fig_dict = fig_agg.get_fig_dict(),
                                             figname = 'mWR_' + str(mWR) + '_' + 'lWlNj1j2_p_perp_smeared', #<-- VARIABLE NAME
                                             axname = 'ax_a',
                                             savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lWlNj1j2_p_perp_smeared.png', #<-- VARIABLE NAME,
                                             label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                             legend_loc = [1, 1],
                                             data = dataset_lWlNj1j2_p_perp_smeared.get_datalist(1), #<-- VARIABLE NAME
                                             bin_width = 10,
                                             x_bounds = [-5*10**2, 5*10**2],
                                             y_bounds = [10**0, 10**8],
                                             xscale = 'linear',
                                             yscale = 'log',
                                             xlabel = r'$l_W l_N j_1 j_2$ $p_{\perp}$', #<-- VARIABLE NAME
                                             ylabel = 'events / 10 GeV',
                                             title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                             annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                             overwrite = False
                                            )
        dataset_lW_p_par_to_pt_smeared.make_1d_hist(
                                             fig_dict = fig_agg.get_fig_dict(),
                                             figname = 'mWR_' + str(mWR) + '_' + 'lW_p_par_to_pt_smeared', #<-- VARIABLE NAME
                                             axname = 'ax_a',
                                             savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lW_p_par_to_pt_smeared.png', #<-- VARIABLE NAME,
                                             label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                             legend_loc = [1, 1],
                                             data = dataset_lW_p_par_to_pt_smeared.get_datalist(1), #<-- VARIABLE NAME
                                             bin_width = 0.01,
                                             x_bounds = [-1, 1],
                                             y_bounds = [10**0, 10**8],
                                             xscale = 'linear',
                                             yscale = 'log',
                                             xlabel = r'$l_W$ $\frac{p_{||}}{p_t}$', #<-- VARIABLE NAME
                                             ylabel = 'events / 0.01',
                                             title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                             annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                             overwrite = False
                                            )
        dataset_lN_p_par_to_pt_smeared.make_1d_hist(
                                             fig_dict = fig_agg.get_fig_dict(),
                                             figname = 'mWR_' + str(mWR) + '_' + 'lN_p_par_to_pt_smeared', #<-- VARIABLE NAME
                                             axname = 'ax_a',
                                             savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lN_p_par_to_pt_smeared.png', #<-- VARIABLE NAME,
                                             label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                             legend_loc = [1, 1],
                                             data = dataset_lN_p_par_to_pt_smeared.get_datalist(1), #<-- VARIABLE NAME
                                             bin_width = 0.01,
                                             x_bounds = [-1, 1],
                                             y_bounds = [10**0, 10**8],
                                             xscale = 'linear',
                                             yscale = 'log',
                                             xlabel = r'$l_N$ $\frac{p_{||}}{p_t}$', #<-- VARIABLE NAME
                                             ylabel = 'events / 0.01',
                                             title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                             annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                             overwrite = False
                                            )
        dataset_lNj1j2_p_par_to_pt_smeared.make_1d_hist(
                                             fig_dict = fig_agg.get_fig_dict(),
                                             figname = 'mWR_' + str(mWR) + '_' + 'lNj1j2_p_par_to_pt_smeared', #<-- VARIABLE NAME
                                             axname = 'ax_a',
                                             savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lNj1j2_p_par_to_pt_smeared.png', #<-- VARIABLE NAME,
                                             label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                             legend_loc = [1, 1],
                                             data = dataset_lNj1j2_p_par_to_pt_smeared.get_datalist(1), #<-- VARIABLE NAME
                                             bin_width = 0.01,
                                             x_bounds = [-1, 1],
                                             y_bounds = [10**0, 10**8],
                                             xscale = 'linear',
                                             yscale = 'log',
                                             xlabel = r'$l_N j_1 j_2$ $\frac{p_{||}}{p_t}$', #<-- VARIABLE NAME
                                             ylabel = 'events / 0.01',
                                             title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                             annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                             overwrite = False
                                            )
        dataset_lWlNj1j2_p_par_to_pt_smeared.make_1d_hist(
                                             fig_dict = fig_agg.get_fig_dict(),
                                             figname = 'mWR_' + str(mWR) + '_' + 'lWlNj1j2_p_par_to_pt_smeared', #<-- VARIABLE NAME
                                             axname = 'ax_a',
                                             savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lWlNj1j2_p_par_to_pt_smeared.png', #<-- VARIABLE NAME,
                                             label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                             legend_loc = [1, 1],
                                             data = dataset_lWlNj1j2_p_par_to_pt_smeared.get_datalist(1), #<-- VARIABLE NAME
                                             bin_width = 0.01,
                                             x_bounds = [-1, 1],
                                             y_bounds = [10**0, 10**8],
                                             xscale = 'linear',
                                             yscale = 'log',
                                             xlabel = r'$l_W l_N j_1 j_2$ $\frac{p_{||}}{p_t}$', #<-- VARIABLE NAME
                                             ylabel = 'events / 0.01',
                                             title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                             annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                             overwrite = False
                                            )
        dataset_lW_p_perp_to_pt_smeared.make_1d_hist(
                                             fig_dict = fig_agg.get_fig_dict(),
                                             figname = 'mWR_' + str(mWR) + '_' + 'lW_p_perp_to_pt_smeared', #<-- VARIABLE NAME
                                             axname = 'ax_a',
                                             savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lW_p_perp_to_pt_smeared.png', #<-- VARIABLE NAME,
                                             label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                             legend_loc = [1, 1],
                                             data = dataset_lW_p_perp_to_pt_smeared.get_datalist(1), #<-- VARIABLE NAME
                                             bin_width = 0.01,
                                             x_bounds = [-1, 1],
                                             y_bounds = [10**0, 10**8],
                                             xscale = 'linear',
                                             yscale = 'log',
                                             xlabel = r'$l_W$ $\frac{p_{\perp}}{p_t}$', #<-- VARIABLE NAME
                                             ylabel = 'events / 0.01',
                                             title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                             annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                             overwrite = False
                                            )
        dataset_lN_p_perp_to_pt_smeared.make_1d_hist(
                                             fig_dict = fig_agg.get_fig_dict(),
                                             figname = 'mWR_' + str(mWR) + '_' + 'lN_p_perp_to_pt_smeared', #<-- VARIABLE NAME
                                             axname = 'ax_a',
                                             savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lN_p_perp_to_pt_smeared.png', #<-- VARIABLE NAME,
                                             label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                             legend_loc = [1, 1],
                                             data = dataset_lN_p_perp_to_pt_smeared.get_datalist(1), #<-- VARIABLE NAME
                                             bin_width = 0.01,
                                             x_bounds = [-1, 1],
                                             y_bounds = [10**0, 10**8],
                                             xscale = 'linear',
                                             yscale = 'log',
                                             xlabel = r'$l_N$ $\frac{p_{\perp}}{p_t}$', #<-- VARIABLE NAME
                                             ylabel = 'events / 0.01',
                                             title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                             annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                             overwrite = False
                                            )
        dataset_lNj1j2_p_perp_to_pt_smeared.make_1d_hist(
                                             fig_dict = fig_agg.get_fig_dict(),
                                             figname = 'mWR_' + str(mWR) + '_' + 'lNj1j2_p_perp_to_pt_smeared', #<-- VARIABLE NAME
                                             axname = 'ax_a',
                                             savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lNj1j2_p_perp_to_pt_smeared.png', #<-- VARIABLE NAME,
                                             label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                             legend_loc = [1, 1],
                                             data = dataset_lNj1j2_p_perp_to_pt_smeared.get_datalist(1), #<-- VARIABLE NAME
                                             bin_width = 0.01,
                                             x_bounds = [-1, 1],
                                             y_bounds = [10**0, 10**8],
                                             xscale = 'linear',
                                             yscale = 'log',
                                             xlabel = r'$l_N j_1 j_2$ $\frac{p_{\perp}}{p_t}$', #<-- VARIABLE NAME
                                             ylabel = 'events / 0.01',
                                             title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                             annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                             overwrite = False
                                            )
        dataset_lWlNj1j2_p_perp_to_pt_smeared.make_1d_hist(
                                             fig_dict = fig_agg.get_fig_dict(),
                                             figname = 'mWR_' + str(mWR) + '_' + 'lWlNj1j2_p_perp_to_pt_smeared', #<-- VARIABLE NAME
                                             axname = 'ax_a',
                                             savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lWlNj1j2_p_perp_to_pt_smeared.png', #<-- VARIABLE NAME,
                                             label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                             legend_loc = [1, 1],
                                             data = dataset_lWlNj1j2_p_perp_to_pt_smeared.get_datalist(1), #<-- VARIABLE NAME
                                             bin_width = 0.01,
                                             x_bounds = [-1, 1],
                                             y_bounds = [10**0, 10**8],
                                             xscale = 'linear',
                                             yscale = 'log',
                                             xlabel = r'$l_W l_N j_1 j_2$ $\frac{p_{\perp}}{p_t}$', #<-- VARIABLE NAME
                                             ylabel = 'events / 0.01',
                                             title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                             annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                             overwrite = False
                                            )
        dataset_lW_p_perp_to_par_smeared.make_1d_hist(
                                             fig_dict = fig_agg.get_fig_dict(),
                                             figname = 'mWR_' + str(mWR) + '_' + 'lW_p_perp_to_par_smeared', #<-- VARIABLE NAME
                                             axname = 'ax_a',
                                             savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lW_p_perp_to_par_smeared.png', #<-- VARIABLE NAME,
                                             label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                             legend_loc = [1, 1],
                                             data = dataset_lW_p_perp_to_par_smeared.get_datalist(1), #<-- VARIABLE NAME
                                             bin_width = 0.01,
                                             x_bounds = [-1, 1],
                                             y_bounds = [10**0, 10**8],
                                             xscale = 'linear',
                                             yscale = 'log',
                                             xlabel = r'$l_W$ $\frac{p_{\perp}}{p_{||}}$', #<-- VARIABLE NAME
                                             ylabel = 'events / 0.01',
                                             title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                             annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                             overwrite = False
                                            )
        dataset_lN_p_perp_to_par_smeared.make_1d_hist(
                                             fig_dict = fig_agg.get_fig_dict(),
                                             figname = 'mWR_' + str(mWR) + '_' + 'lN_p_perp_to_par_smeared', #<-- VARIABLE NAME
                                             axname = 'ax_a',
                                             savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lN_p_perp_to_par_smeared.png', #<-- VARIABLE NAME,
                                             label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                             legend_loc = [1, 1],
                                             data = dataset_lN_p_perp_to_par_smeared.get_datalist(1), #<-- VARIABLE NAME
                                             bin_width = 0.01,
                                             x_bounds = [-1, 1],
                                             y_bounds = [10**0, 10**8],
                                             xscale = 'linear',
                                             yscale = 'log',
                                             xlabel = r'$l_N$ $\frac{p_{\perp}}{p_{||}}$', #<-- VARIABLE NAME
                                             ylabel = 'events / 0.01',
                                             title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                             annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                             overwrite = False
                                            )
        dataset_lNj1j2_p_perp_to_par_smeared.make_1d_hist(
                                             fig_dict = fig_agg.get_fig_dict(),
                                             figname = 'mWR_' + str(mWR) + '_' + 'lNj1j2_p_perp_to_par_smeared', #<-- VARIABLE NAME
                                             axname = 'ax_a',
                                             savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lNj1j2_p_perp_to_par_smeared.png', #<-- VARIABLE NAME,
                                             label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                             legend_loc = [1, 1],
                                             data = dataset_lNj1j2_p_perp_to_par_smeared.get_datalist(1), #<-- VARIABLE NAME
                                             bin_width = 0.01,
                                             x_bounds = [-1, 1],
                                             y_bounds = [10**0, 10**8],
                                             xscale = 'linear',
                                             yscale = 'log',
                                             xlabel = r'$l_N j_1 j_2$ $\frac{p_{\perp}}{p_{||}}$', #<-- VARIABLE NAME
                                             ylabel = 'events / 0.01',
                                             title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                             annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                             overwrite = False
                                            )
        dataset_lWlNj1j2_p_perp_to_par_smeared.make_1d_hist(
                                             fig_dict = fig_agg.get_fig_dict(),
                                             figname = 'mWR_' + str(mWR) + '_' + 'lWlNj1j2_p_perp_to_par_smeared', #<-- VARIABLE NAME
                                             axname = 'ax_a',
                                             savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lWlNj1j2_p_perp_to_par_smeared.png', #<-- VARIABLE NAME,
                                             label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                             legend_loc = [1, 1],
                                             data = dataset_lWlNj1j2_p_perp_to_par_smeared.get_datalist(1), #<-- VARIABLE NAME
                                             bin_width = 0.01,
                                             x_bounds = [-1, 1],
                                             y_bounds = [10**0, 10**8],
                                             xscale = 'linear',
                                             yscale = 'log',
                                             xlabel = r'$l_W l_N j_1 j_2$ $\frac{p_{\perp}}{p_{||}}$', #<-- VARIABLE NAME
                                             ylabel = 'events / 0.01',
                                             title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                             annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                             overwrite = False
                                            )

        dataset_lWlNj1j2_mass_smeared_lW_align.make_1d_hist(
                                             fig_dict = fig_agg.get_fig_dict(),
                                             figname = 'mWR_' + str(mWR) + '_' + 'lWlNj1j2_mass_smeared_lW_align', #<-- VARIABLE NAME
                                             axname = 'ax_a',
                                             savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lWlNj1j2_mass_smeared_lW_align.png', #<-- VARIABLE NAME,
                                             label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                             legend_loc = [1, 1],
                                             data = dataset_lWlNj1j2_mass_smeared_lW_align.get_datalist(1), #<-- VARIABLE NAME
                                             bin_width = 50,
                                             x_bounds = [0, 4500],
                                             y_bounds = [10**0, 10**8],
                                             xscale = 'linear',
                                             yscale = 'log',
                                             xlabel = r'$l_W l_N j_1 j_2$ mass', #<-- VARIABLE NAME
                                             ylabel = 'events / 50 GeV',
                                             title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                             annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                             overwrite = False
                                            )
        #############################
        #############################

    if make_smear_frac_figs == True:
        if mN != 1:
            fit_gauss = True
        else:
            fit_gauss = False

        if fit_gauss == True:
            print(f'\nfitting gaussian to smear frac hist for (mWR, mN) = ({mWR}, {mN})\n')
            ############################################
            ### fit gaussian to jet smear frac hists ###
            ############################################

            #####################################
            ### create the hist data datasets ###
            #####################################
            j1_e_smear_frac_bin_edges_left_ind = np.where(j1_e_smear_frac_bin_edges >= -1)[0][0]
            j2_e_smear_frac_bin_edges_left_ind = np.where(j2_e_smear_frac_bin_edges >= -1)[0][0]

            j1_e_smear_frac_bin_edges = j1_e_smear_frac_bin_edges[j1_e_smear_frac_bin_edges_left_ind:] # remove all elements from the bin edges that are less than -1 since gaussian truncates there
            j2_e_smear_frac_bin_edges = j2_e_smear_frac_bin_edges[j2_e_smear_frac_bin_edges_left_ind:]

    #        print(f'j1_e_smear_frac_bin_edges:\n{j1_e_smear_frac_bin_edges}')
    #        print(f'j2_e_smear_frac_bin_edges:\n{j2_e_smear_frac_bin_edges}')

            j1_e_smear_frac_bin_counts = j1_e_smear_frac_bin_counts[j1_e_smear_frac_bin_edges_left_ind:] # remove the corresponding bin counts
            j2_e_smear_frac_bin_counts = j2_e_smear_frac_bin_counts[j2_e_smear_frac_bin_edges_left_ind:]

            j1_e_smear_frac_bin_centers = (j1_e_smear_frac_bin_edges[0:-1] + j1_e_smear_frac_bin_edges[1:])/2
            j2_e_smear_frac_bin_centers = (j2_e_smear_frac_bin_edges[0:-1] + j2_e_smear_frac_bin_edges[1:])/2

            dataset_j1_e_smear_frac = DataSet(
                                              datalist_1 = j1_e_smear_frac_bin_centers,
                                              datalist_2 = j1_e_smear_frac_bin_counts,
                                             )
            dataset_j2_e_smear_frac = DataSet(
                                              datalist_1 = j2_e_smear_frac_bin_centers,
                                              datalist_2 = j2_e_smear_frac_bin_counts,
                                             )
            #####################################
            #####################################

            ############################
            ### set bounds and seeds ###
            ############################
            j1_e_smear_frac_x_scale = [10**5, 1, 1,] # [a_scale, mu_scale, sigma_scale]
            j2_e_smear_frac_x_scale = [10**5, 1, 1,]

            # keep a bounds in scientific notation (and make sure x_scale matches)
            #                              a        mu   sigma
            #                              ||       ||    ||
            #                              ||       ||    ||
            #                              \/       \/    \/
            j1_e_smear_frac_bounds = ([1.00*10**4, -0.50, 0.00, # low bounds
                                      ],
                                      [1.00*10**7,  0.50, 1.00, # high bounds
                                      ])

            j2_e_smear_frac_bounds = ([1.00*10**4, -0.50, 0.00, # low bounds
                                      ],
                                      [1.00*10**7,  0.50, 1.00, # high bounds
                                      ])

            for i in range(int(len(j1_e_smear_frac_x_scale)/3)): # choose seeds to be midpoints of bounds
                j1_e_smear_frac_a_seed     = (j1_e_smear_frac_bounds[0][3*i    ] + j1_e_smear_frac_bounds[1][3*i    ])/2
                j1_e_smear_frac_mu_seed    = (j1_e_smear_frac_bounds[0][3*i + 1] + j1_e_smear_frac_bounds[1][3*i + 1])/2
                j1_e_smear_frac_sigma_seed = (j1_e_smear_frac_bounds[0][3*i + 2] + j1_e_smear_frac_bounds[1][3*i + 2])/2

                j2_e_smear_frac_a_seed     = (j2_e_smear_frac_bounds[0][3*i    ] + j2_e_smear_frac_bounds[1][3*i    ])/2
                j2_e_smear_frac_mu_seed    = (j2_e_smear_frac_bounds[0][3*i + 1] + j2_e_smear_frac_bounds[1][3*i + 1])/2
                j2_e_smear_frac_sigma_seed = (j2_e_smear_frac_bounds[0][3*i + 2] + j2_e_smear_frac_bounds[1][3*i + 2])/2
            ############################
            ############################

            #######################
            ### perform the fit ###
            #######################
            # order of initParams: amplitude, mean, std_dev
            j1_e_smear_frac_params, j1_e_smear_frac_param_uncertainty, j1_e_smear_frac_chi_sq = dataset_j1_e_smear_frac.fit_to_gauss_sum(
                                                                                                                                         dataset_j1_e_smear_frac.get_datalist(1),
                                                                                                                                         dataset_j1_e_smear_frac.get_datalist(2),
                                                                                                                                         j1_e_smear_frac_x_scale,
                                                                                                                                         j1_e_smear_frac_bounds,
                                                                                                                                         j1_e_smear_frac_a_seed,
                                                                                                                                         j1_e_smear_frac_mu_seed,
                                                                                                                                         j1_e_smear_frac_sigma_seed,
                                                                                                                                        )
            j2_e_smear_frac_params, j2_e_smear_frac_param_uncertainty, j2_e_smear_frac_chi_sq = dataset_j2_e_smear_frac.fit_to_gauss_sum(
                                                                                                                                         dataset_j2_e_smear_frac.get_datalist(1),
                                                                                                                                         dataset_j2_e_smear_frac.get_datalist(2),
                                                                                                                                         j2_e_smear_frac_x_scale,
                                                                                                                                         j2_e_smear_frac_bounds,
                                                                                                                                         j2_e_smear_frac_a_seed,
                                                                                                                                         j2_e_smear_frac_mu_seed,
                                                                                                                                         j2_e_smear_frac_sigma_seed,
                                                                                                                                        )

            j1_e_smear_frac_a     = j1_e_smear_frac_params[0]
            j1_e_smear_frac_mu    = j1_e_smear_frac_params[1]
            j1_e_smear_frac_sigma = j1_e_smear_frac_params[2]

            j2_e_smear_frac_a     = j2_e_smear_frac_params[0]
            j2_e_smear_frac_mu    = j2_e_smear_frac_params[1]
            j2_e_smear_frac_sigma = j2_e_smear_frac_params[2]

    #        print(f'---------------- mWR: {mWR}, mN: {mN} --------------------------------------------------------------------------------------------')
    #        print(f'j1_a: {j1_e_smear_frac_a}\nj1_mu: {j1_e_smear_frac_mu}\nj1_sigma: {j1_e_smear_frac_sigma}\n')
    #        print(f'j2_a: {j2_e_smear_frac_a}\nj2_mu: {j2_e_smear_frac_mu}\nj2_sigma: {j2_e_smear_frac_sigma}\n')
            #######################
            #######################

            ###############################
            ### make gaussian plot data ###
            ###############################
            j1_e_smear_frac_gauss_x_vals = np.linspace(-3, 3, 1001)
            j2_e_smear_frac_gauss_x_vals = np.linspace(-3, 3, 1001)

            j1_e_smear_frac_gauss_y_vals = j1_e_smear_frac_a*np.exp(-0.5*np.power((j1_e_smear_frac_gauss_x_vals - j1_e_smear_frac_mu)/j1_e_smear_frac_sigma, 2))
            j2_e_smear_frac_gauss_y_vals = j2_e_smear_frac_a*np.exp(-0.5*np.power((j2_e_smear_frac_gauss_x_vals - j2_e_smear_frac_mu)/j2_e_smear_frac_sigma, 2))

            dataset_j1_e_smear_frac_gauss = DataSet(
                                                    datalist_1 = j1_e_smear_frac_gauss_x_vals,
                                                    datalist_2 = j1_e_smear_frac_gauss_y_vals
                                                   )
            dataset_j2_e_smear_frac_gauss = DataSet(
                                                    datalist_1 = j2_e_smear_frac_gauss_x_vals,
                                                    datalist_2 = j2_e_smear_frac_gauss_y_vals
                                                   )
            ###############################
            ###############################

            ######################################
            ### plot gaussians over histograms ###
            ######################################
            dataset_j1_e_smear_frac_gauss.make_line_plot(
                                                         fig_dict = fig_agg.get_fig_dict(),
                                                         figname = 'mWR_' + str(mWR) + '_' + 'j1_e_smear_frac', #<-- VARIABLE NAME,
                                                         axname = 'ax_a',
                                                         savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_j1_e_smear_frac.png', #<-- VAR NAME,
                                                         label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV fit: ' + r'$\sigma$' + f' = {np.round(j1_e_smear_frac_sigma, 3)}',
                                                         legend_loc = [1, 1],
                                                         y_data = dataset_j1_e_smear_frac_gauss.get_datalist(2),
                                                         x_data = dataset_j1_e_smear_frac_gauss.get_datalist(1),
                                                         x_bounds = [-2, 2],
                                                         y_bounds = [10**0, 10**8],
                                                         xscale = 'linear',
                                                         yscale = 'log',
                                                         xlabel = None,
                                                         ylabel = None,
                                                         title = None,
                                                         annotations = None,
                                                         overwrite = False
                                                        )
            dataset_j2_e_smear_frac_gauss.make_line_plot(
                                                         fig_dict = fig_agg.get_fig_dict(),
                                                         figname = 'mWR_' + str(mWR) + '_' + 'j2_e_smear_frac', #<-- VARIABLE NAME,
                                                         axname = 'ax_a',
                                                         savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_j2_e_smear_frac.png', #<-- VAR NAME,
                                                         label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV fit: ' + r'$\sigma$' + f' = {np.round(j2_e_smear_frac_sigma, 3)}',
                                                         legend_loc = [1, 1],
                                                         y_data = dataset_j2_e_smear_frac_gauss.get_datalist(2),
                                                         x_data = dataset_j2_e_smear_frac_gauss.get_datalist(1),
                                                         x_bounds = [-2, 2],
                                                         y_bounds = [10**0, 10**8],
                                                         xscale = 'linear',
                                                         yscale = 'log',
                                                         xlabel = None,
                                                         ylabel = None,
                                                         title = None,
                                                         annotations = None,
                                                         overwrite = False
                                                        )
            ######################################
            ######################################

            ############################################
            ############################################

    if make_jet_p_correction_fac_figs == True:
        print(f'\nplotting jet p correction factor figs for (mWR, mN) = ({mWR}, {mN})\n')
        ##############################################
        ### jet momentum correction factor figures ###
        ##############################################
#        dataset_event_0_jet_facs.make_scatterplot(
#                                                  fig_dict = fig_agg.get_fig_dict(),
#                                                  figname = 'mWR_' + str(mWR) + '_mN_' + str(mN) + '_event_0_s_2_vs_s_1', #<-- VARIABLE NAME,
#                                                  axname = 'ax_a',
#                                                  savepath = 'jet_cor_fac_figs/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_mN_' + str(mN) + '_event_0_s_2_vs_s_1.png', #<-- VAR NAME,
#                                                  label = r'$m_{N}$' + ' = ' + str(mN),
#                                                  legend_loc = [1, 1],
#                                                  y_data = dataset_event_0_jet_facs.get_datalist(2),
#                                                  x_data = dataset_event_0_jet_facs.get_datalist(1),
#                                                  #x_bounds = [0, 20],
#                                                  #y_bounds = [-400, 400],
#                                                  xlabel = 'jet 1 p correction factor ' + r'$(s_1)$',
#                                                  ylabel = 'jet 2 p correction factor ' + r'$(s_2)$',
#                                                  title = None,
#                                                  annotations = None,
#                                                  overwrite = False
#                                                 )
#        dataset_event_0_chi_sq.make_scatterplot(
#                                                fig_dict = fig_agg.get_fig_dict(),
#                                                figname = 'mWR_' + str(mWR) + '_mN_' + str(mN) + '_event_0_chi_sq_vs_s_1', #<-- VARIABLE NAME,
#                                                axname = 'ax_a',
#                                                savepath = 'jet_cor_fac_figs/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_mN_' + str(mN) + '_event_0_chi_sq_vs_s_1.png', #<-- VAR NAME,
#                                                label = r'$m_{N}$' + ' = ' + str(mN),
#                                                legend_loc = [1, 1],
#                                                y_data = dataset_event_0_chi_sq.get_datalist(2),
#                                                x_data = dataset_event_0_chi_sq.get_datalist(1),
#                                                #x_bounds = [0, 20],
#                                                #y_bounds = [-400, 400],
#                                                xlabel = 'jet 1 p correction factor ' + r'$(s_1)$',
#                                                ylabel = r'$\chi^2(s_1)$',
#                                                title = None,
#                                                annotations = None,
#                                                overwrite = False
#                                               )
        dataset_best_s_1_array.make_1d_hist(
                                             fig_dict = fig_agg.get_fig_dict(),
                                             figname = 'mWR_' + str(mWR) + '_' + 'j1_correction_factor', #<-- VARIABLE NAME
                                             axname = 'ax_a',
                                             savepath = 'jet_cor_fac_figs/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_j1_correction_factor.png', #<-- VARIABLE NAME,
                                             label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                             legend_loc = [1, 1],
                                             data = dataset_best_s_1_array.get_datalist(1), #<-- VARIABLE NAME
                                             bin_width = 0.05,
                                             x_bounds = [-6, 6],
                                             y_bounds = [10**0, 10**8],
                                             xscale = 'linear',
                                             yscale = 'log',
                                             xlabel = r'$j_1$ correction factor ($s_1$)', #<-- VARIABLE NAME
                                             ylabel = 'events / 0.05',
                                             title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                             annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                             overwrite = False
                                            )
        dataset_best_s_2_array.make_1d_hist(
                                             fig_dict = fig_agg.get_fig_dict(),
                                             figname = 'mWR_' + str(mWR) + '_' + 'j2_correction_factor', #<-- VARIABLE NAME
                                             axname = 'ax_a',
                                             savepath = 'jet_cor_fac_figs/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_j2_correction_factor.png', #<-- VARIABLE NAME,
                                             label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                             legend_loc = [1, 1],
                                             data = dataset_best_s_2_array.get_datalist(1), #<-- VARIABLE NAME
                                             bin_width = 0.05,
                                             x_bounds = [-6, 6],
                                             y_bounds = [10**0, 10**8],
                                             xscale = 'linear',
                                             yscale = 'log',
                                             xlabel = r'$j_2$ correction factor ($s_2$)', #<-- VARIABLE NAME
                                             ylabel = 'events / 0.05',
                                             title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                             annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                             overwrite = False
                                            )
        ##############################################
        ##############################################

    if make_scaled_jet_var_figs == True:
        #################################
        ### scaled jets variable figs ###
        #################################
#        dataset_lNj1j2_scaled_p_par_to_lW_p_par.make_1d_hist(
#                                             fig_dict = fig_agg.get_fig_dict(),
#                                             figname = 'mWR_' + str(mWR) + '_' + 'lNj1j2_scaled_p_par_to_lW_p_par', #<-- VARIABLE NAME
#                                             axname = 'ax_a',
#                                             savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lNj1j2_scaled_p_par_to_lW_p_par.png', #<-- VARIABLE NAME,
#                                             label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
#                                             legend_loc = [1, 1],
#                                             data = dataset_lNj1j2_scaled_p_par_to_lW_p_par.get_datalist(1), #<-- VARIABLE NAME
#                                             bin_width = 0.001,
#                                             x_bounds = [-1.1, -0.9],
#                                             y_bounds = [10**0, 10**8],
#                                             xscale = 'linear',
#                                             yscale = 'log',
#                                             xlabel = r'$\frac{p_{||}, l_N j_1 j_2}{p_{||}, l_W}$ (corrected jets)', #<-- VARIABLE NAME
#                                             ylabel = 'events / 0.001',
#                                             title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
#                                             annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
#                                             overwrite = False
#                                            )
#        dataset_lNj1j2_old_p_par_to_lW_p_par.make_1d_hist(
#                                             fig_dict = fig_agg.get_fig_dict(),
#                                             figname = 'mWR_' + str(mWR) + '_' + 'lNj1j2_old_p_par_to_lW_p_par', #<-- VARIABLE NAME
#                                             axname = 'ax_a',
#                                             savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lNj1j2_old_p_par_to_lW_p_par.png', #<-- VARIABLE NAME,
#                                             label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
#                                             legend_loc = [1, 1],
#                                             data = dataset_lNj1j2_old_p_par_to_lW_p_par.get_datalist(1), #<-- VARIABLE NAME
#                                             bin_width = 0.1,
#                                             x_bounds = [-10, 10],
#                                             y_bounds = [10**0, 10**8],
#                                             xscale = 'linear',
#                                             yscale = 'log',
#                                             xlabel = r'$\frac{p_{||}, \ l_N j_1 j_2}{p_{||}, \ l_W}$ (uncorrected jets)', #<-- VARIABLE NAME
#                                             ylabel = 'events / 0.1',
#                                             title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
#                                             annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
#                                             overwrite = False
#                                            )
#        dataset_lWlNj1j2_p_par_smeared_scaled.make_1d_hist(
#                                             fig_dict = fig_agg.get_fig_dict(),
#                                             figname = 'mWR_' + str(mWR) + '_' + 'lWlNj1j2_scaled_p_par', #<-- VARIABLE NAME
#                                             axname = 'ax_a',
#                                             savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lWlNj1j2_scaled_p_par.png', #<-- VARIABLE NAME,
#                                             label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
#                                             legend_loc = [1, 1],
#                                             data = dataset_lWlNj1j2_p_par_smeared_scaled.get_datalist(1), #<-- VARIABLE NAME
#                                             bin_width = 0.1,
#                                             x_bounds = [-10, 10],
#                                             y_bounds = [10**0, 10**8],
#                                             xscale = 'linear',
#                                             yscale = 'log',
#                                             xlabel = r'$p_{||}, \ l_W l_N j_1 j_2$ (corrected jets)', #<-- VARIABLE NAME
#                                             ylabel = 'events / 0.1 GeV',
#                                             title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
#                                             annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
#                                             overwrite = False
#                                            )
#        dataset_lWlNj1j2_p_perp_smeared_scaled.make_1d_hist(
#                                             fig_dict = fig_agg.get_fig_dict(),
#                                             figname = 'mWR_' + str(mWR) + '_' + 'lWlNj1j2_scaled_p_perp', #<-- VARIABLE NAME
#                                             axname = 'ax_a',
#                                             savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lWlNj1j2_scaled_p_perp.png', #<-- VARIABLE NAME,
#                                             label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
#                                             legend_loc = [1, 1],
#                                             data = dataset_lWlNj1j2_p_perp_smeared_scaled.get_datalist(1), #<-- VARIABLE NAME
#                                             bin_width = 10,
#                                             x_bounds = [-500, 500],
#                                             y_bounds = [10**0, 10**8],
#                                             xscale = 'linear',
#                                             yscale = 'log',
#                                             xlabel = r'$p_{\perp}, \ l_W l_N j_1 j_2$ (corrected jets)', #<-- VARIABLE NAME
#                                             ylabel = 'events / 10 GeV',
#                                             title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
#                                             annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
#                                             overwrite = False
#                                            )
        lNj1j2_mass_smeared_scaled_bin_data = dataset_lNj1j2_mass_smeared_scaled.make_1d_hist(
                                                                                      fig_dict = fig_agg.get_fig_dict(),
                                                                                      figname = 'mWR_' + str(mWR) + '_' + 'lNj1j2_mass_smeared_scaled', #<-- VARIABLE NAME
                                                                                      axname = 'ax_a',
                                                                                      savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lNj1j2_mass_smeared_scaled.png', #<-- VARIABLE
                                                                                      label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                                                                      legend_loc = [1, 1],
                                                                                      data = dataset_lNj1j2_mass_smeared_scaled.get_datalist(1), #<-- VARIABLE NAME
                                                                                      bin_width = bin_width_3obj,
                                                                                      x_bounds = [lo_3obj_mass_x_bound, hi_3obj_mass_x_bound],
                                                                                      y_bounds = [lo_3obj_mass_y_bound, hi_3obj_mass_y_bound],
                                                                                      xscale = 'linear',
                                                                                      yscale = 'log',
                                                                                      xlabel = r'$l_N j_1 j_2$ mass [GeV]', #<-- VARIABLE NAME
                                                                                      ylabel = f'events / {bin_width_3obj} GeV',
                                                                                      title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                                                                      annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                                                                      overwrite = False
                                                                                     )
        lWlNj1j2_mass_smeared_scaled_bin_data = dataset_lWlNj1j2_mass_smeared_scaled.make_1d_hist(
                                                                                    fig_dict = fig_agg.get_fig_dict(),
                                                                                    figname = 'mWR_' + str(mWR) + '_' + 'lWlNj1j2_mass_smeared_scaled', #<-- VARIABLE NAME
                                                                                    axname = 'ax_a',
                                                                                    savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lWlNj1j2_mass_smeared_scaled.png', #<-- VARIABLE
                                                                                    label = r'$m_{N}$' + ' = ' + str(mN) + ' GeV',
                                                                                    legend_loc = [1, 1],
                                                                                    data = dataset_lWlNj1j2_mass_smeared_scaled.get_datalist(1), #<-- VARIABLE NAME
                                                                                    bin_width = bin_width_4obj,
                                                                                    x_bounds = [lo_4obj_mass_x_bound, hi_4obj_mass_x_bound],
                                                                                    y_bounds = [lo_4obj_mass_y_bound, hi_4obj_mass_y_bound],
                                                                                    xscale = 'linear',
                                                                                    yscale = 'log',
                                                                                    xlabel = r'$l_W l_N j_1 j_2$ mass [GeV]', #<-- VARIABLE NAME
                                                                                    ylabel = f'events / {bin_width_4obj} GeV',
                                                                                    title = 'Monte Carlo ' + r'$W_{R}$' + ' Decay Simulation',
                                                                                    annotations = [[r'$m_{W_{R}}$' + f' = {mWR} GeV', 0.06, 0.92]],
                                                                                    overwrite = False
                                                                                   )
        #################################
        #################################

    if analyze_tails == True:
        ###############################################################
        ### finding hist tail boundary edges for 3obj and 4obj mass ###
        ###############################################################
        lNj1j2_mass_smeared_bin_counts = lNj1j2_mass_smeared_bin_data[0]
        lNj1j2_mass_smeared_bin_edges  = lNj1j2_mass_smeared_bin_data[1]

        lWlNj1j2_mass_smeared_bin_counts = lWlNj1j2_mass_smeared_bin_data[0]
        lWlNj1j2_mass_smeared_bin_edges  = lWlNj1j2_mass_smeared_bin_data[1]

        lNj1j2_mass_smeared_scaled_bin_counts = lNj1j2_mass_smeared_scaled_bin_data[0]
        lNj1j2_mass_smeared_scaled_bin_edges  = lNj1j2_mass_smeared_scaled_bin_data[1]

        lWlNj1j2_mass_smeared_scaled_bin_counts = lWlNj1j2_mass_smeared_scaled_bin_data[0]
        lWlNj1j2_mass_smeared_scaled_bin_edges  = lWlNj1j2_mass_smeared_scaled_bin_data[1]

        bin_dataset_lNj1j2_mass_smeared          = BinDataSet(
                                                              bin_edges = lNj1j2_mass_smeared_bin_edges,
                                                              bin_counts = lNj1j2_mass_smeared_bin_counts
                                                             )
        bin_dataset_lWlNj1j2_mass_smeared        = BinDataSet(
                                                              bin_edges = lWlNj1j2_mass_smeared_bin_edges,
                                                              bin_counts = lWlNj1j2_mass_smeared_bin_counts
                                                             )
        bin_dataset_lNj1j2_mass_smeared_scaled   = BinDataSet(
                                                              bin_edges = lNj1j2_mass_smeared_scaled_bin_edges,
                                                              bin_counts = lNj1j2_mass_smeared_scaled_bin_counts
                                                             )
        bin_dataset_lWlNj1j2_mass_smeared_scaled = BinDataSet(
                                                              bin_edges = lWlNj1j2_mass_smeared_scaled_bin_edges,
                                                              bin_counts = lWlNj1j2_mass_smeared_scaled_bin_counts
                                                             )
        
#        print('bin edges bin edges bin edges bin edges bin edges bin edges bin edges bin edges bin edges bin edges bin edges bin edges bin edges bin edges bin edges bin edges bin edges')
#        bin_dataset_lWlNj1j2_mass_smeared_scaled.print_bin_edges()
#        print('bin edges bin edges bin edges bin edges bin edges bin edges bin edges bin edges bin edges bin edges bin edges bin edges bin edges bin edges bin edges bin edges bin edges')
#        print('bin counts bin counts bin counts bin counts bin counts bin counts bin counts bin counts bin counts bin counts bin counts bin counts bin counts bin counts bin counts')
#        bin_dataset_lWlNj1j2_mass_smeared_scaled.print_bin_counts()
#        print(f'len(lWlNj1j2_mass_smeared_scaled_bin_counts): {len(lWlNj1j2_mass_smeared_scaled_bin_counts)}')
#        print('bin counts bin counts bin counts bin counts bin counts bin counts bin counts bin counts bin counts bin counts bin counts bin counts bin counts bin counts bin counts')

        lNj1j2_mass_smeared_tail_inds          = bin_dataset_lNj1j2_mass_smeared.find_first_bin_below_ind(
                                                                                                          start_ind = 0,
                                                                                                          min_count = 0.01,
                                                                                                          start_at_max = True,
                                                                                                          min_count_type = 'frac'
                                                                                                         )
        lWlNj1j2_mass_smeared_tail_inds        = bin_dataset_lWlNj1j2_mass_smeared.find_first_bin_below_ind(
                                                                                                            start_ind = 0,
                                                                                                            min_count = 0.01,
                                                                                                            start_at_max = True,
                                                                                                            min_count_type = 'frac'
                                                                                                           )
        lNj1j2_mass_smeared_scaled_tail_inds   = bin_dataset_lNj1j2_mass_smeared_scaled.find_first_bin_below_ind(
                                                                                                                 start_ind = 0,
                                                                                                                 min_count = 0.01,
                                                                                                                 start_at_max = True,
                                                                                                                 min_count_type = 'frac'
                                                                                                                )
        lWlNj1j2_mass_smeared_scaled_tail_inds = bin_dataset_lWlNj1j2_mass_smeared_scaled.find_first_bin_below_ind(
                                                                                                                   start_ind = 0,
                                                                                                                   min_count = 0.01,
                                                                                                                   start_at_max = True,
                                                                                                                   min_count_type = 'frac'
                                                                                                                  )

        lNj1j2_mass_smeared_left_tail_hi_ind           =   lNj1j2_mass_smeared_tail_inds[0] + 1
        lNj1j2_mass_smeared_right_tail_lo_ind          =   lNj1j2_mass_smeared_tail_inds[1]

        lWlNj1j2_mass_smeared_left_tail_hi_ind         = lWlNj1j2_mass_smeared_tail_inds[0] + 1
        lWlNj1j2_mass_smeared_right_tail_lo_ind        = lWlNj1j2_mass_smeared_tail_inds[1]

        lNj1j2_mass_smeared_scaled_left_tail_hi_ind    =   lNj1j2_mass_smeared_scaled_tail_inds[0] + 1
        lNj1j2_mass_smeared_scaled_right_tail_lo_ind   =   lNj1j2_mass_smeared_scaled_tail_inds[1]

        lWlNj1j2_mass_smeared_scaled_left_tail_hi_ind  = lWlNj1j2_mass_smeared_scaled_tail_inds[0] + 1
        lWlNj1j2_mass_smeared_scaled_right_tail_lo_ind = lWlNj1j2_mass_smeared_scaled_tail_inds[1]


        lNj1j2_mass_smeared_left_tail_hi_edge           =   lNj1j2_mass_smeared_bin_edges[lNj1j2_mass_smeared_left_tail_hi_ind]
        lNj1j2_mass_smeared_right_tail_lo_edge          =   lNj1j2_mass_smeared_bin_edges[lNj1j2_mass_smeared_right_tail_lo_ind]

        lWlNj1j2_mass_smeared_left_tail_hi_edge         = lWlNj1j2_mass_smeared_bin_edges[lWlNj1j2_mass_smeared_left_tail_hi_ind]
        lWlNj1j2_mass_smeared_right_tail_lo_edge        = lWlNj1j2_mass_smeared_bin_edges[lWlNj1j2_mass_smeared_right_tail_lo_ind]

        lNj1j2_mass_smeared_scaled_left_tail_hi_edge    =   lNj1j2_mass_smeared_scaled_bin_edges[lNj1j2_mass_smeared_scaled_left_tail_hi_ind]
        lNj1j2_mass_smeared_scaled_right_tail_lo_edge   =   lNj1j2_mass_smeared_scaled_bin_edges[lNj1j2_mass_smeared_scaled_right_tail_lo_ind]

        lWlNj1j2_mass_smeared_scaled_left_tail_hi_edge  = lWlNj1j2_mass_smeared_scaled_bin_edges[lWlNj1j2_mass_smeared_scaled_left_tail_hi_ind]
        lWlNj1j2_mass_smeared_scaled_right_tail_lo_edge = lWlNj1j2_mass_smeared_scaled_bin_edges[lWlNj1j2_mass_smeared_scaled_right_tail_lo_ind]
        ###############################################################
        ###############################################################

        ###############################
        ### peak width calculations ###
        ###############################
        lNj1j2_mass_smeared_peak_width          =   lNj1j2_mass_smeared_right_tail_lo_edge        -   lNj1j2_mass_smeared_left_tail_hi_edge

        lWlNj1j2_mass_smeared_peak_width        = lWlNj1j2_mass_smeared_right_tail_lo_edge        - lWlNj1j2_mass_smeared_left_tail_hi_edge

        lNj1j2_mass_smeared_scaled_peak_width   =   lNj1j2_mass_smeared_scaled_right_tail_lo_edge -   lNj1j2_mass_smeared_scaled_left_tail_hi_edge

        lWlNj1j2_mass_smeared_scaled_peak_width = lWlNj1j2_mass_smeared_scaled_right_tail_lo_edge - lWlNj1j2_mass_smeared_scaled_left_tail_hi_edge


        lNj1j2_mass_smeared_peak_width_corr_to_uncorr = lNj1j2_mass_smeared_scaled_peak_width/lNj1j2_mass_smeared_peak_width
        lNj1j2_mass_smeared_peak_width_uncorr_to_corr = 1/lNj1j2_mass_smeared_peak_width_corr_to_uncorr

        lWlNj1j2_mass_smeared_peak_width_corr_to_uncorr = lWlNj1j2_mass_smeared_scaled_peak_width/lWlNj1j2_mass_smeared_peak_width
        lWlNj1j2_mass_smeared_peak_width_uncorr_to_corr = 1/lWlNj1j2_mass_smeared_peak_width_corr_to_uncorr
        ###############################
        ###############################

        ##############################
        ### tail and peak bin sums ###
        ##############################
        lNj1j2_mass_smeared_left_tail_bin_counts = lNj1j2_mass_smeared_bin_counts[0:lNj1j2_mass_smeared_left_tail_hi_ind]
        lNj1j2_mass_smeared_peak_bin_counts = lNj1j2_mass_smeared_bin_counts[lNj1j2_mass_smeared_left_tail_hi_ind:lNj1j2_mass_smeared_right_tail_lo_ind]
        lNj1j2_mass_smeared_right_tail_bin_counts = lNj1j2_mass_smeared_bin_counts[lNj1j2_mass_smeared_right_tail_lo_ind:]

        lWlNj1j2_mass_smeared_left_tail_bin_counts = lWlNj1j2_mass_smeared_bin_counts[0:lWlNj1j2_mass_smeared_left_tail_hi_ind]
        lWlNj1j2_mass_smeared_peak_bin_counts = lWlNj1j2_mass_smeared_bin_counts[lWlNj1j2_mass_smeared_left_tail_hi_ind:lWlNj1j2_mass_smeared_right_tail_lo_ind]
        lWlNj1j2_mass_smeared_right_tail_bin_counts = lWlNj1j2_mass_smeared_bin_counts[lWlNj1j2_mass_smeared_right_tail_lo_ind:]

        lNj1j2_mass_smeared_scaled_left_tail_bin_counts = lNj1j2_mass_smeared_scaled_bin_counts[0:lNj1j2_mass_smeared_scaled_left_tail_hi_ind]
        lNj1j2_mass_smeared_scaled_peak_bin_counts = lNj1j2_mass_smeared_scaled_bin_counts[lNj1j2_mass_smeared_scaled_left_tail_hi_ind:lNj1j2_mass_smeared_scaled_right_tail_lo_ind]
        lNj1j2_mass_smeared_scaled_right_tail_bin_counts = lNj1j2_mass_smeared_scaled_bin_counts[lNj1j2_mass_smeared_scaled_right_tail_lo_ind:]

        lWlNj1j2_mass_smeared_scaled_left_tail_bin_counts = lWlNj1j2_mass_smeared_scaled_bin_counts[0:lWlNj1j2_mass_smeared_scaled_left_tail_hi_ind]
        lWlNj1j2_mass_smeared_scaled_peak_bin_counts = lWlNj1j2_mass_smeared_scaled_bin_counts[lWlNj1j2_mass_smeared_scaled_left_tail_hi_ind:lWlNj1j2_mass_smeared_scaled_right_tail_lo_ind]
        lWlNj1j2_mass_smeared_scaled_right_tail_bin_counts = lWlNj1j2_mass_smeared_scaled_bin_counts[lWlNj1j2_mass_smeared_scaled_right_tail_lo_ind:]


        lNj1j2_mass_smeared_left_tail_bin_sum  = np.sum(lNj1j2_mass_smeared_left_tail_bin_counts)
        lNj1j2_mass_smeared_peak_bin_sum       = np.sum(lNj1j2_mass_smeared_peak_bin_counts)
        lNj1j2_mass_smeared_right_tail_bin_sum = np.sum(lNj1j2_mass_smeared_right_tail_bin_counts)

        lWlNj1j2_mass_smeared_left_tail_bin_sum  = np.sum(lWlNj1j2_mass_smeared_left_tail_bin_counts)
        lWlNj1j2_mass_smeared_peak_bin_sum       = np.sum(lWlNj1j2_mass_smeared_peak_bin_counts)
        lWlNj1j2_mass_smeared_right_tail_bin_sum = np.sum(lWlNj1j2_mass_smeared_right_tail_bin_counts)

        lNj1j2_mass_smeared_scaled_left_tail_bin_sum  = np.sum(lNj1j2_mass_smeared_scaled_left_tail_bin_counts)
        lNj1j2_mass_smeared_scaled_peak_bin_sum       = np.sum(lNj1j2_mass_smeared_scaled_peak_bin_counts)
        lNj1j2_mass_smeared_scaled_right_tail_bin_sum = np.sum(lNj1j2_mass_smeared_scaled_right_tail_bin_counts)

        lWlNj1j2_mass_smeared_scaled_left_tail_bin_sum  = np.sum(lWlNj1j2_mass_smeared_scaled_left_tail_bin_counts)
        lWlNj1j2_mass_smeared_scaled_peak_bin_sum       = np.sum(lWlNj1j2_mass_smeared_scaled_peak_bin_counts)
        lWlNj1j2_mass_smeared_scaled_right_tail_bin_sum = np.sum(lWlNj1j2_mass_smeared_scaled_right_tail_bin_counts)


        lNj1j2_mass_smeared_tails_bin_sum          =   lNj1j2_mass_smeared_left_tail_bin_sum        +   lNj1j2_mass_smeared_right_tail_bin_sum
        lWlNj1j2_mass_smeared_tails_bin_sum        = lWlNj1j2_mass_smeared_left_tail_bin_sum        + lWlNj1j2_mass_smeared_right_tail_bin_sum
        lNj1j2_mass_smeared_scaled_tails_bin_sum   =   lNj1j2_mass_smeared_scaled_left_tail_bin_sum +   lNj1j2_mass_smeared_scaled_right_tail_bin_sum
        lWlNj1j2_mass_smeared_scaled_tails_bin_sum = lWlNj1j2_mass_smeared_scaled_left_tail_bin_sum + lWlNj1j2_mass_smeared_scaled_right_tail_bin_sum


        lNj1j2_mass_smeared_bin_sum          =   lNj1j2_mass_smeared_tails_bin_sum        +   lNj1j2_mass_smeared_peak_bin_sum
        lWlNj1j2_mass_smeared_bin_sum        = lWlNj1j2_mass_smeared_tails_bin_sum        + lWlNj1j2_mass_smeared_peak_bin_sum
        lNj1j2_mass_smeared_scaled_bin_sum   =   lNj1j2_mass_smeared_scaled_tails_bin_sum +   lNj1j2_mass_smeared_scaled_peak_bin_sum
        lWlNj1j2_mass_smeared_scaled_bin_sum = lWlNj1j2_mass_smeared_scaled_tails_bin_sum + lWlNj1j2_mass_smeared_scaled_peak_bin_sum


        lNj1j2_mass_smeared_bin_sum_tails_to_tot = lNj1j2_mass_smeared_tails_bin_sum/(lNj1j2_mass_smeared_peak_bin_sum + lNj1j2_mass_smeared_tails_bin_sum)
        lNj1j2_mass_smeared_bin_sum_peak_to_tot  = lNj1j2_mass_smeared_peak_bin_sum /(lNj1j2_mass_smeared_peak_bin_sum + lNj1j2_mass_smeared_tails_bin_sum)

        lWlNj1j2_mass_smeared_bin_sum_tails_to_tot = lWlNj1j2_mass_smeared_tails_bin_sum/(lWlNj1j2_mass_smeared_peak_bin_sum + lWlNj1j2_mass_smeared_tails_bin_sum)
        lWlNj1j2_mass_smeared_bin_sum_peak_to_tot  = lWlNj1j2_mass_smeared_peak_bin_sum /(lWlNj1j2_mass_smeared_peak_bin_sum + lWlNj1j2_mass_smeared_tails_bin_sum)

        lNj1j2_mass_smeared_scaled_bin_sum_tails_to_tot = lNj1j2_mass_smeared_scaled_tails_bin_sum/(lNj1j2_mass_smeared_scaled_peak_bin_sum + lNj1j2_mass_smeared_scaled_tails_bin_sum)
        lNj1j2_mass_smeared_scaled_bin_sum_peak_to_tot  = lNj1j2_mass_smeared_scaled_peak_bin_sum /(lNj1j2_mass_smeared_scaled_peak_bin_sum + lNj1j2_mass_smeared_scaled_tails_bin_sum)

        lWlNj1j2_mass_smeared_scaled_bin_sum_tails_to_tot = lWlNj1j2_mass_smeared_scaled_tails_bin_sum/(lWlNj1j2_mass_smeared_scaled_peak_bin_sum + lWlNj1j2_mass_smeared_scaled_tails_bin_sum)
        lWlNj1j2_mass_smeared_scaled_bin_sum_peak_to_tot  = lWlNj1j2_mass_smeared_scaled_peak_bin_sum /(lWlNj1j2_mass_smeared_scaled_peak_bin_sum + lWlNj1j2_mass_smeared_scaled_tails_bin_sum)
        ##############################
        ##############################

        ##############################
        ### make the csv data dict ###
        ##############################
        csv_data_dict = {
                         'mWR': mWR,
                         'mN': mN,

                           'lNj1j2_mass_smeared_peak_width':          lNj1j2_mass_smeared_peak_width,
                         'lWlNj1j2_mass_smeared_peak_width':        lWlNj1j2_mass_smeared_peak_width,
                           'lNj1j2_mass_smeared_scaled_peak_width':   lNj1j2_mass_smeared_scaled_peak_width,
                         'lWlNj1j2_mass_smeared_scaled_peak_width': lWlNj1j2_mass_smeared_scaled_peak_width,

                           'lNj1j2_mass_smeared_peak_width_corr_to_uncorr':   lNj1j2_mass_smeared_peak_width_corr_to_uncorr,
                           'lNj1j2_mass_smeared_peak_width_uncorr_to_corr':   lNj1j2_mass_smeared_peak_width_uncorr_to_corr,
                         'lWlNj1j2_mass_smeared_peak_width_corr_to_uncorr': lWlNj1j2_mass_smeared_peak_width_corr_to_uncorr,
                         'lWlNj1j2_mass_smeared_peak_width_uncorr_to_corr': lWlNj1j2_mass_smeared_peak_width_uncorr_to_corr,

                         'lNj1j2_mass_smeared_bin_sum':            lNj1j2_mass_smeared_bin_sum,
                         'lWlNj1j2_mass_smeared_bin_sum':        lWlNj1j2_mass_smeared_bin_sum,
                         'lNj1j2_mass_smeared_scaled_bin_sum':     lNj1j2_mass_smeared_scaled_bin_sum,
                         'lWlNj1j2_mass_smeared_scaled_bin_sum': lWlNj1j2_mass_smeared_scaled_bin_sum,

                         'lNj1j2_mass_smeared_bin_sum_tails_to_tot':            lNj1j2_mass_smeared_bin_sum_tails_to_tot,
                         'lNj1j2_mass_smeared_bin_sum_peak_to_tot':            lNj1j2_mass_smeared_bin_sum_peak_to_tot,
                         'lWlNj1j2_mass_smeared_bin_sum_tails_to_tot':        lWlNj1j2_mass_smeared_bin_sum_tails_to_tot,
                         'lWlNj1j2_mass_smeared_bin_sum_peak_to_tot':        lWlNj1j2_mass_smeared_bin_sum_peak_to_tot,
                         'lNj1j2_mass_smeared_scaled_bin_sum_tails_to_tot':     lNj1j2_mass_smeared_scaled_bin_sum_tails_to_tot,
                         'lNj1j2_mass_smeared_scaled_bin_sum_peak_to_tot':     lNj1j2_mass_smeared_scaled_bin_sum_peak_to_tot,
                         'lWlNj1j2_mass_smeared_scaled_bin_sum_tails_to_tot': lWlNj1j2_mass_smeared_scaled_bin_sum_tails_to_tot,
                         'lWlNj1j2_mass_smeared_scaled_bin_sum_peak_to_tot': lWlNj1j2_mass_smeared_scaled_bin_sum_peak_to_tot,
                        }

        fieldnames = list(csv_data_dict.keys())
        ##############################
        ##############################

        if plot_tail_bounds == True:
            ################################
            ### plot tail boundary lines ###
            ################################
            dataset_lNj1j2_mass_smeared.plot_vert_line(
                                                       fig_dict = fig_agg.get_fig_dict(),
                                                       figname = 'mWR_' + str(mWR) + '_' + 'lNj1j2_mass_smeared', #<-- VARIABLE NAME
                                                       axname = 'ax_a',
                                                       savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lNj1j2_mass_smeared.png', #<-- VARIABLE NAME
                                                       x = [lNj1j2_mass_smeared_left_tail_hi_edge, lNj1j2_mass_smeared_right_tail_lo_edge], #<-- VARIABLE NAME
                                                       ymin = lo_3obj_mass_y_bound,
                                                       ymax = hi_3obj_mass_y_bound,
                                                       linestyles = 'solid',
                                                       label = '',
                                                      )
            dataset_lWlNj1j2_mass_smeared.plot_vert_line(
                                                          fig_dict = fig_agg.get_fig_dict(),
                                                          figname = 'mWR_' + str(mWR) + '_' + 'lWlNj1j2_mass_smeared', #<-- VARIABLE NAME
                                                          axname = 'ax_a',
                                                          savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lWlNj1j2_mass_smeared.png', #<-- VARIABLE NAME
                                                          x = [lWlNj1j2_mass_smeared_left_tail_hi_edge, lWlNj1j2_mass_smeared_right_tail_lo_edge], #<-- VARIABLE NAME
                                                          ymin = lo_4obj_mass_y_bound,
                                                          ymax = hi_4obj_mass_y_bound,
                                                          linestyles = 'solid',
                                                          label = '',
                                                         )
            dataset_lNj1j2_mass_smeared_scaled.plot_vert_line(
                                                              fig_dict = fig_agg.get_fig_dict(),
                                                              figname = 'mWR_' + str(mWR) + '_' + 'lNj1j2_mass_smeared_scaled', #<-- VARIABLE NAME
                                                              axname = 'ax_a',
                                                              savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lNj1j2_mass_smeared_scaled.png', #<-- VARIABLE NAME
                                                              x = [lNj1j2_mass_smeared_scaled_left_tail_hi_edge, lNj1j2_mass_smeared_scaled_right_tail_lo_edge], #<-- VARIABLE NAME
                                                              ymin = lo_3obj_mass_y_bound,
                                                              ymax = hi_3obj_mass_y_bound,
                                                              linestyles = 'solid',
                                                              label = '',
                                                             )
            dataset_lWlNj1j2_mass_smeared_scaled.plot_vert_line(
                                                              fig_dict = fig_agg.get_fig_dict(),
                                                              figname = 'mWR_' + str(mWR) + '_' + 'lWlNj1j2_mass_smeared_scaled', #<-- VARIABLE NAME
                                                              axname = 'ax_a',
                                                              savepath = 'hists/mWR_' + str(mWR) + '/mWR_' + str(mWR) + '_lWlNj1j2_mass_smeared_scaled.png', #<-- VARIABLE NAME
                                                              x = [lWlNj1j2_mass_smeared_scaled_left_tail_hi_edge, lWlNj1j2_mass_smeared_scaled_right_tail_lo_edge], #<-- VARIABLE NAME
                                                              ymin = lo_4obj_mass_y_bound,
                                                              ymax = hi_4obj_mass_y_bound,
                                                              linestyles = 'solid',
                                                              label = '',
                                                             )
            ################################
            ################################

    return fieldnames, csv_data_dict

start = time.time()

csv_data = []

############################################
### run the analysis for all mass points ###
############################################
for data_file_name in sorted_npz_data_files: #'''each data file is a single mWR, mN point'''
    curr_csv_data = analyze_data(
                                 data_dir,
                                 data_file_name,
                                 fig_agg,
                                 make_pt_figs = False,
                                 make_mass_figs = True,
                                 make_eta_figs = False,
                                 make_phi_figs = False,
                                 make_smear_frac_figs = False,
                                 make_lW_pt_aligned_figs = False,
                                 make_jet_p_correction_fac_figs = False,
                                 make_scaled_jet_var_figs = True,
                                 analyze_tails = True,
                                 plot_tail_bounds = True
                                )

    fieldnames = curr_csv_data[0]
    curr_csv_data_dict = curr_csv_data[1]
    csv_data.append(curr_csv_data_dict)
############################################
############################################

##############################
### make the csv data file ###
##############################
with open('analyzer_output.csv', 'w', newline = '') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
    writer.writeheader()
    writer.writerows(csv_data)
##############################
##############################

########################
### save the figures ###
########################
fig_agg.save_all_figs()
#fig_agg.print_fig_dict()
########################
########################

end = time.time()
tot_secs = end - start

if tot_secs < 60:
    print(f'Ran analysis in {np.round(tot_secs, 2)} s')
else:
    tot_mins = math.floor(tot_secs/60)
    remain_secs = tot_secs - 60*tot_mins

    print(f'Ran analysis in {tot_mins} m {np.round(remain_secs, 2)} s')

