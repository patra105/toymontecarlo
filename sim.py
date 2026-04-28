import numpy as np
from scipy.stats import truncnorm
import vector
import matplotlib.pyplot as plt
import mplhep as hep
import os, math, time

# input is a np.ndarray of shape
# [[E_0 px_0 py_0 pz_0]
#  [E_1 px_1 py_1 pz_1]
#  [E_2 px_2 py_2 pz_2]
#  [E_3 px_3 py_3 pz_3]
#           .
#           .
#           .
#  [E_n px_n py_n pz_n]]
def convert_to_4D_p_array(four_vec_array):

    dtype = [('E', float), ('px', float), ('py', float), ('pz', float)]

    struc_four_vec_array = four_vec_array.astype(float).view(dtype).reshape(-1)
    p_array_4D = struc_four_vec_array.view(vector.MomentumNumpy4D)

    return p_array_4D


# mWR: WR rest mass (all WRs have the same rest mass)
# num_events: number of events
def make_rand_WR_data_lab(mWR, WR_std_dev, num_events):

    if WR_std_dev != 0:
        mean = 1.0
        lo_bound = (0.0001 - mean)/WR_std_dev
        hi_bound = np.inf
        mass_factors_list = truncnorm.rvs(lo_bound, hi_bound, loc = mean, scale = WR_std_dev, size = num_events)
    else:
        mass_factors_list = np.ones(num_events)

    mWR_array = mass_factors_list*mWR

    sign_array = 2*np.random.randint(2, size = num_events) - 1 # random sign for each z momentum

    pz_array = sign_array*0.05*mWR*np.random.rand(num_events) # random z momenta in [0,0.05*m)
    px_array = np.zeros(num_events) # WR only has z momentum
    py_array = np.zeros(num_events)

    E_array = np.sqrt(np.power(pz_array, 2) + np.power(mWR_array, 2))

    E_array  =  E_array.reshape(-1, 1) # wrap each element of each of these 1d arrays in its own list
    px_array = px_array.reshape(-1, 1)
    py_array = py_array.reshape(-1, 1)
    pz_array = pz_array.reshape(-1, 1)

    # naming convention for 4 vector arrays --> <particle>_4D_p_array_<reference frame>
    WR_four_vec_array_lab = np.concatenate((E_array, px_array, py_array, pz_array), axis = 1)

#    print(f'\neach row is data for one WR:\n{WR_four_vec_array_lab}')
#    print(f'\nfour_vec_array[0]:\n{WR_four_vec_array_lab[0]}')
#    print(f'\nfour_vec_array[:,0]:\n{WR_four_vec_array_lab[:,0]}')

    WR_p_array_4D_lab = convert_to_4D_p_array(WR_four_vec_array_lab)

    return WR_p_array_4D_lab


def make_rand_angles(num_events): # these are the angles of decay for ONE of the two daugter particles as viewed from the parent particle rest frame

    cos_theta_array = 2*np.random.rand(num_events) - 1 # random cos(theta) values in [-1,1)
    theta_array = np.arccos(cos_theta_array)

    phi_array = 2*np.pi*np.random.rand(num_events) - np.pi # random phi values in [-pi,pi)

    return theta_array, phi_array


def boost_to_lab(parent_4D_p_array_lab, daughter_4D_p_array_parent):

    parent_beta_array = parent_4D_p_array_lab.beta

    gamma_array = np.divide(1, np.sqrt(1 - np.power(parent_beta_array, 2)))

    parent_mass_array = parent_4D_p_array_lab.mass

    denominator_array = np.multiply(gamma_array, parent_mass_array)

    parent_px_array = parent_4D_p_array_lab.px
    parent_py_array = parent_4D_p_array_lab.py
    parent_pz_array = parent_4D_p_array_lab.pz

    parent_betax_array = np.divide(parent_px_array, denominator_array) # parent beta values as viewed from the lab frame
    parent_betay_array = np.divide(parent_py_array, denominator_array)
    parent_betaz_array = np.divide(parent_pz_array, denominator_array)

    lab_betax_array = -parent_betax_array # lab beta values as viewed from rest frame of parent
    lab_betay_array = -parent_betay_array
    lab_betaz_array = -parent_betaz_array

    daughter_E_array_parent  = daughter_4D_p_array_parent.E
    daughter_px_array_parent = daughter_4D_p_array_parent.px
    daughter_py_array_parent = daughter_4D_p_array_parent.py
    daughter_pz_array_parent = daughter_4D_p_array_parent.pz

    pxbetax_array = np.multiply(lab_betax_array, daughter_px_array_parent)
    pybetay_array = np.multiply(lab_betay_array, daughter_py_array_parent)
    pzbetaz_array = np.multiply(lab_betaz_array, daughter_pz_array_parent)

    pibetai_sum_array = pxbetax_array + pybetay_array + pzbetaz_array
    factor_array = np.divide(gamma_array - 1, np.power(parent_beta_array, 2))
    daughter_gammaE_array_parent = np.multiply(gamma_array, daughter_E_array_parent)

    daughter_E_array_lab  = np.multiply(gamma_array, daughter_E_array_parent - pibetai_sum_array)
    daughter_px_array_lab = daughter_px_array_parent + np.multiply(factor_array, np.multiply(lab_betax_array, pibetai_sum_array)) - np.multiply(daughter_gammaE_array_parent, lab_betax_array)
    daughter_py_array_lab = daughter_py_array_parent + np.multiply(factor_array, np.multiply(lab_betay_array, pibetai_sum_array)) - np.multiply(daughter_gammaE_array_parent, lab_betay_array)
    daughter_pz_array_lab = daughter_pz_array_parent + np.multiply(factor_array, np.multiply(lab_betaz_array, pibetai_sum_array)) - np.multiply(daughter_gammaE_array_parent, lab_betaz_array)

#    print(f'daughter_E_array_lab:\n{daughter_E_array_lab}')
#    print(f'daughter_px_array_lab:\n{daughter_px_array_lab}')
#    print(f'daughter_py_array_lab:\n{daughter_py_array_lab}')
#    print(f'daughter_pz_array_lab:\n{daughter_pz_array_lab}')

    daughter_E_array_lab  =  daughter_E_array_lab.reshape(-1, 1) # wrap each element of each of these 1d arrays in its own list
    daughter_px_array_lab = daughter_px_array_lab.reshape(-1, 1)
    daughter_py_array_lab = daughter_py_array_lab.reshape(-1, 1)
    daughter_pz_array_lab = daughter_pz_array_lab.reshape(-1, 1)

    daughter_four_vec_array_lab = np.concatenate((daughter_E_array_lab, daughter_px_array_lab, daughter_py_array_lab, daughter_pz_array_lab), axis = 1)

#    print(f'\neach row is data for one daughter:\n{daughter_four_vec_array_lab}')
#    print(f'\nfour_vec_array[0]:\n{daughter_four_vec_array_lab[0]}')
#    print(f'\nfour_vec_array[:,0]:\n{daughter_four_vec_array_lab[:,0]}')

    daughter_4D_p_array_lab = convert_to_4D_p_array(daughter_four_vec_array_lab)

    return daughter_4D_p_array_lab


def rand_decay(parent_4D_p_array_lab, daughter_m1, daughter_m2, daughter_1_mass_fac_list, daughter_2_mass_fac_list):

    E0_array = parent_4D_p_array_lab.mass # rest mass of parent particle
#    print(E0_array)

    daughter_m1_array = daughter_1_mass_fac_list*daughter_m1
    daughter_m2_array = daughter_2_mass_fac_list*daughter_m2

    ##################################
    ### ensure energy conservation ###
    ##################################
    mask = (daughter_m1_array + daughter_m2_array) <= E0_array
    daughter_m1_array = daughter_m1_array[mask]
    daughter_m2_array = daughter_m2_array[mask]
    E0_array = E0_array[mask]
    parent_4D_p_array_lab = parent_4D_p_array_lab[mask]
    daughter_1_mass_fac_list = daughter_1_mass_fac_list[mask]
    daughter_2_mass_fac_list = daughter_2_mass_fac_list[mask]
    ##################################
    ##################################

    num_events = len(E0_array)
 
    E0_2_array = np.power(E0_array, 2)
    E0_4_array = np.power(E0_array, 4)

    daughter_m1_2_array = np.power(daughter_m1_array, 2)
    daughter_m1_4_array = np.power(daughter_m1_array, 4)
    daughter_m2_2_array = np.power(daughter_m2_array, 2)
    daughter_m2_4_array = np.power(daughter_m2_array, 4)

    fourth_power_sum = E0_4_array + daughter_m1_4_array + daughter_m2_4_array
    squared_products_sum = np.multiply(E0_2_array, daughter_m1_2_array + daughter_m2_2_array) + np.multiply(daughter_m1_2_array, daughter_m2_2_array)

    p_array = np.divide(np.sqrt(0.25*fourth_power_sum - 0.5*squared_products_sum), E0_array)

    theta_array, phi_array = make_rand_angles(num_events)

    x_factor_array = np.multiply(np.sin(theta_array), np.cos(phi_array))
    y_factor_array = np.multiply(np.sin(theta_array), np.sin(phi_array))
    z_factor_array = np.cos(theta_array)

    p1_x_array = np.multiply(p_array, x_factor_array)
    p1_y_array = np.multiply(p_array, y_factor_array)
    p1_z_array = np.multiply(p_array, z_factor_array)

    p2_x_array = -1*p1_x_array
    p2_y_array = -1*p1_y_array
    p2_z_array = -1*p1_z_array

    E1_array = np.sqrt(np.power(p_array, 2) + np.power(daughter_m1_array, 2))
    E2_array = np.sqrt(np.power(p_array, 2) + np.power(daughter_m2_array, 2))

    E1_array   =   E1_array.reshape(-1, 1)
    p1_x_array = p1_x_array.reshape(-1, 1)
    p1_y_array = p1_y_array.reshape(-1, 1)
    p1_z_array = p1_z_array.reshape(-1, 1)

    E2_array   =   E2_array.reshape(-1, 1)
    p2_x_array = p2_x_array.reshape(-1, 1)
    p2_y_array = p2_y_array.reshape(-1, 1)
    p2_z_array = p2_z_array.reshape(-1, 1)

    daughter1_four_vec_array_parent = np.concatenate((E1_array, p1_x_array, p1_y_array, p1_z_array), axis = 1)
    daughter2_four_vec_array_parent = np.concatenate((E2_array, p2_x_array, p2_y_array, p2_z_array), axis = 1)

    daughter1_4D_p_array_parent = convert_to_4D_p_array(daughter1_four_vec_array_parent)
    daughter2_4D_p_array_parent = convert_to_4D_p_array(daughter2_four_vec_array_parent)

    ####################################################
    ### boosting from parent rest frame to lab frame ###
    ####################################################
    daughter1_4D_p_array_lab = daughter1_4D_p_array_parent.boost_p4(parent_4D_p_array_lab)
    daughter2_4D_p_array_lab = daughter2_4D_p_array_parent.boost_p4(parent_4D_p_array_lab)

#    daughter1_4D_p_array_lab = boost_to_lab(parent_4D_p_array_lab, daughter1_4D_p_array_parent)
#    daughter2_4D_p_array_lab = boost_to_lab(parent_4D_p_array_lab, daughter2_4D_p_array_parent)
    ####################################################
    ####################################################

    return daughter1_4D_p_array_lab, daughter2_4D_p_array_lab, daughter_1_mass_fac_list, daughter_2_mass_fac_list


def smear_energy(obj_4D_p_array_lab, obj_type):

    obj_pt_array = obj_4D_p_array_lab.pt

    if obj_type == 'jet':
        S = 0.92
        C = 0.04

        std_dev_array = np.sqrt(np.divide(S**2, obj_pt_array) + C**2)

        mean = 1.0

        lower_bound = np.divide((0 - mean), std_dev_array)
        upper_bound = np.inf

        factors_array = truncnorm.rvs(lower_bound, upper_bound, loc = mean, scale = std_dev_array)

    if obj_type == 'lepton':
        factors_array = np.ones(len(obj_4D_p_array_lab))

    obj_E_array  = obj_4D_p_array_lab.E
    obj_px_array = obj_4D_p_array_lab.px
    obj_py_array = obj_4D_p_array_lab.py
    obj_pz_array = obj_4D_p_array_lab.pz

    obj_m_array  = obj_4D_p_array_lab.m

    obj_p_array  = obj_4D_p_array_lab.p

    obj_frac_px_array = np.divide(obj_px_array, obj_p_array)
    obj_frac_py_array = np.divide(obj_py_array, obj_p_array)
    obj_frac_pz_array = np.divide(obj_pz_array, obj_p_array)

    obj_E_array = np.multiply(obj_E_array, factors_array)
    obj_p_array = np.sqrt(np.power(obj_E_array, 2) - np.power(obj_m_array, 2))

    obj_px_array = np.multiply(obj_frac_px_array, obj_p_array)
    obj_py_array = np.multiply(obj_frac_py_array, obj_p_array)
    obj_pz_array = np.multiply(obj_frac_pz_array, obj_p_array)

    obj_E_array  =  obj_E_array.reshape(-1, 1) # wrap each element of each of these 1d arrays in its own list
    obj_px_array = obj_px_array.reshape(-1, 1)
    obj_py_array = obj_py_array.reshape(-1, 1)
    obj_pz_array = obj_pz_array.reshape(-1, 1)

    # naming convention for 4 vector arrays --> <particle>_4D_p_array_<reference frame>
    obj_four_vec_array_lab = np.concatenate((obj_E_array, obj_px_array, obj_py_array, obj_pz_array), axis = 1)

    obj_4D_p_array_lab_smeared = convert_to_4D_p_array(obj_four_vec_array_lab)

    return obj_4D_p_array_lab_smeared


# m: WR rest mass (all WRs have the same rest mass)
# num_events: number of events to generate
def run_sim(mWR, WR_std_dev, mN, N_std_dev, mlW, mWR_off, mlN, mj1, mj2, num_events):

    output_dir = 'simulation_npz_data/'
    os.makedirs(output_dir, exist_ok = True)

    ###################
    ### decay chain ###
    ###################
    # naming convention for 4 vector arrays --> <particle>_4D_p_array_<reference frame>
    WR_4D_p_array_lab = make_rand_WR_data_lab(mWR, WR_std_dev, num_events)
#    print(WR_4D_p_array_lab)
#    print(f'\nWR invariant masses:\n{WR_4D_p_array.mass}')

    ########################################
    ### make mass smearing factors lists ###
    ########################################
    num_events = len(WR_4D_p_array_lab)

    if N_std_dev != 0:
        mean = 1.0
        lo_bound = (0.0001 - mean)/N_std_dev
        hi_bound = np.inf
        N_mass_factors_list = truncnorm.rvs(lo_bound, hi_bound, loc = mean, scale = N_std_dev, size = num_events)
    else:
        N_mass_factors_list = np.ones(num_events)

    ones_list = np.ones(num_events)
    ########################################
    ########################################

    N_4D_p_array_lab, lW_4D_p_array_lab, N_mass_factors_list, ones_list      = rand_decay(
                                                                                          parent_4D_p_array_lab = WR_4D_p_array_lab,
                                                                                          daughter_m1 = mN,
                                                                                          daughter_m2 = mlW,
                                                                                          daughter_1_mass_fac_list = N_mass_factors_list,
                                                                                          daughter_2_mass_fac_list = ones_list,
                                                                                         )

    WR_off_4D_p_array_lab, lN_4D_p_array_lab, N_mass_factors_list, ones_list = rand_decay(
                                                                                          parent_4D_p_array_lab = N_4D_p_array_lab,
                                                                                          daughter_m1 = mWR_off,
                                                                                          daughter_m2 = mlN,
                                                                                          daughter_1_mass_fac_list = N_mass_factors_list,
                                                                                          daughter_2_mass_fac_list = ones_list,
                                                                                         )

    j1_4D_p_array_lab, j2_4D_p_array_lab, ones_list, ones_list               = rand_decay(
                                                                                          parent_4D_p_array_lab = WR_off_4D_p_array_lab,
                                                                                          daughter_m1 = mj1,
                                                                                          daughter_m2 = mj2,
                                                                                          daughter_1_mass_fac_list = ones_list,
                                                                                          daughter_2_mass_fac_list = ones_list,
                                                                                         )
    ###################
    ###################

    num_events = len(j1_4D_p_array_lab)

    if len(lW_4D_p_array_lab) == len(lN_4D_p_array_lab) == len(j1_4D_p_array_lab) == len(j2_4D_p_array_lab):
        pass
    else:
        print('-!-!-!-!-!-!-!-!-!-!-!-> THE ARRAYS DO NOT HAVE THE SAME LENGTHS <-!-!-!-!-!-!-!-!-!-!-!-')

    #####################################
    ### smearing the leptons and jets ###
    #####################################
    lW_4D_p_array_lab_smeared = smear_energy(lW_4D_p_array_lab, 'lepton')
    lN_4D_p_array_lab_smeared = smear_energy(lN_4D_p_array_lab, 'lepton')
    j1_4D_p_array_lab_smeared = smear_energy(j1_4D_p_array_lab, 'jet')
    j2_4D_p_array_lab_smeared = smear_energy(j2_4D_p_array_lab, 'jet')
    #####################################
    #####################################

    ###################################
    ### prepare data for npz format ###
    ###################################
    lW_data         = np.stack([lW_4D_p_array_lab.E,         lW_4D_p_array_lab.px,         lW_4D_p_array_lab.py,         lW_4D_p_array_lab.pz],         axis=1)
    lN_data         = np.stack([lN_4D_p_array_lab.E,         lN_4D_p_array_lab.px,         lN_4D_p_array_lab.py,         lN_4D_p_array_lab.pz],         axis=1)
    j1_data         = np.stack([j1_4D_p_array_lab.E,         j1_4D_p_array_lab.px,         j1_4D_p_array_lab.py,         j1_4D_p_array_lab.pz],         axis=1)
    j2_data         = np.stack([j2_4D_p_array_lab.E,         j2_4D_p_array_lab.px,         j2_4D_p_array_lab.py,         j2_4D_p_array_lab.pz],         axis=1)

    lW_data_smeared = np.stack([lW_4D_p_array_lab_smeared.E, lW_4D_p_array_lab_smeared.px, lW_4D_p_array_lab_smeared.py, lW_4D_p_array_lab_smeared.pz], axis=1)
    lN_data_smeared = np.stack([lN_4D_p_array_lab_smeared.E, lN_4D_p_array_lab_smeared.px, lN_4D_p_array_lab_smeared.py, lN_4D_p_array_lab_smeared.pz], axis=1)
    j1_data_smeared = np.stack([j1_4D_p_array_lab_smeared.E, j1_4D_p_array_lab_smeared.px, j1_4D_p_array_lab_smeared.py, j1_4D_p_array_lab_smeared.pz], axis=1)
    j2_data_smeared = np.stack([j2_4D_p_array_lab_smeared.E, j2_4D_p_array_lab_smeared.px, j2_4D_p_array_lab_smeared.py, j2_4D_p_array_lab_smeared.pz], axis=1)
    ###################################
    ###################################

#    print(f'\nlW_data:\n{lW_data}')

    np.savez(
             output_dir + f'mWR_{mWR}_mN_{mN}_{num_events}events_sim_data.npz',
             lW_data         = lW_data,
             lN_data         = lN_data,
             j1_data         = j1_data,
             j2_data         = j2_data,
             lW_data_smeared = lW_data_smeared,
             lN_data_smeared = lN_data_smeared,
             j1_data_smeared = j1_data_smeared,
             j2_data_smeared = j2_data_smeared,
            )

    return 0


def main():

    start = time.time()

    # valid mWR values for WRCoffea --> [600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000, 3200, 3400] (GeV)
    # valid mN values for WRCoffea --> [100, 200, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800,
    #                                   1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300]
    #
    # mass_points_list structure:
    # [
    #  [mWR_0, [mN_0_0, mN_0_1, ..., mN_0_q], [WR_std_dev_0_0, WR_std_dev_0_1, ..., WR_std_dev_0_q], [N_std_dev_0_0, N_std_dev_0_1, ..., N_std_dev_0_q]],
    #  [mWR_1, [mN_1_0, mN_1_1, ..., mN_1_q], [WR_std_dev_1_0, WR_std_dev_1_1, ..., WR_std_dev_1_q], [N_std_dev_1_0, N_std_dev_1_1, ..., N_std_dev_1_q]],
    #  [mWR_2, [mN_2_0, mN_2_1, ..., mN_2_q], [WR_std_dev_2_0, WR_std_dev_2_1, ..., WR_std_dev_2_q], [N_std_dev_2_0, N_std_dev_2_1, ..., N_std_dev_2_q]],
    #                                                                           .
    #                                                                           .
    #                                                                           .
    #  [mWR_k, [mN_k_0, mN_k_1, ..., mN_k_q], [WR_std_dev_k_0, WR_std_dev_k_1, ..., WR_std_dev_k_q], [N_std_dev_k_0, N_std_dev_k_1, ..., N_std_dev_k_q]],
    # ]

    mass_points_list = [
                        [1200, [1, 200, 1100], [0, 0, 0], [0, 0, 0]],
                        [2000, [1, 800, 1900], [0, 0, 0], [0, 0, 0]], # no Gaussian spread on either the mWR or the mN
                        [3200, [1, 800, 3000], [0, 0, 0], [0, 0, 0]],
                       ]

#    mass_points_list = [
#                        [1200, [1, 200, 1100], [0.0478, 0.0478, 0.0902], [0, 0, 0]],
#                        [2000, [1, 800, 1900], [0.0573, 0.0573, 0.0743], [0, 0, 0]], # Gaussian spread only on the mWR
#                        [3200, [1, 800, 3000], [0.0438, 0.0438, 0.0610], [0, 0, 0]],
#                       ]

#    mass_points_list = [
#                        [1200, [1, 200, 1100], [0, 0, 0], [0.0478, 0.0478, 0.0902]],
#                        [2000, [1, 800, 1900], [0, 0, 0], [0.0573, 0.0573, 0.0743]], # Gaussian spread only on the mN
#                        [3200, [1, 800, 3000], [0, 0, 0], [0.0438, 0.0438, 0.0610]],
#                       ]

    # mass_points_list = [
    #                     [1200, [1, 200, 1100], [0.0478, 0.0478, 0.0902], [0.0478, 0.0478, 0.0902]],
    #                     [2000, [1, 800, 1900], [0.0573, 0.0573, 0.0743], [0.0573, 0.0573, 0.0743]], # Gaussian spread on both the mWR and the mN
    #                     [3200, [1, 800, 3000], [0.0438, 0.0438, 0.0610], [0.0438, 0.0438, 0.0610]],
    #                     ]

    num_events_list = [10, 10**3, 10**4, 10**5, 10**6, 10**7]

    num_sims = 0
    for mWR_mN_list in mass_points_list[0:]:

        mWR             = mWR_mN_list[0]
        mN_list         = mWR_mN_list[1]
        WR_std_dev_list = mWR_mN_list[2]
        N_std_dev_list  = mWR_mN_list[3]

        for i in range(len(mN_list[0:])):

            mN = mN_list[i]
            WR_std_dev = WR_std_dev_list[i]
            N_std_dev  =  N_std_dev_list[i]
            mlW = 0
            mWR_off = (2/3)*mN
            mlN = 0
            mj1 = 0
            mj2 = 0

            for num_events in num_events_list[-2:-1]:

                run_sim(
                        mWR = mWR,
                        WR_std_dev = WR_std_dev,
                        mN = mN,
                        N_std_dev = N_std_dev,
                        mlW = mlW,
                        mWR_off = mWR_off,
                        mlN = mlN,
                        mj1 = mj1,
                        mj2 = mj2,
                        num_events = num_events,
                       )

                num_sims += 1

    end = time.time()

    seconds = end - start

    if seconds < 60:
        print(f'\nall {num_sims} simulations ran in {np.round(seconds, decimals = 3)} s')
    else:
        print(f'\nall {num_sims} simulations ran in {math.floor(seconds/60)} min {np.round(seconds%60, decimals = 2)} s')

if __name__ == '__main__':
    main()

