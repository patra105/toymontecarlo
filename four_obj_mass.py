"""
4-object invariant mass comparison plots

For each (mWR, mN) point this script creates:

1) Unsmeared 4-object invariant mass
2) Smeared 4-object invariant mass
3) Smeared + corrected 4-object invariant mass
4) s1 distribution
5) s2 distribution

All plots are saved as PDF files.

IMPORTANT:
All divide-by-zero protections have been removed intentionally.
NumPy floating-point exceptions are enabled to expose failures.
"""

import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import os
import re
from collections import defaultdict


# ============================================================
# RAISE FLOATING POINT EXCEPTIONS
# ============================================================

np.seterr(all='raise')


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def dotprod_T(p1, p2):

    return p1[:, 1] * p2[:, 1] + p1[:, 2] * p2[:, 2]


def invariant_mass(particles):

    total_E = sum(p[:, 0] for p in particles)

    total_px = sum(p[:, 1] for p in particles)

    total_py = sum(p[:, 2] for p in particles)

    total_pz = sum(p[:, 3] for p in particles)

    p_squared = total_px**2 + total_py**2 + total_pz**2

    return np.sqrt(total_E**2 - p_squared)


def compute_s1_s2(lW, lN, j1, j2):

    # ========================================================
    # TRANSVERSE MOMENTA
    # ========================================================

    pT1 = np.sqrt(dotprod_T(j1, j1))

    pT2 = np.sqrt(dotprod_T(j2, j2))

    pTlW = np.sqrt(dotprod_T(lW, lW))

    # ========================================================
    # JET RESOLUTION
    # ========================================================

    S = 0.92

    C = 0.04

    sig1 = pT1 * np.sqrt(S**2 / pT1 + C**2)

    sig2 = pT2 * np.sqrt(S**2 / pT2 + C**2)

    # ========================================================
    # MATRIX COEFFICIENTS
    # ========================================================

    a11 = dotprod_T(j1, lW)

    a12 = dotprod_T(j2, lW)

    a21 = pT1**2 / sig1**2

    a22 = -(pT2**2 / sig2**2) * (dotprod_T(j1, lW) / dotprod_T(j2, lW))

    d1 = -pTlW**2 - dotprod_T(lW, lN)

    d2 = pT1**2 / sig1**2 - (pT2**2 / sig2**2) * (dotprod_T(j1, lW) / dotprod_T(j2, lW))

    # ========================================================
    # SOLVE FOR s1, s2
    # ========================================================

    denom = a11 * a22 - a12 * a21

    s1 = (a22 * d1 - a12 * d2) / denom

    s2 = (a11 * d2 - a21 * d1) / denom

    return s1, s2


def corrected_4obj_mass(lW, lN, j1, j2):

    s1, s2 = compute_s1_s2(lW, lN, j1, j2)

    # ========================================================
    # RESCALE JET FOUR-VECTORS
    # ========================================================

    j1_scaled = j1 * s1[:, None]

    j2_scaled = j2 * s2[:, None]

    corrected_mass = invariant_mass([lW, lN, j1_scaled, j2_scaled])

    return corrected_mass, s1, s2


# ============================================================
# CONFIGURATION
# ============================================================

data_dir = 'simulation_npz_data/'

mass_output_dir = 'mass_histograms_corrected_pdf/'

s_output_dir = 's1_s2_histograms_pdf/'

os.makedirs(mass_output_dir, exist_ok=True)

os.makedirs(s_output_dir, exist_ok=True)

x_bounds = [0, 5000]

y_bounds = [1, 1e8]

bin_width = 50


# ============================================================
# FIND NPZ FILES
# ============================================================

files_by_mWR = defaultdict(list)

npz_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npz')])

for fname in npz_files:

    mWR_match = re.search(r'mWR_(\d+)', fname)

    mN_match = re.search(r'mN_(\d+)', fname)

    events_match = re.search(r'(\d+)events_sim_data', fname)

    if mWR_match and mN_match and events_match:

        num_events = int(events_match.group(1))

        if num_events == 1000000:

            mWR = int(mWR_match.group(1))

            mN = int(mN_match.group(1))

            files_by_mWR[mWR].append((mN, fname))


for mWR in files_by_mWR:

    files_by_mWR[mWR].sort(key=lambda x: x[0])


# ============================================================
# STYLE
# ============================================================

style = hep.style.CMS

style['font.size'] = 13

plt.style.use(style)


# ============================================================
# MAIN LOOP
# ============================================================

for mWR in sorted(files_by_mWR.keys()):

    for mN, fname in files_by_mWR[mWR]:

        print(f'\nProcessing mWR={mWR}, mN={mN}')

        try:

            # =================================================
            # LOAD DATA
            # =================================================

            data_path = os.path.join(data_dir, fname)

            data = np.load(data_path)

            # =================================================
            # UNSMEARED
            # =================================================

            lW_u = data['lW_data']

            lN_u = data['lN_data']

            j1_u = data['j1_data']

            j2_u = data['j2_data']

            mass_unsmeared = invariant_mass([lW_u, lN_u, j1_u, j2_u])

            # =================================================
            # SMEARED
            # =================================================

            lW_s = data['lW_data_smeared']

            lN_s = data['lN_data_smeared']

            j1_s = data['j1_data_smeared']

            j2_s = data['j2_data_smeared']

            mass_smeared = invariant_mass([lW_s, lN_s, j1_s, j2_s])

            # =================================================
            # SMEARED + CORRECTED
            # =================================================

            mass_corrected, s1, s2 = corrected_4obj_mass(
                lW_s,
                lN_s,
                j1_s,
                j2_s
            )

            # =================================================
            # MASS HISTOGRAM
            # =================================================

            fig, ax = plt.subplots(figsize=(10, 8))

            bins = np.arange(
                x_bounds[0],
                x_bounds[1] + bin_width,
                bin_width
            )

            ax.hist(
                mass_unsmeared,
                bins=bins,
                histtype='step',
                linewidth=2.8,
                color='black',
                label='Unsmeared'
            )

            ax.hist(
                mass_smeared,
                bins=bins,
                histtype='step',
                linewidth=2.8,
                color='red',
                label='Smeared'
            )

            ax.hist(
                mass_corrected,
                bins=bins,
                histtype='step',
                linewidth=2.8,
                color='blue',
                label='Smeared + corrected'
            )

            ax.set_yscale('log')

            ax.set_xlim(x_bounds)

            ax.set_ylim(y_bounds)

            ax.set_xlabel(
                '4-object invariant mass [GeV]',
                fontsize=16
            )

            ax.set_ylabel(
                f'Events / {bin_width} GeV',
                fontsize=16
            )

            ax.set_title(
                rf'$m_{{W_R}}$={mWR} GeV, '
                rf'$m_N$={mN} GeV',
                fontsize=16
            )

            ax.grid(True, alpha=0.3, which='both')

            ax.legend(fontsize=13)

            fig.tight_layout()

            mass_save_path = os.path.join(
                mass_output_dir,
                f'mWR_{mWR}_mN_{mN}_comparison.pdf'
            )

            fig.savefig(mass_save_path)

            plt.close(fig)

            # =================================================
            # s1 HISTOGRAM
            # =================================================

            fig_s1, ax_s1 = plt.subplots(figsize=(8, 6))

            s_bins = np.arange(0.5, 1.5001, 0.01)

            ax_s1.hist(
                s1,
                bins=s_bins,
                histtype='stepfilled',
                linewidth=1.8
            )

            ax_s1.set_xlim([0.5, 1.5])

            ax_s1.set_xlabel(r'$s_1$', fontsize=16)

            ax_s1.set_ylabel('Events', fontsize=16)

            ax_s1.set_title(
                rf'$s_1$ distribution'
                '\n'
                rf'$m_{{W_R}}$={mWR} GeV, '
                rf'$m_N$={mN} GeV',
                fontsize=15
            )

            ax_s1.grid(True, alpha=0.3)

            fig_s1.tight_layout()

            s1_save_path = os.path.join(
                s_output_dir,
                f'mWR_{mWR}_mN_{mN}_s1.pdf'
            )

            fig_s1.savefig(s1_save_path)

            plt.close(fig_s1)

            # =================================================
            # s2 HISTOGRAM
            # =================================================

            fig_s2, ax_s2 = plt.subplots(figsize=(8, 6))

            ax_s2.hist(
                s2,
                bins=s_bins,
                histtype='stepfilled',
                linewidth=1.8
            )

            ax_s2.set_xlim([0.5, 1.5])

            ax_s2.set_xlabel(r'$s_2$', fontsize=16)

            ax_s2.set_ylabel('Events', fontsize=16)

            ax_s2.set_title(
                rf'$s_2$ distribution'
                '\n'
                rf'$m_{{W_R}}$={mWR} GeV, '
                rf'$m_N$={mN} GeV',
                fontsize=15
            )

            ax_s2.grid(True, alpha=0.3)

            fig_s2.tight_layout()

            s2_save_path = os.path.join(
                s_output_dir,
                f'mWR_{mWR}_mN_{mN}_s2.pdf'
            )

            fig_s2.savefig(s2_save_path)

            plt.close(fig_s2)

            print(f'  ✓ Saved plots for mWR={mWR}, mN={mN}')

        except FloatingPointError as e:

            print('\n================================================')
            print('FLOATING POINT ERROR DETECTED')
            print('================================================')

            print(f'File: {fname}')

            print(f'mWR={mWR}, mN={mN}')

            print(e)

            print('================================================\n')

        except Exception as e:

            print(f'\nGeneral error while processing {fname}')

            print(e)


# ============================================================
# DONE
# ============================================================

print('\n' + '=' * 60)

print('Finished processing all files.')

print('=' * 60)