"""
Quick 4-object and 3-object mass histogram analyzer
Loads NPZ files and creates histograms with different mN masses overlaid for each mWR
Creates both smeared and unsmeared plots for various observables
Filters to keep only 1,000,000 event files
No dependency on dataset.py or figagg.py - uses matplotlib directly
"""

import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import os
import re
from collections import defaultdict

# Configuration
data_dir = 'simulation_npz_data/'
output_base_dir = 'mass_histograms/'

# Observable configurations: (name, obs_type, particles, x_bounds, bin_width)
# obs_type: 'mass' (sum), 'pt' (individual), 'eta' (individual), 'phi' (individual)
observable_configs = [
    # Invariant masses
    ('4obj_mass', 'mass', ['lW', 'lN', 'j1', 'j2'], [0, 5000], 50),
    ('3obj_mass', 'mass', ['lN', 'j1', 'j2'], [0, 4000], 50),
    # Transverse momentum
    ('lN_pt', 'pt', ['lN'], [0, 1500], 25),
    ('lW_pt', 'pt', ['lW'], [0, 1500], 25),
    ('j1_pt', 'pt', ['j1'], [0, 1500], 25),
    ('j2_pt', 'pt', ['j2'], [0, 1500], 25),
    # Pseudorapidity
    ('lN_eta', 'eta', ['lN'], [-4, 4], 0.2),
    ('lW_eta', 'eta', ['lW'], [-4, 4], 0.2),
    ('j1_eta', 'eta', ['j1'], [-4, 4], 0.2),
    ('j2_eta', 'eta', ['j2'], [-4, 4], 0.2),
    # Azimuthal angle
    ('lN_phi', 'phi', ['lN'], [-np.pi, np.pi], np.pi/20),
    ('lW_phi', 'phi', ['lW'], [-np.pi, np.pi], np.pi/20),
    ('j1_phi', 'phi', ['j1'], [-np.pi, np.pi], np.pi/20),
    ('j2_phi', 'phi', ['j2'], [-np.pi, np.pi], np.pi/20),
]

y_bounds = [10**0, 10**8]

# Create output directories if they don't exist
for obs_name, _, _, _, _ in observable_configs:
    os.makedirs(os.path.join(output_base_dir, f'{obs_name}_smeared'), exist_ok=True)
    os.makedirs(os.path.join(output_base_dir, f'{obs_name}_unsmeared'), exist_ok=True)

# Parse data files and group by mWR - filter for only 1,000,000 event files
files_by_mWR = defaultdict(list)

npz_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npz')])

for data_file_name in npz_files:
    mWR_match = re.search(r'mWR_(\d+)', data_file_name)
    mN_match = re.search(r'mN_(\d+)', data_file_name)
    num_events_match = re.search(r'(\d+)events_sim_data', data_file_name)
    
    if mWR_match and mN_match and num_events_match:
        num_events = int(num_events_match.group(1))
        
        # Only keep files with exactly 1,000,000 events
        if num_events == 1000000:
            mWR = int(mWR_match.group(1))
            mN = int(mN_match.group(1))
            files_by_mWR[mWR].append((mN, data_file_name))

# Sort each mWR's files by mN
for mWR in files_by_mWR:
    files_by_mWR[mWR].sort(key=lambda x: x[0])

print(f"Found {len(files_by_mWR)} WR mass points (1,000,000 events only)")
for mWR in sorted(files_by_mWR.keys()):
    mN_list = [mN for mN, _ in files_by_mWR[mWR]]
    print(f"  mWR = {mWR} GeV: mN = {mN_list}")

# Create histograms for each mWR
style = hep.style.CMS
style['font.size'] = 13
plt.style.use(style)

# Color palette for different mN
colors = ['#1eb935', '#1e25b9', '#b96c1e', '#000000', '#7c1eb9', '#b91e1e']

# Loop through observable types
for obs_name, obs_type, particle_keys, x_bounds, bin_width in observable_configs:
    print(f"\n{'='*60}")
    print(f"Creating {obs_name} plots")
    print(f"{'='*60}")
    
    # Loop through both smeared and unsmeared versions
    for smear_type in ['smeared', 'unsmeared']:
        print(f"\n{'-'*60}")
        print(f"Creating {smear_type.upper()} {obs_name} plots...")
        print(f"{'-'*60}")
        
        output_dir = os.path.join(output_base_dir, f'{obs_name}_{smear_type}')
        data_suffix = '_smeared' if smear_type == 'smeared' else ''
        
        for mWR in sorted(files_by_mWR.keys()):
            print(f"\nProcessing mWR = {mWR} GeV ({smear_type} {obs_name})...")
            
            fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
            
            for color_idx, (mN, data_file_name) in enumerate(files_by_mWR[mWR]):
                try:
                    # Load data
                    data_path = os.path.join(data_dir, data_file_name)
                    data = np.load(data_path)
                    
                    # Extract particle data (smeared or unsmeared)
                    particle_data = {}
                    for particle in particle_keys:
                        key = f'{particle}_data{data_suffix}'
                        particle_data[particle] = data[key]
                    
                    # Compute observable based on type
                    if obs_type == 'mass':
                        # Compute invariant mass using 4-vector addition
                        # Each entry is [E, px, py, pz]
                        total_E = sum(particle_data[p][:, 0] for p in particle_keys)
                        total_px = sum(particle_data[p][:, 1] for p in particle_keys)
                        total_py = sum(particle_data[p][:, 2] for p in particle_keys)
                        total_pz = sum(particle_data[p][:, 3] for p in particle_keys)
                        
                        # Compute invariant mass: m = sqrt(E^2 - p^2)
                        p_squared = total_px**2 + total_py**2 + total_pz**2
                        values = np.sqrt(np.maximum(total_E**2 - p_squared, 0))
                        obs_label = 'invariant mass [GeV]'
                        
                    elif obs_type == 'pt':
                        # Transverse momentum
                        particle = particle_keys[0]
                        px = particle_data[particle][:, 1]
                        py = particle_data[particle][:, 2]
                        values = np.sqrt(px**2 + py**2)
                        obs_label = f'{particle} $p_T$ [GeV]'
                        
                    elif obs_type == 'eta':
                        # Pseudorapidity
                        particle = particle_keys[0]
                        px = particle_data[particle][:, 1]
                        py = particle_data[particle][:, 2]
                        pz = particle_data[particle][:, 3]
                        pt = np.sqrt(px**2 + py**2)
                        values = np.arcsinh(pz / np.maximum(pt, 1e-10))
                        obs_label = f'{particle} $\\eta$'
                        
                    elif obs_type == 'phi':
                        # Azimuthal angle
                        particle = particle_keys[0]
                        px = particle_data[particle][:, 1]
                        py = particle_data[particle][:, 2]
                        values = np.arctan2(py, px)
                        obs_label = f'{particle} $\\phi$ [rad]'
                    
                    # Create bins
                    bins = np.arange(x_bounds[0], x_bounds[1] + bin_width, bin_width)
                    
                    # Plot histogram
                    color = colors[color_idx % len(colors)]
                    ax.hist(
                        values,
                        bins=bins,
                        histtype='step',
                        linewidth=2.5,
                        color=color,
                        label=f'$m_N$ = {mN} GeV (N={len(values)})',
                    )
                    
                    print(f"  ✓ mN = {mN}: {len(values)} events, range: {values.min():.1f} - {values.max():.1f}")
                    
                except Exception as e:
                    print(f"  ✗ Error loading {data_file_name}: {e}")
            
            # Configure plot
            ax.set_yscale('log')
            ax.set_ylim(y_bounds)
            ax.set_xlim(x_bounds)
            ax.set_xlabel(obs_label, fontsize=16)
            ax.set_ylabel(f'Events / {bin_width:.3g}', fontsize=16)
            
            # Add smearing info to title
            smear_label = "(Smeared)" if smear_type == 'smeared' else "(Unsmeared)"
            ax.set_title(f'Monte Carlo $W_R$ Decay - {obs_name} ($m_{{W_R}}$ = {mWR} GeV) {smear_label}', fontsize=14)
            ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
            ax.grid(True, alpha=0.3, which='both')
            
            # Save figure
            fig_path = os.path.join(output_dir, f'mWR_{mWR}_{obs_name}_{smear_type}.png')
            fig.tight_layout()
            fig.savefig(fig_path, dpi=150)
            print(f"  → Saved: {fig_path}")
            plt.close(fig)

print("\n" + "="*60)
print("✓ All histograms created successfully!")
print(f"Output saved to: {output_base_dir}")
print("="*60)