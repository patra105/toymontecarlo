"""
Inspect NPZ file structure to see what keys are available
"""
import numpy as np
import os

data_dir = 'simulation_npz_data/'
sample_file = os.path.join(data_dir, os.listdir(data_dir)[0])

print(f"Inspecting: {sample_file}\n")
data = np.load(sample_file)

print("Available keys in NPZ file:")
for key in sorted(data.keys()):
    print(f"  - {key}: shape {data[key].shape}, dtype {data[key].dtype}")