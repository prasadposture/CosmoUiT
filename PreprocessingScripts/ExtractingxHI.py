import numpy as np
from CosmoFuncs import fld_rdr_xhi
import glob
import os

file_paths = glob.glob(os.path.join('/media/disk2/prasad', 'xHI_map_z_7.000_*_*_*'))
# this file doesnt exist since original data has been deleted after extraction

params = []
for i, path in enumerate(file_paths):
    file_name = os.path.basename(path)
    _, _, _, _, Mh, Nion, Rmfp = file_name.split('-')
    params.append([Mh, Nion, Rmfp])
    num_array = fld_rdr_xhi(path)
    output_path = os.path.join('/media/disk2/prasad/ProcessedData', f'xHI{i}.npy')
    np.save(output_path, arr=num_array)

np.save(os.path.join('/media/disk2/prasad/ProcessedData', 'Params.npy'), arr=np.array(params))