# Used for converting 384^3 Array to 48^3 Array 
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import block_reduce
from tqdm import trange

# for Neutral Fraction Fields
num_files=7204
for i in trange(num_files, desc='Files Reduced'):
    np.save(f'/media/disk2/prasad/ReducedData48/xHI{i}.npy', arr=block_reduce(np.load(f'/media/Disk2/prasad/ProcessedDataNPY/xHI{i}.npy'), block_size=(8,8,8), func=np.mean))


# for Dark Matter Field
dm = np.load('/media/disk2/prasad/215Mpc_DM_HM/DMField.npz')['array'] #384^3 array
print(dm.shape)
red_dm = block_reduce(dm, block_size=(8,8,8), func=np.mean)
print(red_dm.shape)
np.save('/media/disk2/prasad/215Mpc_DM_HM/ReducedDM.npy', arr=red_dm)

# for Halo Field
halo = np.load('/media/disk2/prasad/215Mpc_DM_HM/HaloField.npz')['array'] #384^3 array
print(halo.shape)
red_halo= block_reduce(halo, block_size=(8,8,8), func=np.mean)
print(red_halo.shape)
np.save('/media/disk2/prasad/215Mpc_DM_HM/ReducedHalo.npy', arr=red_halo)