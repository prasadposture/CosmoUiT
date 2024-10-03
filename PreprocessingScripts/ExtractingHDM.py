import numpy as np
from CosmoFuncs import file_reader

DM = file_reader('/media/disk2/prasad/215Mpc_DM_HM/DM_map_7.000')
np.save('/media/disk2/prasad/215Mpc_DM_HM/DMField.npy', arr=DM)

Halo = file_reader('/media/disk2/prasad/215Mpc_DM_HM/Halo_map_7.000')
np.save('/media/disk2/prasad/215Mpc_DM_HM/HaloField.npy', arr=Halo)