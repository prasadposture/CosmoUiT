import numpy as np
import matplotlib.pyplot as plt


# File Readers
def file_reader(filename):
    """
    Reads halo / density maps as 3D numpy array
    """
    f = open(filename) #path to map
    N  = np.fromfile(f, count=3, dtype='uint64')
    N1,N2,N3 = N
    l = np.fromfile(f, count=1, dtype='float32')
    data1 = np.fromfile(f, count=N1*N2*N3, dtype='float32')
    f.close()
    data = np.reshape(data1, (N1,N2,N3), order='C') # row major order: going row-by-row
    return data

def fld_rdr(filename):
    """
    fld_rdr: field reader
    Used for reading the obtained 21cm fields of ReionYuga 
    """
    f = open(filename)
    N = np.fromfile(f, count=3, dtype='uint32')
    N1, N2, N3 =  N
    data = np.fromfile(f, count=N1*N2*N3, dtype='float32')
    f.close()
    return np.reshape(data, (N1, N2, N3), order='C')

# has different format?
def fld_rdr_xhi(filename):
    """
    fld_rdr: field reader
    Used for reading the obtained neutral fraction fields
    """
    with open(filename, 'rb') as f:
        N = np.fromfile(f, count=3, dtype='int32')
        N1, N2, N3 = N
        # grid_spacing = np.fromfile(f, count= 1, dtype='float32')
        data = np.fromfile(f, count=N1*N2*N3, dtype='float32')
        # print(N1, N2, N3)
        # print(grid_spacing)
    return np.reshape(data, (N1, N2, N3), order='C')

def CII_reader(file):
    """
    Function for reading te CII maps

    Args:
        file (string): file path

    Returns:
        3D Array containing the CII maps with intensity
    """
    f=open(file,'rb')
    N=np.fromfile(f,count=3,dtype='int64')
    grid_spacing=np.fromfile(f,count=1,dtype='float32')
    data=np.fromfile(f,count=N[0]*N[1]*N[2],dtype='float32')
    f.close()
    data_reshape=data.reshape((N[0],N[1],N[2]), order='C')
    return data_reshape  


# Array plotters
def plotter(grid_3d, title=None, slice_index=16, vmx=None, vmn=None):
    plt.figure(figsize=(10,11.5))
    plt.imshow(grid_3d[:, :, slice_index], cmap='magma', origin='lower', extent=[0, grid_3d.shape[0], 0, grid_3d.shape[1]], vmax=vmx, vmin=vmn)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.colorbar()
    plt.title(title)
    plt.savefig(f'{title}.png')
    plt.show()
    
def plotter_3d(grid_3d, title=None):
    N = grid_3d.shape[0]
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    x = np.linspace(-10, 10, N, endpoint = True)
    y = np.linspace(-10, 10, N, endpoint = True)
    z = np.linspace(-10, 10, N, endpoint = True)
    X, Y, Z = np.meshgrid(x,y,z)
    ax.scatter(X.flatten(), Y.flatten(), Z.flatten(), c=grid_3d, cmap='magma')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    norm = plt.Normalize(grid_3d.min(), grid_3d.max())
    sm = plt.cm.ScalarMappable(cmap='Spectral', norm=norm)
    fig.colorbar(mappable=sm, ax=ax, pad=0.05)
    

# Convolution Operation Calculators
def Conv3DPOutput(input_size, filter_size, stride, padding):
    """

    Args:
        input_size (_type_): _description_
        filter_size (_type_): _description_
        stride (_type_): _description_
        padding (_type_): _description_

    Returns:
        int : the shape of the output array after convolution along an axis
    """
    return np.floor(((input_size - filter_size + 2*padding)/stride)+1)

def ConvTranspose3DOutput(input_size, filter_size, stride, input_padding, output_padding):
    """Note: Keep the output padding smaller than the input padding

    Args:
        input_size (_type_): _description_
        filter_size (_type_): _description_
        stride (_type_): _description_
        input_padding (_type_): _description_
        output_padding (_type_): _description_
        
    return:
        the shape of the output array after deconvolution along an axis
    """
    return np.floor((input_size-1)*stride -2*input_padding + filter_size + output_padding) #floor or ceiling since now we are upsampling here


def notify(task):
    """
    Add the task you just completed!
    """
    import smtplib
    smtObj = smtplib.SMTP('smtp.gmail.com', 587)
    type(smtObj)
    smtObj.ehlo()
    smtObj.starttls()
    smtObj.login('posture.prasad.19bsc049@gmail.com', 'momu lciv khtu ephx')
    smtObj.sendmail('posture.prasad.19bsc049@gmail.com', 'msc2303121013@iiti.ac.in', f'Subject: Code Alert. \n{task}')