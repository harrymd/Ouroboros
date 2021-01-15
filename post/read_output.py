import os

import numpy as np

from common import get_Ouroboros_out_dirs

# Loading data from Ouroboros. ------------------------------------------------
def load_eigenfreq_Ouroboros(Ouroboros_info, mode_type, n_q = None, l_q = None, i_toroidal = None):
    
    ## Unpack the Ouroboros parameters.
    #dir_output  = Ouroboros_info['dir_output']
    #n_layers    = Ouroboros_info['n_layers']
    #n_max       = Ouroboros_info['n_lims'][1]
    #l_max       = Ouroboros_info['l_lims'][1]
    #g_switch    = Ouroboros_info['g_switch']

    ## By default, the toroidal modes have g_switch = 0.
    #if mode_type == 'T':

    #    g_switch = 0

    if mode_type == 'T':

        assert i_toroidal is not None, 'For toroidal modes, the optional argument \'i toroidal\' must specify the layer number.'

    # Generate the name of the output directory based on the Ouroboros
    # parameters.
    _, _, _, dir_eigval  = get_Ouroboros_out_dirs(Ouroboros_info, mode_type)

    # Generate the name of the output file.
    if i_toroidal is None:

        file_eigval = 'eigenvalues.txt'

    else:

        file_eigval = 'eigenvalues_{:>03d}.txt'.format(i_toroidal)

    path_eigval = os.path.join(dir_eigval, file_eigval)

    # Load the data from the output file.
    n, l, f = np.loadtxt(path_eigval).T
    
    # Convert single-value output files to arrays.
    n = np.atleast_1d(n)
    l = np.atleast_1d(l)
    f = np.atleast_1d(f)
    
    # Convert radial and angular order to integer type.
    n = n.astype(np.int)
    l = l.astype(np.int)
    
    # If an optional query (n_q, l_q) is supplied, find the mode matching
    # that query and return the frequency.
    if (n_q is not None) and (l_q is not None):

        i = np.where((n == n_q) & (l == l_q))[0][0]

        f_q = f[i]

        return f_q

    else:

        return n, l, f

def load_eigenfunc_Ouroboros(Ouroboros_info, mode_type, n, l, i_toroidal = None):

    # Unpack the Ouroboros parameters.
    dir_output  = Ouroboros_info['dir_output']
    n_layers    = Ouroboros_info['n_layers']
    n_max       = Ouroboros_info['n_lims'][1]
    l_max       = Ouroboros_info['l_lims'][1]
    g_switch    = Ouroboros_info['g_switch']

    # By default, the toroidal modes have g_switch = 0.
    if mode_type == 'T':

        g_switch = 0

    if mode_type == 'T':

        assert i_toroidal is not None, 'For toroidal modes, the optional argument \'i toroidal\' must specify the layer number.'

    # The eigenfunction files files have different names depending on the mode type.
    # This is due to the automatic separation of uncoupled toroidal modes
    if i_toroidal is None:

        dir_eigenfuncs = 'eigenfunctions'

    else:

        dir_eigenfuncs = 'eigenfunctions_{:>03d}'.format(i_toroidal)

    # Find the directory containing the eigenfunctions (which is contained
    # within the eigenvalue directory) based on the Ouroboros parameters.
    _, _, _, dir_eigval      = get_Ouroboros_out_dirs(Ouroboros_info, mode_type)
    dir_eigenfuncs  = os.path.join(dir_eigval, dir_eigenfuncs)
    file_eigenfunc  = '{:>05d}_{:>05d}.npy'.format(n, l)
    path_eigenfunc  = os.path.join(dir_eigenfuncs, file_eigenfunc)
    
    # Load the eigenfunction.
    
    # Radial case.
    if mode_type == 'R':

        r, U = np.load(path_eigenfunc)

        return r, U
    
    # Spheroidal case.
    elif mode_type == 'S':

        r, U, V = np.load(path_eigenfunc)
        
        return r, U, V
    
    # Toroidal case.
    elif mode_type in ['T', 'I']:

        r, W = np.load(path_eigenfunc)

        return r, W
    
    # Error catching.
    else:

        raise NotImplementedError

def get_kernel_dir(dir_output, Ouroboros_info, mode_type):
    
    # Unpack the Ouroboros run information.
    name_model  = Ouroboros_info['model']
    n_layers    = Ouroboros_info['n_layers']
    n_max       = Ouroboros_info['n_lims'][1]
    l_max       = Ouroboros_info['l_lims'][1]
    g_switch    = Ouroboros_info['g_switch']
    version     = Ouroboros_info['version']

    #dir_base = os.path.join(os.sep, 'Users', 'hrmd_work', 'Documents', 'research', 'stoneley')
    #dir_output = os.path.join(dir_base, 'output', 'Ouroboros_py_v3_hrmd')

    dir_RPNM = os.path.join(dir_output, version)

    # Define run parameters.
    name_model  = 'prem_noq_noocean'

    # Set the 'g_switch': 0 -> noG, 1 -> G, 2 -> GP.
    #g_switch = 1 
    # Get the g_switch string.
    g_switch_strs = ['noGP', 'G', 'GP']
    g_switch_str = g_switch_strs[g_switch]
    
    # Set the mode type: 'T': toroidal, 'R': radial, 'S': spheroidal.
    mode_type   = 'S'
    
    # Set the 'switch' string.
    if mode_type == 'T':

        switch = 'T'

    else:

        switch = '{:}_{:}'.format(mode_type, g_switch_str)
    
    # Safety checks.
    if mode_type == 'T':

        assert g_switch == 0, 'For consistency in file naming, if mode_type == \'T\' (toroidal modes), it is required that g_switch == 0 (although the g_switch variable does not affect the calculation of toroidal modes).'

    name_model_with_layers = '{:}_{:>05d}'.format(name_model, n_layers)
    dir_model = os.path.join(dir_RPNM, name_model_with_layers)

    name_run = '{:>05d}_{:>05d}_{:1d}'.format(n_max, l_max, g_switch)
    dir_run = os.path.join(dir_model, name_run)

    dir_type = os.path.join(dir_run, mode_type)
    dir_kernels = os.path.join(dir_type, 'kernels')

    return dir_kernels
