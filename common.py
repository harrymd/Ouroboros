import os
from itertools import groupby
from operator import itemgetter

import numpy as np

# Manipulating directories. ---------------------------------------------------
def mkdir_if_not_exist(dir_):
    '''
    Create a directory if it does not already exist.
    '''

    if not os.path.isdir(dir_):
        
        print('Making directory {:}'.format(dir_))
        os.mkdir(dir_)

    return

def get_Ouroboros_out_dirs(Ouroboros_info, mode_type):
    
    # Unpack the Ouroboros parameters.
    dir_output  = Ouroboros_info['dir_output']
    name_model  = Ouroboros_info['name_model']
    n_layers    = Ouroboros_info['n_layers']
    n_max       = Ouroboros_info['n_lims'][1]
    l_max       = Ouroboros_info['l_lims'][1]
    g_switch    = Ouroboros_info['g_switch']

    # Create model directory if it doesn't exist.
    name_model_with_layers = '{:}_{:>05d}'.format(name_model, n_layers)
    dir_model = os.path.join(dir_output, name_model_with_layers)
    mkdir_if_not_exist(dir_model)

    # Create the run directory if it doesn't exist.
    name_run = '{:>05d}_{:>05d}'.format(n_max, l_max)
    dir_run = os.path.join(dir_model, name_run)

    # By default, the toroidal modes have g_switch = 0.
    if mode_type == 'T':

        g_switch = 0

    # Find the output file.
    name_model_with_layers  = '{:}_{:>05d}'.format(name_model, n_layers)
    dir_model               = os.path.join(dir_output, name_model_with_layers)
    name_run                = '{:>05d}_{:>05d}'.format(n_max, l_max)

    if mode_type in ['R', 'S']:

        dir_g = os.path.join(dir_run, 'grav_{:1d}'.format(Ouroboros_info['g_switch']))
        dir_type = os.path.join(dir_g, mode_type)

    else:

        dir_g = None
        dir_type = os.path.join(dir_run, mode_type)

    return dir_model, dir_run, dir_g, dir_type

# Reading input files. --------------------------------------------------------
def read_Ouroboros_input_file(path_input_file):
    '''
    Reads the RadialPNM input file.
    '''

    # Announcement
    print('Reading input file, {:}'.format(path_input_file))
    
    # Read line by line.
    with open(path_input_file, 'r') as in_id:
        
        path_model      = in_id.readline().split()[1]
        dir_output      = in_id.readline().split()[1]
        g_switch        = int(in_id.readline().split()[1])
        mode_types      = in_id.readline().split()[1:]
        l_min, l_max    = [int(x) for x in in_id.readline().split()[1:]]
        n_min, n_max    = [int(x) for x in in_id.readline().split()[1:]]
        n_layers        = int(in_id.readline().split()[1])

    name_model = os.path.splitext(os.path.basename(path_model))[0]

    ## Check values.
    #assert os.path.isfile(path_model), 'Model file ({:}) does not exist'.format(path_model)
    #assert os.path.isdir(dir_output), 'Output directory ({:}) does not exist'.format(dir_output)
    #assert g_switch in [0, 1, 2], 'g_switch variable ({:}) should be 0, 1 or 2'.format(g_switch)
    #assert mode_type in ['R', 'S', 'T'], 'mode_type variable ({:}) should be R, S or T'.format(mode_type)

    #if mode_type in ['S', 'T']:

    #    assert l_min <= l_max, 'l_min ({:}) should be less than or equal to l_max ({:})'.format(l_min, l_max) 
    #    assert n_min <= n_max, 'n_min ({:}) should be less than or equal to n_max ({:})'.format(n_min, n_max) 

    # Print input file information.
    print('Input file was read successfully:')
    print('Path to model file: {:}'.format(path_model))
    print('Output directory: {:}'.format(dir_output))
    print('Gravity switch: {:1d}'.format(g_switch))
    print('Mode types:', end = '')
    for mode_type in mode_types:
        print(' {:1}'.format(mode_type), end = '')
    print('\n', end = '')
    if ('S' in mode_types) or ('T' in mode_types): 
        print('l range: {:d} to {:d}'.format(l_min, l_max))
    print('n range: {:d} to {:d}'.format(n_min, n_max))
    print('Number of layers: {:d}'.format(n_layers))

    # Store in dictionary.
    Ouroboros_info = dict()
    Ouroboros_info['path_model']    = path_model
    Ouroboros_info['name_model']    = name_model
    Ouroboros_info['dir_output']    = dir_output 
    Ouroboros_info['g_switch']      = g_switch 
    Ouroboros_info['mode_types']    = mode_types
    Ouroboros_info['l_lims']        = [l_min, l_max]
    Ouroboros_info['n_lims']        = [n_min, n_max]
    Ouroboros_info['n_layers']      = n_layers

    return Ouroboros_info

def read_Mineos_input_file():
    '''
    Reads the Mineos input file.
    '''

    input_file = 'input_Mineos.txt'
    print('Reading input file, {:}'.format(input_file))
    
    # Read line by line.
    with open(input_file, 'r') as in_id:
        
        path_model      = in_id.readline().strip()
        dir_output      = in_id.readline().strip()
        g_switch        = int(in_id.readline())
        mode_type       = in_id.readline().strip()
        l_min, l_max    = [int(x) for x in in_id.readline().split()]
        n_min, n_max    = [int(x) for x in in_id.readline().split()]

    # Check values.
    assert os.path.isfile(path_model), 'Model file ({:}) does not exist'.format(path_model)
    assert os.path.isdir(dir_output), 'Output directory ({:}) does not exist'.format(dir_output)
    assert g_switch in [0, 1, 2], 'g_switch variable ({:}) should be 0, 1 or 2'.format(g_switch)
    assert mode_type in ['R', 'S', 'T'], 'mode_type variable ({:}) should be R, S or T'.format(mode_type)

    if mode_type in ['S', 'T']:

        assert l_min <= l_max, 'l_min ({:}) should be less than or equal to l_max ({:})'.format(l_min, l_max) 
        assert n_min <= n_max, 'n_min ({:}) should be less than or equal to n_max ({:})'.format(n_min, n_max) 

    # Print input file information.
    print('Input file was read successfully:')
    print('Path to model file: {:}'.format(path_model))
    print('Output directory: {:}'.format(dir_output))
    print('Gravity switch: {:1d}'.format(g_switch))
    print('Mode type: {:1}'.format(mode_type))
    if mode_type in ['S', 'T']:
        print('l range: {:d} to {:d}'.format(l_min, l_max))
    print('n range: {:d} to {:d}'.format(n_min, n_max))

    return path_model, dir_output, g_switch, mode_type, l_min, l_max, n_min, n_max 

def load_model(model_path, skiprows = 3):
    '''
    Load a planetary model. Models are given in Mineos format (tabular setting); see the Mineos manual, section 3.1.2.1. This means that all units are S.I. units.
    '''

    # Load the model data.
    # Allow models with no header lines.
    try:

        model_data = np.loadtxt(model_path, skiprows = 0) 

    except:

        model_data = np.loadtxt(model_path, skiprows = skiprows) 

    # Simple format.
    if model_data.shape[1] == 4: 

        r, rho, v_p, v_s = model_data.T

    # Mineos format.
    elif model_data.shape[1] == 9:

        # Extract the variables.
        # r         Radial coordinate in m.
        # rho       Density in kg/m3.
        # v_p       Simple isotropic mean P-wave speed in m/s.
        # v_s       Simple isotropic mean S-wave speed in m/s.
        r   =  model_data[:,0]
        rho =  model_data[:,1]
        v_p = (model_data[:, 2] + model_data[:, 6])/2.0
        v_s = (model_data[:, 3] + model_data[:, 7])/2.0
    
    # Store in a dictionary.
    model = dict()
    model['r'] = r
    model['rho'] = rho
    model['v_p'] = v_p
    model['v_s'] = v_s
    model['n_layers'] = len(r)

    return model

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
        U[0] = 0.0 # Bug in Ouroboros causes U[0] to be large.

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

# Manipulating Earth models. --------------------------------------------------
def get_r_fluid_solid_boundary(radius, vs):

    n_layers = len(radius)
    i_fluid = np.where(vs == 0.0)[0]
    
    i_solid_fluid_boundary = []
    # This is based on https://docs.python.org/2.6/library/itertools.html#examples,
    # second example in section 9.7.2.
    for k, g in groupby(enumerate(i_fluid), lambda ix: ix[0] - ix[1]):

        i_fluid_subsection = list(map(itemgetter(1), g))
        if (i_fluid_subsection[0] != 0) and (i_fluid_subsection[0] != (n_layers - 1)):

            i_solid_fluid_boundary.append(i_fluid_subsection[0])

        if (i_fluid_subsection[-1] != 0) and (i_fluid_subsection[-1] != (n_layers - 1)):

            #i_solid_fluid_boundary.append(i_fluid_subsection[-1])
            i_solid_fluid_boundary.append(i_fluid_subsection[-1] + 1)

    r_solid_fluid_boundary = [radius[i] for i in i_solid_fluid_boundary]

    return i_fluid, r_solid_fluid_boundary, i_solid_fluid_boundary
