import os
from itertools import groupby
from operator import itemgetter

import numpy as np
try:
    from    obspy.geodetics             import gps2dist_azimuth
except ModuleNotFoundError:
    print('Could not import obspy. Synthetic seismograms will not be available.')

# Define mapping between Mineos jcom switch and mode-type character.
jcom_to_mode_type_dict = {1 : 'R', 2 : 'T', 3 : 'S', 4 : 'I'}

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
    grav_switch    = Ouroboros_info['grav_switch']

    # Create model directory if it doesn't exist.
    name_model_with_layers = '{:}_{:>05d}'.format(name_model, n_layers)
    dir_model = os.path.join(dir_output, name_model_with_layers)
    mkdir_if_not_exist(dir_model)

    # Create the run directory if it doesn't exist.
    name_run = '{:>05d}_{:>05d}'.format(n_max, l_max)
    dir_run = os.path.join(dir_model, name_run)

    # By default, the toroidal modes have g_switch = 0.
    if mode_type == 'T':

        grav_switch = 0

    # Find the output file.
    name_model_with_layers  = '{:}_{:>05d}'.format(name_model, n_layers)
    dir_model               = os.path.join(dir_output, name_model_with_layers)
    name_run                = '{:>05d}_{:>05d}'.format(n_max, l_max)

    if mode_type in ['R', 'S']:

        dir_g = os.path.join(dir_run, 'grav_{:1d}'.format(Ouroboros_info['grav_switch']))
        dir_type = os.path.join(dir_g, mode_type)

    else:

        dir_g = None
        dir_type = os.path.join(dir_run, mode_type)

    return dir_model, dir_run, dir_g, dir_type

def get_Ouroboros_summation_out_dirs(run_info, summation_info, name_summation_dir = 'summation'):

    summation_info['dir_summation'] = os.path.join(run_info['dir_run'], name_summation_dir)
    summation_info['dir_channels'] = os.path.join(summation_info['dir_summation'], summation_info['name_channels'])
    summation_info['dir_cmt'] = os.path.join(summation_info['dir_channels'], summation_info['name_cmt'])
    
    if summation_info['path_mode_list'] is not None:

        summation_info['dir_mode_list'] = os.path.join(summation_info['dir_cmt'],
                summation_info['name_mode_list'])
        summation_info['dir_output'] = summation_info['dir_mode_list']

    else:

        summation_info['dir_mode_list'] = None
        summation_info['dir_output'] = summation_info['dir_cmt']

    return summation_info

def get_Mineos_out_dirs(run_info):
    '''
    Returns the directory containing the Mineos eigenvalue output.
    '''

    dir_model = os.path.join(run_info['dir_output'], run_info['name_model'])
    name_run = '{:>05d}_{:>05d}_{:1d}'.format(run_info['n_lims'][1], run_info['l_lims'][1], run_info['grav_switch'])
    dir_run = os.path.join(dir_model, name_run)

    return dir_model, dir_run

def get_Mineos_summation_out_dirs(run_info, summation_info, file_green_in = None, file_syndat_in = None, file_syndat_out = None, name_summation_dir = 'summation'):

    summation_info['dir_summation'] = os.path.join(run_info['dir_run'], name_summation_dir) 
    summation_info['dir_channels'] = os.path.join(summation_info['dir_summation'], summation_info['name_channels'])
    summation_info['dir_cmt'] = os.path.join(summation_info['dir_channels'], summation_info['name_cmt'])

    if ('path_mode_list' in summation_info.keys()) and (summation_info['path_mode_list'] is not None):

        summation_info['dir_mode_list'] = os.path.join(summation_info['dir_cmt'],
                summation_info['name_mode_list'])
        summation_info['dir_output'] = summation_info['dir_mode_list']

    else:

        summation_info['dir_mode_list'] = None
        summation_info['dir_output'] = summation_info['dir_cmt']
    
    summation_info['path_channel_db'] = os.path.join(summation_info['dir_channels'], 'channel_db')
    summation_info['path_green_out_db'] = os.path.join(summation_info['dir_cmt'], 'green')

    if file_green_in is not None:

        summation_info['file_green_in'] = file_green_in

    if file_syndat_in is not None:

        summation_info['path_syndat_in'] = os.path.join(summation_info['dir_cmt'], file_syndat_in)


    if file_syndat_out is not None:

        summation_info['file_syndat_out'] = file_syndat_out
        summation_info['path_syndat_out'] = os.path.join(summation_info['dir_cmt'], summation_info['file_syndat_out'])

    return summation_info

# Reading input files. --------------------------------------------------------
def read_Ouroboros_input_file(path_input_file):
    '''
    Reads the RadialPNM input file.
    '''

    ## Announcement
    #print('Reading input file, {:}'.format(path_input_file))
    
    # Read line by line.
    with open(path_input_file, 'r') as in_id:
        
        code            = in_id.readline().split()[1]
        assert code == 'ouroboros'
        path_model      = in_id.readline().split()[1]
        dir_output      = in_id.readline().split()[1]
        grav_switch        = int(in_id.readline().split()[1])
        mode_types      = in_id.readline().split()[1:]
        n_min, n_max    = [int(x) for x in in_id.readline().split()[1:]]
        l_min, l_max    = [int(x) for x in in_id.readline().split()[1:]]
        n_layers        = int(in_id.readline().split()[1])
        use_attenuation = bool(int(in_id.readline().split()[1]))
        if use_attenuation:
            f_target_mHz    = float(in_id.readline().split()[1])
        else:
            f_target_mHz = None

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
    print('Gravity switch: {:1d}'.format(grav_switch))
    print('Mode types:', end = '')
    for mode_type in mode_types:
        print(' {:1}'.format(mode_type), end = '')
    print('\n', end = '')
    if ('S' in mode_types) or ('T' in mode_types): 
        print('l range: {:d} to {:d}'.format(l_min, l_max))
    print('n range: {:d} to {:d}'.format(n_min, n_max))
    print('Number of layers: {:d}'.format(n_layers))
    print('Attenuative: {:}'.format(use_attenuation))

    # Store in dictionary.
    Ouroboros_info = dict()
    Ouroboros_info['code']          = code
    Ouroboros_info['path_model']    = path_model
    Ouroboros_info['name_model']    = name_model
    Ouroboros_info['dir_output']    = dir_output 
    Ouroboros_info['grav_switch']      = grav_switch 
    Ouroboros_info['mode_types']    = mode_types
    Ouroboros_info['l_lims']        = [l_min, l_max]
    Ouroboros_info['n_lims']        = [n_min, n_max]
    Ouroboros_info['n_layers']      = n_layers
    Ouroboros_info['use_attenuation'] = use_attenuation
    Ouroboros_info['f_target_mHz']  = f_target_mHz

    return Ouroboros_info

def read_Ouroboros_summation_input_file(path_input):

    # Announcement
    print('Reading input file, {:}'.format(path_input))
    
    # Read line by line.
    with open(path_input, 'r') as in_id:
        
        mode_types = in_id.readline().split()[1:]
        f_lims = [float(x) for x in in_id.readline().split()[1:]]
        path_channels = in_id.readline().split()[-1]
        path_cmt = in_id.readline().split()[-1]
        d_t = float(in_id.readline().split()[-1])
        n_samples = int(in_id.readline().split()[-1])
        pulse = in_id.readline().split()[-1]
        output_type = in_id.readline().split()[-1]
        attenuation = in_id.readline().split()[-1]
        correct_response = bool(int(in_id.readline().split()[-1]))
        epi_dist_azi_method = in_id.readline().split()[-1]
        path_mode_list = in_id.readline().split()[-1]

    name_channels = os.path.splitext(os.path.basename(path_channels))[0]
    name_cmt = os.path.splitext(os.path.basename(path_cmt))[0]

    if path_mode_list == 'none':
        
        path_mode_list = None
        name_mode_list = None

    else:

        name_mode_list = os.path.splitext(os.path.basename(path_mode_list))[0]

    # Store in dictionary.
    summation_info = dict()
    summation_info['mode_types'] = mode_types
    summation_info['f_lims'] = f_lims
    summation_info['path_channels'] = path_channels
    summation_info['path_cmt'] = path_cmt
    summation_info['name_channels'] = name_channels
    summation_info['name_cmt'] = name_cmt
    summation_info['d_t'] = d_t
    summation_info['n_samples'] = n_samples
    summation_info['pulse_type'] = pulse
    summation_info['output_type'] = output_type
    summation_info['attenuation'] = attenuation
    summation_info['correct_response'] = correct_response
    summation_info['epi_dist_azi_method'] = epi_dist_azi_method
    summation_info['path_mode_list'] = path_mode_list
    summation_info['name_mode_list'] = name_mode_list

    return summation_info

def read_Mineos_input_file(path_input_file):
    '''
    Reads the Mineos input file.
    '''

    #print('Reading input file, {:}'.format(path_input_file))
    
    # Read line by line.
    with open(path_input_file, 'r') as in_id:
        
        code            = in_id.readline().split()[-1]
        assert code == 'mineos'
        path_model      = in_id.readline().split()[-1]
        dir_output      = in_id.readline().split()[-1]
        grav_switch     = int(in_id.readline().split()[-1])
        mode_types      = [x for x in in_id.readline().split()[1:]]
        n_min, n_max    = [int(x) for x in in_id.readline().split()[1:]]
        l_min, l_max    = [int(x) for x in in_id.readline().split()[1:]]
        f_min, f_max    = [float(x) for x in in_id.readline().split()[1:]]
        eps             = float(in_id.readline().split()[-1])
        max_depth_str   = in_id.readline().split()[-1]
        if max_depth_str == 'all':

            max_depth = 'all'

        else:

            max_depth = float(max_depth_str)

    name_model = os.path.splitext(os.path.basename(path_model))[0]

    # Check values.
    assert os.path.isfile(path_model), 'Model file ({:}) does not exist'.format(path_model)
    #assert os.path.isdir(dir_output), 'Output directory ({:}) does not exist'.format(dir_output)
    assert grav_switch in [1, 2], 'grav_switch variable ({:}) should be 1 or 2 (Mineos does not support grav_switch == 0).'.format(grav_switch)
    for mode_type in mode_types:
        assert mode_type in ['R', 'S', 'T', 'I'],  'mode_type variable ({:}) should be R, S, T or I'.format(mode_type)
    assert f_min < f_max, 'f_min ({:} mHz) should be less than f_max ({:} mHz)'.format(f_min, f_max)
    assert f_min >= 0.0, 'f_min ({:} mHz) should be greater than zero.'

    if mode_type in ['S', 'T']:

        assert l_min <= l_max, 'l_min ({:}) should be less than or equal to l_max ({:})'.format(l_min, l_max) 
        assert n_min <= n_max, 'n_min ({:}) should be less than or equal to n_max ({:})'.format(n_min, n_max) 

    # Print input file information.
    print('Input file was read successfully:')
    print('Path to model file: {:}'.format(path_model))
    print('Output directory: {:}'.format(dir_output))
    print('Gravity switch: {:1d}'.format(grav_switch))
    print('Mode types:', end = '')
    for mode_type in mode_types:
        print(' {:1}'.format(mode_type), end = '')
    print('\n', end = '')
    if ('S' in mode_types) or ('T' in mode_types): 
        print('l range: {:d} to {:d}'.format(l_min, l_max))
    print('n range: {:d} to {:d}'.format(n_min, n_max))

    # Store in dictionary.
    mineos_info = { 'code'          : code,     
                    'path_model'    : path_model,
                    'name_model'    : name_model,
                    'dir_output'    : dir_output,
                    'grav_switch'   : grav_switch,
                    'mode_types'    : mode_types,
                    'n_lims'        : [n_min, n_max],
                    'l_lims'        : [l_min, l_max],
                    'f_lims'        : [f_min, f_max],
                    'eps'           : eps,
                    'max_depth'     : max_depth}

    return mineos_info

def read_Mineos_summation_input_file(path_input):
    
    # Read the summation input file.
    print('Reading {:}'.format(path_input))
    with open(path_input, 'r') as in_id:

        path_channels = in_id.readline().split()[1]
        path_cmt = in_id.readline().split()[1]
        f_lims_line = in_id.readline().split()
        if len(f_lims_line) == 2:

            if f_lims_line[1] == 'same':

                f_lims = 'same'
        else:

            f_lims = [float(x) for x in f_lims_line[1:]]
        n_samples = int(in_id.readline().split()[1])
        data_type = int(in_id.readline().split()[1])
        plane = int(in_id.readline().split()[1])

    name_channels = os.path.splitext(os.path.basename(path_channels))[0]
    name_cmt = os.path.splitext(os.path.basename(path_cmt))[0]

    # Store the information in a dictionary.
    summation_info = dict()
    summation_info['path_channels'] = path_channels
    summation_info['name_channels'] = os.path.splitext(os.path.basename(path_channels))[0]
    summation_info['path_cmt'] = path_cmt
    summation_info['name_cmt'] = os.path.splitext(os.path.basename(path_cmt))[0]
    summation_info['f_lims'] = f_lims
    summation_info['n_samples'] = n_samples
    summation_info['data_type'] = data_type
    summation_info['plane'] = plane

    return summation_info

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

def load_model_full(model_path):
    '''
    Load a planetary model. Models are given in Mineos format (tabular setting); see the Mineos manual, section 3.1.2.1. This means that all units are S.I. units.
    '''

    # Read the header.
    # T_ref     Reference period in seconds.
    with open(model_path, 'r') as in_id:

        header = in_id.readline().strip()
        T_ref = float(in_id.readline().split()[1])
        n_layers, i_icb, i_cmb = [int(x) for x in in_id.readline().split()]

    # Load the data.
    model_data = np.loadtxt(model_path, skiprows = 3)

    # Extract the variables.
    # r         Radial coordinate in m.
    # rho       Density in kg/m3.
    # v_pv      P-wave speed vertically polarised (m/s).
    # v_sv      S-wave speed vertically polarised (m/s).
    # Q_k       Bulk attenuation quality factor (units ?).
    # Q_mu      Shear attenuation quality factor (units ?).
    # v_ph      P-wave speed horizontally polarised (m/s).
    # v_sh      S-wave speed horizontally polarised (m/s).
    r       =  model_data[:, 0]
    rho     =  model_data[:, 1]
    v_pv    =  model_data[:, 2]
    v_sv    =  model_data[:, 3]
    Q_ka    =  model_data[:, 4]
    Q_mu    =  model_data[:, 5]
    v_ph    =  model_data[:, 6]
    v_sh    =  model_data[:, 7]
    eta     =  model_data[:, 8]

    # Derived quantities.
    # v_p       Simple isotropic mean P-wave speed in m/s.
    # v_s       Simple isotropic mean S-wave speed in m/s.
    # ka        Bulk modulus (Pa).
    # mu        Shear modulus (Pa). 
    v_p = (v_pv + v_ph)/2.0
    v_s = (v_sv + v_sh)/2.0
    mu = rho*(v_s**2.0)
    ka = rho*((v_p**2.0) - (4.0/3.0)*(v_s**2.0))
    
    # Store in a dictionary.
    model = dict()
    model['header'] = header
    model['T_ref']  = T_ref
    model['f_ref_Hz'] = 1.0/T_ref
    model['r']      = r
    model['rho']    = rho
    model['v_pv']   = v_pv
    model['v_ph']   = v_ph
    model['v_sv']   = v_sv
    model['v_sh']   = v_sh
    model['Q_ka']   = Q_ka
    model['Q_mu']   = Q_mu
    model['eta']    = eta
    model['v_p']    = v_p
    model['v_s']    = v_s
    model['mu']     = mu
    model['ka']     = ka 
    model['n_layers'] = n_layers
    model['i_icb'] = i_icb
    model['i_cmb'] = i_cmb

    return model

def get_path_adjusted_model(run_info):

    name_out = '{:}_at_{:>06.3f}_mHz.txt'.format(run_info['name_model'],
                                               run_info[ 'f_target_mHz'])
    path_out = os.path.join(run_info['dir_output'], name_out)

    return path_out

def mode_types_to_jcoms(mode_types):

    jcoms = []

    for i, mode_type in enumerate(['R', 'T', 'S', 'I']):

        if mode_type in mode_types:

            jcoms.append(i + 1)

    return jcoms

def read_channel_file(path_channel):
    
    inventory = dict()
    
    first_row = True
    with open(path_channel, 'r') as in_id:
        
        lines = in_id.readlines()

        for line in lines:
        
            if not line[0] == '@':
                
                if not first_row:

                    inventory[station] = dict()
                    inventory[station]['channels'] = channels
                    inventory[station]['coords'] = coords

                else:

                    first_row = False

                station, lat_str, lon_str, ele_str = line.split()[0:4]
                lat = float(lat_str)
                lon = float(lon_str)
                ele = float(ele_str)*1.0E3 # Convert to km.
                coords = {'latitude' : lat, 'longitude' : lon, 'elevation' : ele}
                channels = dict() 

            else:
                
                channel = dict()
                line_split = line.split()
                channel_name = line_split[1]
                channel['depth'] = float(line_split[2])
                channel['horiz_angle'] = float(line_split[3])
                channel['vert_angle'] = float(line_split[4])

                channels[channel_name] = channel

            inventory[station] = dict()
            inventory[station]['channels'] = channels
            inventory[station]['coords'] = coords

    return inventory

# Loading data from Ouroboros. ------------------------------------------------
def load_eigenfreq_Ouroboros(Ouroboros_info, mode_type, n_q = None, l_q = None, i_toroidal = None):
    
    if mode_type == 'T':

        assert i_toroidal is not None, 'For toroidal modes, the optional argument \'i toroidal\' must specify the layer number.'

    # Generate the name of the output directory based on the Ouroboros
    # parameters.
    _, _, _, dir_eigval  = get_Ouroboros_out_dirs(Ouroboros_info, mode_type)
    
    # Get name of eigenvalues file.
    file_name = 'eigenvalues'

    if i_toroidal is None:

        file_eigval = '{:}.txt'.format(file_name)

    else:

        file_eigval = '{:}_{:>03d}.txt'.format(file_name, i_toroidal)
    
    path_eigval = os.path.join(dir_eigval, file_eigval)

    # Load the data from the output file.
    n, l, f_0, f, Q = np.loadtxt(path_eigval).T
    #if Ouroboros_info['use_attenuation']:


    #else:

    #    n, l, f = np.loadtxt(path_eigval).T
    
    # Convert single-value output files to arrays.
    n = np.atleast_1d(n)
    l = np.atleast_1d(l)
    f = np.atleast_1d(f)
    if Ouroboros_info['use_attenuation']:

        Q = np.atleast_1d(Q)
    
    # Convert radial and angular order to integer type.
    n = n.astype(np.int)
    l = l.astype(np.int)
    
    # If an optional query (n_q, l_q) is supplied, find the mode matching
    # that query and return the frequency.
    if (n_q is not None) and (l_q is not None):

        i = np.where((n == n_q) & (l == l_q))[0][0]

        f_q = f[i]

        if Ouroboros_info['use_attenuation']:
            
            Q_q = Q[i]
            f_0_q = f_0[i]
            mode_info = {'f' : f_q, 'f_0' : f_0_q, 'Q' : Q_q}

        else:

            mode_info = {'f' : f_q}

    else:

        if Ouroboros_info['use_attenuation']:

            mode_info = {'n' : n, 'l' : l, 'f' : f, 'Q' : Q, 'f_0' : f_0}

        else:

            mode_info = {'n' : n, 'l' : l, 'f' : f}

    return mode_info

def load_eigenfunc_Ouroboros(Ouroboros_info, mode_type, n, l, i_toroidal = None, norm_func = 'mineos', units = 'SI', omega = None):
    '''
    Normalisation
    See Ouroboros/doc/Ouroboros_normalisation_notes.pdf.
    mineos      Use the Mineos/Ouroboros normalisation formula with Mineos units.
    ouroboros   Use the Mineos/Ouroboros normalisation formula with Ouroboros units.
    SI          Use the Mineos/Ouroboros normalisation formula with SI units. 
    DT          Use the Dahlen and Tromp normalisation formula with SI units.
    '''

    ## Unpack the Ouroboros parameters.
    #dir_output  = Ouroboros_info['dir_output']
    #n_layers    = Ouroboros_info['n_layers']
    #n_max       = Ouroboros_info['n_lims'][1]
    #l_max       = Ouroboros_info['l_lims'][1]
    #grav_switch    = Ouroboros_info['grav_switch']

    ## By default, the toroidal modes have g_switch = 0.
    #if mode_type == 'T':

    #    grav_switch = 0

    if mode_type == 'T':

        assert i_toroidal is not None, 'For toroidal modes, the optional argument \'i toroidal\' must specify the layer number.'

    # This is due to the automatic separation of uncoupled toroidal modes
    dir_name = 'eigenfunctions'
    if i_toroidal is None:

        dir_eigenfuncs = '{:}'.format(dir_name)

    else:

        dir_eigenfuncs = '{:}_{:>03d}'.format(dir_name, i_toroidal)

    # Find the directory containing the eigenfunctions (which is contained
    # within the eigenvalue directory) based on the Ouroboros parameters.
    _, _, _, dir_eigval      = get_Ouroboros_out_dirs(Ouroboros_info, mode_type)
    dir_eigenfuncs  = os.path.join(dir_eigval, dir_eigenfuncs)
    file_eigenfunc  = '{:>05d}_{:>05d}.npy'.format(n, l)
    path_eigenfunc  = os.path.join(dir_eigenfuncs, file_eigenfunc)
    
    # Define normalisation constants.
    # See Ouroboros/docs/Ouroboros_normalisation_notes.pdf.
    if units == 'ouroboros':

        eigfunc_norm = 1.0

        grad_norm = 1.0

        pot_norm = 1.0

    elif units == 'mineos':

        r_n = 6371.0E3 # m
        rho_n = 5515.0 # kg/m3
        G = 6.674E-11 # SI units.

        eigfunc_norm = rho_n*np.sqrt((1.0E-9)*np.pi*G*((r_n)**3.0))

        grad_norm = r_n*1.0E-3

        pot_norm = np.sqrt(r_n/(1.0E3*np.pi*G)) 

    elif units == 'SI':

        eigfunc_norm = np.sqrt(1.0E-9)

        grad_norm = 1.0E-3

        pot_norm = 1.0E3*np.sqrt(1.0E-9)

    else:

        raise ValueError

    if norm_func == 'DT':

        eigfunc_norm = eigfunc_norm*omega
        pot_norm = pot_norm*omega
        k = np.sqrt(l*(l + 1.0))

    else:
        
        assert norm_func == 'mineos', 'Options: mineos, DT'
        #pot_norm = pot_norm/omega

    # Radial case.
    if mode_type == 'R':

        r, U, Up, P, Pp = np.load(path_eigenfunc)
        U[0] = 0.0 # Bug in Ouroboros causes U[0] to be large.

        # Apply normalisation.
        U   = U*eigfunc_norm
        Up  = Up*eigfunc_norm*grad_norm
        P   = P*pot_norm
        Pp  = Pp*pot_norm*grad_norm

        eigenfunc_dict = {'r' : r, 'U' : U, 'Up' : Up, 'P' : P, 'Pp' : Pp}

    # Spheroidal case.
    elif mode_type == 'S':

        r, U, Up, V, Vp, P, Pp = np.load(path_eigenfunc)

        # Apply normalisation.
        U   = U*eigfunc_norm
        V   = V*eigfunc_norm
        Up  = Up*eigfunc_norm*grad_norm
        Vp  = Vp*eigfunc_norm*grad_norm
        P   = P*pot_norm
        Pp  = Pp*pot_norm*grad_norm
        if norm_func == 'DT':

            V   = V*k
            Vp  = Vp*k

        eigenfunc_dict = {'r' : r, 'U' : U, 'V' : V, 'Up' : Up, 'Vp' : Vp,
                            'P' : P, 'Pp' : Pp}
        
    # Toroidal case.
    elif mode_type == 'T':

        r, W = np.load(path_eigenfunc)

        # Apply normalisation.
        W = W*eigfunc_norm
        if norm_func == 'DT':

            W = W*k

        eigenfunc_dict = {'r' : r, 'W' : W}
    
    # Error catching.
    else:

        raise NotImplementedError

    # Convert to m (for consistency with Mineos).
    eigenfunc_dict['r'] = eigenfunc_dict['r']*1.0E3

    return eigenfunc_dict

def get_kernel_dir(dir_output, Ouroboros_info, mode_type):
    
    # Unpack the Ouroboros run information.
    name_model  = Ouroboros_info['model']
    n_layers    = Ouroboros_info['n_layers']
    n_max       = Ouroboros_info['n_lims'][1]
    l_max       = Ouroboros_info['l_lims'][1]
    grav_switch    = Ouroboros_info['grav_switch']
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

def load_kernel(run_info, mode_type, n, l, units = 'standard'):

    # Load kernel.
    _, _, _, dir_out = get_Ouroboros_out_dirs(run_info, mode_type)
    dir_kernels = os.path.join(dir_out, 'kernels')
    name_kernel_file = 'kernels_{:>05d}_{:>05d}.npy'.format(n, l)
    path_kernel = os.path.join(dir_kernels, name_kernel_file)
    kernel_arr = np.load(path_kernel)

    # Unpack the array.
    r, K_ka, K_mu = kernel_arr

    # Load mode frequency (necessary for normalisation).
    mode_info = load_eigenfreq(run_info, mode_type, n_q = n, l_q = l)
    f_mHz = mode_info['f']

    # Get to the right units.
    if units == 'standard':

        # Convert from (Hz 1/Pa 1/m) to (mHz 1/GPa 1/km).
        Hz_to_mHz   = 1.0E3
        Pa_to_GPa   = 1.0E-9
        m_to_km     = 1.0E-3
        #
        scale_ka = Hz_to_mHz/(Pa_to_GPa*m_to_km)
        scale_mu = scale_ka
        #
        K_ka = K_ka*scale_ka
        K_mu = K_mu*scale_mu
        
    elif units == 'SI':

        # The kernels are already in SI units.
        pass

    else:

        print(units)
        raise ValueError

    # For unknown reasons, it is necessary to multiply by
    # omega**2.0 where omega is measured in mHz.
    scale = (f_mHz**2.0)
    K_ka = K_ka*scale
    K_mu = K_mu*scale

    return r, K_ka, K_mu

# Loading data from Mineos. ---------------------------------------------------
def load_eigenfreq_Mineos(run_info, mode_type, n_q = None, l_q = None, n_skip = None):
    
    # Find the Mineos output directory.
    _, run_info['dir_run'] = get_Mineos_out_dirs(run_info)

    if n_skip == None:

        with open(run_info['path_model'], 'r') as in_id:

            in_id.readline()
            in_id.readline()
            n_layers = int(in_id.readline().split()[0])

        n_skip = n_layers + 11
    
    file_eigval = 'minos_bran_out_{:}.txt'.format(mode_type)
    path_eigval = os.path.join(run_info['dir_run'], file_eigval) 

    #n, l, f, Q = np.loadtxt(path_eigval, skiprows = n_skip, usecols = (0, 2, 4, 7)).T
    n, l, c, f, u, Q = np.loadtxt(path_eigval, skiprows = n_skip, usecols = (0, 2, 3, 4, 6, 7)).T

    n = n.astype(np.int)
    l = l.astype(np.int)

    if (n_q is not None) and (l_q is not None):

        i = np.where((n == n_q) & (l == l_q))[0][0]

        f_q = f[i]
        c_q = c[i]
        u_q = u[i]
        Q_q = Q[i]

        mode_info = {'f' : f_q, 'Q' : Q_q, 'c' : c_q, 'u' : u_q}

    else:

        mode_info = {'n' : n, 'l' : l, 'f' : f, 'Q' : Q, 'c' : c, 'u' : u}

    return mode_info

def load_eigenfunc_Mineos(run_info, mode_type, n, l, norm_func = 'mineos', units = 'SI', omega = None):
    '''
    Normalisation
    See Ouroboros/doc/Ouroboros_normalisation_notes.pdf.
    mineos      Use the Mineos/Ouroboros normalisation formula with Mineos units.
    ouroboros   Use the Mineos/Ouroboros normalisation formula with Ouroboros units.
    SI          Use the Mineos/Ouroboros normalisation formula with SI units. 
    DT          Use the Dahlen and Tromp normalisation formula with SI units.
    '''

    # Find the Mineos output directory.
    _, run_info['dir_run'] = get_Mineos_out_dirs(run_info)
    dir_eig_funcs = os.path.join(run_info['dir_run'], 'eigen_txt_{:}'.format(mode_type))

    # Syndat uses a slightly different naming convention. Radial modes are
    # prefixed with 'S' instead of 'R', and inner core toroidal modes are
    # prefixed with 'C'.
    mode_type_to_mode_type_str_dict = {'R' : 'S', 'S' : 'S', 'T' : 'T', 'I' : 'C'}
    mode_type_str = mode_type_to_mode_type_str_dict[mode_type]

    # Get the path of the eigenfunction file.
    file_eig_func = '{:}.{:>07d}.{:>07d}.ASC'.format(mode_type_str, n, l)
    path_eig_func = os.path.join(dir_eig_funcs, file_eig_func)

    # Define normalisation constants.
    if units in ['ouroboros', 'SI']:

        r_n = 6371.0E3 # m
        rho_n = 5515.0 # kg/m3
        G = 6.674E-11 # SI units.

    if units == 'ouroboros':

        En = 1.0/(rho_n*np.sqrt((1.0E-9)*np.pi*G*((r_n)**3.0)))
        Epn = En/(r_n*1.0E-3)
        Pn = np.sqrt((1.0E3*np.pi*G)/r_n) 
        Ppn = Pn/(r_n*1.0E-3)

    elif units == 'mineos':

        En = 1.0
        Epn = 1.0
        Pn = 1.0
        Ppn = 1.0

    elif units == 'SI':

        En = 1.0/(rho_n*np.sqrt(np.pi*G*((r_n)**3.0)))
        Epn = En/r_n
        Pn = np.sqrt((np.pi*G)/r_n) 
        Ppn = Pn/r_n

    if norm_func == 'DT':
        
        assert omega is not None, 'To apply D&T normalisation, the angular frequency (rad per s) must be specified.'
        En = En*omega
        Epn = Epn*omega
        Pn = Pn*omega
        Ppn = Ppn*omega
        k = np.sqrt(l*(l + 1.0))

    # Load the data.
    data = np.loadtxt(path_eig_func, skiprows = 1).T

    # Unpack and return
    if mode_type == 'R':

        r, U, Up = data
        
        # Apply normalisation.
        U  = En*U
        Up = Epn*Up

        eigenfunc_dict = {'r' : r, 'U' : U, 'Up' : Up}

    elif mode_type == 'S':

        r, U, Up, V, Vp, P, Pp  = data

        # Apply normalisation.
        U  = En*U
        Up = Epn*Up
        V  = En*V
        Vp = Epn*Vp
        P  = Pn*P
        Pp = Ppn*Pp
        #
        if norm_func == 'DT':

            V  = V*k
            Vp = Vp*k

        eigenfunc_dict = {'r' : r, 'U' : U, 'Up' : Up, 'V' : V, 'Vp' : Vp,
                            'P' : P, 'Pp' : Pp}

    elif mode_type in ['T', 'I']:

        r, W, Wp = data

        # Apply normalisation.
        W = En*W
        Wp = Epn*Wp
        #
        if norm_func == 'DT':
            
            W  = W*k
            Wp = Wp*k

        eigenfunc_dict = {'r' : r, 'W' : W, 'Wp' : Wp}

    else:

        raise ValueError

    # Reverse the order of the output to conform with Ouroboros.
    for var in eigenfunc_dict:

        eigenfunc_dict[var] = eigenfunc_dict[var][::-1]

    return eigenfunc_dict

# Loading data wrappers. -------------------------------------------------------
def read_input_file(path_input):

    # Announcement
    print('Reading input file, {:}'.format(path_input))

    # Read line by line.
    with open(path_input, 'r') as in_id:

        code = in_id.readline().split()[1]

    if code == 'ouroboros':

        run_info = read_Ouroboros_input_file(path_input)

    elif code == 'mineos':

        run_info = read_Mineos_input_file(path_input)

    else:

        raise ValueError('Code {:} not recognised.'.format(code))

    return run_info

def read_summation_input_file(path_input, code):

    if code == 'mineos':

        summation_info = read_Mineos_summation_input_file(path_input)

    elif code == 'ouroboros':

        summation_info = read_Ouroboros_summation_input_file(path_input)

    else:

        raise ValueError

    return summation_info

def load_eigenfreq(run_info, mode_type, n_q = None, l_q = None, i_toroidal = None):

    if run_info['code'] == 'ouroboros':

        mode_info = load_eigenfreq_Ouroboros(run_info, mode_type, n_q = n_q, l_q = l_q, i_toroidal = i_toroidal)

    elif run_info['code'] == 'mineos':

        mode_info = load_eigenfreq_Mineos(run_info, mode_type, n_q = n_q, l_q = l_q)

    return mode_info

def load_eigenfunc(run_info, mode_type, n, l, i_toroidal = None, norm_args = {'norm_func' : 'mineos', 'units' : 'SI', 'omega' : None}):

    if run_info['code'] == 'ouroboros':

        eigenfunc_dict = load_eigenfunc_Ouroboros(run_info, mode_type, n, l, i_toroidal = i_toroidal, **norm_args)

    elif run_info['code'] == 'mineos':

        eigenfunc_dict = load_eigenfunc_Mineos(run_info, mode_type, n, l, **norm_args)

    else:

        raise ValueError

    return eigenfunc_dict

def align_mode_lists(n_0, l_0, n_1, l_1):

    num_modes_0 = len(n_0)
    #f_1_sorted_by_0 = np.zeros(num_modes_0)
    
    i_align_0 = np.array(list(range(num_modes_0)), dtype = np.int)
    i_align_1 = np.zeros(num_modes_0, dtype = np.int) - 1
    for i in range(num_modes_0):

        j = np.where((l_1 == l_0[i]) & (n_1 == n_0[i]))[0]
        
        assert len(j) < 2
        if len(j) == 1:

            #f_1_sorted_by_0[i] = f_1[j]
            i_align_1[i] = j

        else:

            #f_1_sorted_by_0[i] = np.nan
            i_align_0[i] = -1 

    #i_good = np.where(~np.isnan(f_1_sorted_by_0))[0]
    i_good = np.where(i_align_0 >= 0)[0]
    n = n_0[i_good]
    l = l_0[i_good]
    i_align_0 = i_align_0[i_good]
    i_align_1 = i_align_1[i_good]

    return n, l, i_align_0, i_align_1

def filter_mode_list(mode_info, path_mode_list):
    
    mode_type = 'S'

    n_choose, l_choose = np.loadtxt(path_mode_list, dtype = np.int).T

    i_choose = []
    for i in range(len(n_choose)):

        i_choose.append(np.where((n_choose[i] == mode_info[mode_type]['n']) & (l_choose[i] == mode_info[mode_type]['l']))[0][0])
    
    mode_info_new = dict()
    mode_info_new[mode_type] = dict()
    for key in mode_info[mode_type]:

        mode_info_new[mode_type][key] = mode_info[mode_type][key][i_choose]
    
    return mode_info_new

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

def interp_n_parts(r, r_model, x_model, i_fluid_solid_boundary, i_fluid_solid_boundary_model):
    '''
    Careful interpolation of model parameters, preserving the fluid-solid discontinuities.
    '''
    
    n_parts = len(i_fluid_solid_boundary) + 1
    assert n_parts == (len(i_fluid_solid_boundary_model) + 1)
    
    i_fluid_solid_boundary = list(i_fluid_solid_boundary)
    i_fluid_solid_boundary.insert(0, 0)
    i_fluid_solid_boundary.append(None)

    i_fluid_solid_boundary_model = list(i_fluid_solid_boundary_model)
    i_fluid_solid_boundary_model.insert(0, 0)
    i_fluid_solid_boundary_model.append(None)
    
    x_list = []
    for i in range(n_parts):

        i0 = i_fluid_solid_boundary[i]
        i1 = i_fluid_solid_boundary[i + 1]

        i0_model = i_fluid_solid_boundary_model[i]
        i1_model = i_fluid_solid_boundary_model[i + 1]
        
        x_i = np.interp(r[i0 : i1], r_model[i0_model : i1_model], x_model[i0_model : i1_model])
        x_list.append(x_i)

    x = np.concatenate(x_list)

    return x

def write_model(model, path_out, header_str):

    print("Writing to {:}".format(path_out))

    out_fmt = '{:>7.0f}. {:>8.2f} {:>8.2f} {:>8.2f} {:>8.1f} {:>8.1f} {:>8.2f} {:>8.2f} {:>8.5f}\n'

    with open(path_out, 'w') as out_id:

        out_id.write(header_str + '\n')
        out_id.write('{:>4d} {:>8.5f} {:>3d}\n'.format(0, 1.0/model['f_ref_Hz'], 1))
        out_id.write('{:>6d} {:>3d} {:>3d}\n'.format(model['n_layers'], model['i_icb'],
                        model['i_cmb']))
        
        for i in range(model['n_layers']):

            out_id.write(out_fmt.format(
                model['r'][i], model['rho'][i], model['v_pv'][i], model['v_sv'][i],
                model['Q_ka'][i], model['Q_mu'][i], model['v_ph'][i],
                model['v_sh'][i], model['eta'][i]))

    return

# Manipulating waveform data. -------------------------------------------------
def add_epi_dist_and_azim(inv, cmt, stream):

    # Calculate epicentral distance and azimuth for each station, and
    # assign to trace.
    for trace in stream:
        
        if isinstance(inv, dict):

            station_coords = inv[trace.stats.station]['coords']

        else:

            station_coords = inv.get_coordinates(trace.id)

        try:

            epi_dist_m, az_ev_sta, az_sta_ev = \
                    gps2dist_azimuth(   cmt['lat_hypo'], cmt['lon_hypo'],
                                        station_coords['latitude'], station_coords['longitude'])

        except KeyError:

            epi_dist_m, az_ev_sta, az_sta_ev = \
                    gps2dist_azimuth(   cmt['lat_centroid'], cmt['lon_centroid'],
                                        station_coords['latitude'], station_coords['longitude'])

        station_rel_coords = {  'epi_dist_m' : epi_dist_m,
                                'az_ev_sta' : az_ev_sta,
                                'az_sta_ev' : az_sta_ev}

        trace.stats['rel_coords'] = station_rel_coords

    return stream
