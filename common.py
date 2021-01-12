import os

import numpy as np

def mkdir_if_not_exist(dir_):
    '''
    Create a directory if it does not already exist.
    '''

    if not os.path.isdir(dir_):
        
        print('Making directory {:}'.format(dir_))
        os.mkdir(dir_)

    return

def read_RadialPNM_input_file():
    '''
    Reads the RadialPNM input file.
    '''

    input_file = 'input_RadialPNM.txt'
    print('Reading input file, {:}'.format(input_file))
    
    # Read line by line.
    with open(input_file, 'r') as in_id:
        
        path_model      = in_id.readline().strip()
        dir_output      = in_id.readline().strip()
        g_switch        = int(in_id.readline())
        mode_type       = in_id.readline().strip()
        l_min, l_max    = [int(x) for x in in_id.readline().split()]
        n_min, n_max    = [int(x) for x in in_id.readline().split()]
        n_layers        = int(in_id.readline())

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
    print('Number of layers: {:d}'.format(n_layers))

    return path_model, dir_output, g_switch, mode_type, l_min, l_max, n_min, n_max, n_layers

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

    # Load the model data.
    try:

        model_data = np.loadtxt(model_path, skiprows = 0) 

    except:

        model_data = np.loadtxt(model_path, skiprows = skiprows) 

    shape = np.shape(model_data)
    # radius:million meters, eigen-frequency:mHz
    radius = model_data[:,0]/10**6
    # density:g/cm^3
    rho = model_data[:,1]/10**3
    # average velocity: km/s
    vp = (model_data[:,2]+model_data[:,6])/2/10**3
    vs = (model_data[:,3]+model_data[:,7])/2/10**3
    
    mu = rho*np.power(vs,2) #shear modulus
    ka = rho*np.power(vp,2)-4/3*rho*np.power(vs,2) #bulk modulus

    return model_data, shape, radius, rho, vp, vs, mu, ka
