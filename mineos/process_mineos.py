import os

import numpy as np
import obspy

def read_minos_bran_output(out_file):

    print(out_file)
    
    with open(out_file, 'r') as in_ID:
        
        n_header = 5
        for i in range(n_header):
            
            in_ID.readline()
            
        n_layer = -1
        reading_input_model = True
        while reading_input_model:
            
            line = in_ID.readline().split()
            if len(line) == 0:
                
                reading_input_model = False
                
            n_layer = n_layer + 1
            
    n_row_skip = 11 + n_layer
    #
    dt = np.dtype({     'names' : ['n', 'l', 'c', 'w', 'u', 'Q', 'e'],
                        'formats': ['i', 'i', 'f', 'f', 'f', 'f', 'f']})
    #

    
    
    mode_data = np.loadtxt(out_file,
                            dtype       = dt,
                            skiprows    = n_row_skip,
                            usecols     = (0, 2, 3, 4, 6, 7, 8))
                                
    return mode_data
    
def read_eigen2asc_output(jcom, n, l, eigen_dir, cols):
    '''
    Input
    
    jcom        2 for toroidal modes, S for spheroidal modes.
    n           Mode number.
    l           Angular order
    eigen_dir   Directory containing output of eigen2asc.
    cols        A list of which columns to return. The order of the
                list will control the order of columns the output
                array.
                The options are
                    r   Radial distance.
                For spheroidal modes, the options are:
                    U   Vertical component;
                    V   Horizontal component;
                    P   Gravitational potential;
                    Ud, Vd, Pd  Radial derivatives of the above.
                For toroidal modes, the options are:
                    W   Toroidal component;
                    Wd  Radial derivative of the above.
    '''
    
    cols_dict = {   'T' : ['r', 'W', 'Wd'],
                    'S' : ['r', 'U', 'Ud', 'V', 'Vd', 'P', 'Pd']}
    jcom_to_mode_type = {2 : 'T', 3 : 'S'}
    
    mode_type = jcom_to_mode_type[jcom]
    
    for col in cols:
        
        if col not in cols_dict[mode_type]:
            
            error_message = (
                'Output {} is not allowed for mode of type {}.\n'
                'Allowed outputs are:')
            
            for col_allowed in list(cols_dict[mode_type].keys()):
                
                error_message = error_message + ' ' + col_allowed
                
            raise ValueError(error_message)
            
    cols_to_col_nums = {'r'     : 0,
                        'W'     : 1,
                        'Wd'    : 2,
                        'U'     : 1,
                        'Ud'    : 2,
                        'V'     : 3,
                        'Vd'    : 4,
                        'P'     : 5,
                        'Pd'    : 6}
    
    col_nums = []
    for col in cols:
        
        col_nums.append(cols_to_col_nums[col])
    
    eigen_file = '{}.{:07d}.{:07d}.ASC'.format(mode_type, n, l)
    
    eigen_path = os.path.join(eigen_dir, eigen_file)
    
    eigen_data = np.loadtxt(eigen_path, skiprows = 1, usecols = col_nums)
    
    return eigen_data
    
def read_sac_output(dir_output, station):
    
    from glob import glob
    
    import obspy
    
    #dir_sac         = os.path.join(dir_project, 'sac')
    wildcard        = '*{}*'.format(station)
    path_wildcard   = os.path.join(dir_output, wildcard)
    
    paths   = glob(path_wildcard)
    
    #components = ['LHZ', 'LHE', 'LHN']
    components = ['LHZ']

    stream = obspy.core.stream.Stream()
    for component in components:
        
        for path in paths:
            
            if component in path:
                
                trace = obspy.core.stream.read(path)[0]
                stream.append(trace)

    return stream

# def read_sac_output_and_get_comparison_data(dir_project, station):
#
#     import  obspy
#
#     from    stoneley.code.download_waveforms import download_comparison_waveforms
#
#

def main():
    
    pass
    
if __name__ == '__main__':
    
    main()
