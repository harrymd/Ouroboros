import os
from shutil import copyfile
import time

from common import read_RadialPNM_input_file, mkdir_if_not_exist
from compute_modes import radial_modes, spheroidal_modes, toroidal_modes

def main():
    '''
    Control script for running RadialPNM.
    Run from the command line using
    python3 main.py
    in the directory containing this script.
    '''
    
    print('RadialPNM')
    start_time = time.time()

    # Read the input file.
    RadialPNM_input_file = 'input_RadialPNM.txt'
    path_model, dir_output, g_switch, mode_type, l_min, l_max, n_min, n_max, n_layers = \
        read_RadialPNM_input_file()
    
    # Get the name of the model from the file.
    name_model = os.path.splitext(os.path.basename(path_model))[0]

    # Store n- and l- limits as arrays.
    n_lims      = [n_min, n_max]
    l_lims      = [l_min, l_max]
    
    # Set the 'g_switch' string: 0 -> noG, 1 -> G, 2 -> GP.
    g_switch_strs = ['noGP', 'G', 'GP']
    g_switch_str = g_switch_strs[g_switch]

    # Set the 'switch' string, e.g. 'S_GP'.
    if mode_type == 'T':

        switch = None 

    else:

        switch = '{:}_{:}'.format(mode_type, g_switch_str)
    
    # Safety checks.
    if mode_type == 'T':

        assert g_switch == 0, 'For consistency in file naming, if mode_type == \'T\' (toroidal modes), it is required that g_switch == 0 (although the g_switch variable does not affect the calculation of toroidal modes).'

    # Create model directory if it doesn't exist.
    name_model_with_layers = '{:}_{:>05d}'.format(name_model, n_layers)
    dir_model = os.path.join(dir_output, name_model_with_layers)
    #path_model = os.path.join(dir_model_files, '{:}.txt'.format(name_model))
    mkdir_if_not_exist(dir_model)

    # Create the run directory if it doesn't exist.
    name_run = '{:>05d}_{:>05d}_{:1d}'.format(n_max, l_max, g_switch)
    dir_run = os.path.join(dir_model, name_run)
    mkdir_if_not_exist(dir_run)

    # Create the type directory if it doesn't exist.
    dir_type = os.path.join(dir_run, mode_type)
    mkdir_if_not_exist(dir_type)

    # Copy the input file to the output directory.
    copyfile(RadialPNM_input_file, os.path.join(dir_type, RadialPNM_input_file))

    # Run the code.
    if mode_type == 'T':

        toroidal_modes(path_model, dir_type, l_min, l_max, n_min, n_max, n_layers)
    
    elif mode_type == 'R':
        
        radial_modes(path_model, dir_type, n_min, n_max, n_layers, switch)

    elif mode_type == 'S':

        spheroidal_modes(path_model, dir_type, l_min, l_max, n_min, n_max, n_layers, switch)

    else:

        raise ValueError

    elapsed_time = time.time() - start_time
    print('Total time used: {:>.3f} s'.format(elapsed_time))

    return

if __name__ == '__main__':

    main()
