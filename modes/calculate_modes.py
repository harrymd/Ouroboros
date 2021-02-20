import os
import argparse
from shutil import copyfile
import time

from common import get_Ouroboros_out_dirs, read_Ouroboros_input_file, mkdir_if_not_exist
from compute_modes import radial_modes, spheroidal_modes, toroidal_modes

def main():
    '''
    Control script for running RadialPNM.
    Run from the command line using
    python3 main.py
    in the directory containing this script.
    '''
    
    # Make announcement.
    print('Ouroboros')

    # Parse input arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_input_file", help = "File path (relative or absolute) to Ouroboros input file.")
    input_args = parser.parse_args()
    Ouroboros_input_file = input_args.path_to_input_file
    name_input = os.path.splitext(os.path.basename(Ouroboros_input_file))[0]

    # Read the input file.
    Ouroboros_info = read_Ouroboros_input_file(Ouroboros_input_file)

    # Get the name of the model from the file.
    #name_model = os.path.splitext(os.path.basename(Ouroboros_info['path_model']))[0]

    ## Create model directory if it doesn't exist.
    #name_model_with_layers = '{:}_{:>05d}'.format(name_model, Ouroboros_info['n_layers'])
    #dir_model = os.path.join(Ouroboros_info['dir_output'], name_model_with_layers)
    #mkdir_if_not_exist(dir_model)

    ## Create the run directory if it doesn't exist.
    #name_run = '{:>05d}_{:>05d}'.format(Ouroboros_info['n_lims'][1],
    #            Ouroboros_info['l_lims'][1])
    #dir_run = os.path.join(dir_model, name_run)
    #mkdir_if_not_exist(dir_run)
    
    # Set the 'g_switch' string: 0 -> noG, 1 -> G, 2 -> GP.
    g_switch_strs = ['noGP', 'G', 'GP']
    g_switch_str = g_switch_strs[Ouroboros_info['grav_switch']]

    Ouroboros_info['dirs_type'] = dict()

    for mode_type in Ouroboros_info['mode_types']:
        
        # Start timer.
        start_time = time.time()

        # Set the 'switch' string, e.g. 'S_GP'.
        if mode_type == 'T':

            switch = None 

        else:

            switch = '{:}_{:}'.format(mode_type, g_switch_str)

        Ouroboros_info['switch'] = switch

        dir_model, dir_run, dir_g, dir_type = get_Ouroboros_out_dirs(Ouroboros_info, mode_type)
        for dir_ in [dir_model, dir_run, dir_g, dir_type]:
            
            if dir_ is not None:
                
                mkdir_if_not_exist(dir_)

        ## Create the type directory if it doesn't exist.
        #if mode_type in ['R', 'S']:

        #    dir_g = os.path.join(dir_run, 'grav_{:1d}'.format(Ouroboros_info['g_switch']))
        #    mkdir_if_not_exist(dir_g)
        #    dir_type = os.path.join(dir_g, mode_type)

        #else:

        #    dir_type = os.path.join(dir_run, mode_type)

        #mkdir_if_not_exist(dir_type)
        Ouroboros_info['dirs_type'][mode_type] = dir_type

        # Copy the input file to the output directory.
        copyfile(Ouroboros_input_file, os.path.join(dir_type, name_input))

        # Run the code.
        if mode_type == 'T':

            toroidal_modes(Ouroboros_info)
        
        elif mode_type == 'R':
            
            radial_modes(Ouroboros_info)

        elif mode_type == 'S':

            spheroidal_modes(Ouroboros_info)

        else:

            raise ValueError

        elapsed_time = time.time() - start_time
        path_time = os.path.join(dir_type, 'time.txt')
        print('Time used: {:>.3f} s, saving time info to {:}'.format(elapsed_time, path_time))
        with open(path_time, 'w') as out_id:

            out_id.write('{:>.3f} seconds'.format(elapsed_time))

    return

if __name__ == '__main__':

    main()
