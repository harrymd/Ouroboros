import argparse
import os

import numpy as np

from Ouroboros.common import (get_Ouroboros_out_dirs, 
                        #get_r_fluid_solid_boundary, interp_n_parts,
                        load_eigenfreq_Ouroboros,
                        load_eigenfunc_Ouroboros, mkdir_if_not_exist,
                        #load_model_full,
                        read_Ouroboros_input_file)
from Ouroboros.kernels.kernels import radial_derivative

def gradient_wrapper(run_info, dir_output, mode_type, n, l, **normalisation_args):

    # Load eigenfunction.
    r, U, V = load_eigenfunc_Ouroboros(run_info, mode_type, n, l, **normalisation_args)

    # Calculate derivatives.
    Up = radial_derivative(r, U)
    Vp = radial_derivative(r, V)

    # Save.
    dir_gradient = os.path.join(dir_output, 'gradient')
    file_out = 'grad_{:>05d}_{:>05d}.npy'.format(n, l)
    path_out = os.path.join(dir_gradient, file_out)
    #
    mkdir_if_not_exist(dir_gradient)
    out_arr = np.array([r, Up, Vp])
    print("Saving to {:}".format(path_out))
    np.save(path_out, out_arr)

    return

def main():

    # Read input arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_input_file", help = "File path (relative or absolute) to Ouroboros input file.")
    args = parser.parse_args()

    # Rename input arguments.
    path_input = args.path_to_input_file

    # Read input file.
    run_info = read_Ouroboros_input_file(path_input)

    # Set normalisation of eigenfunctions (and therefore potential).
    # Use Dahlen and Tromp normalisation function ('DT') so that we can
    # use expressions from Dahlen and Tromp without modification.
    # Use Ouroboros units so output is consistent with eigenfunctions.
    normalisation_args = {'norm_func' : 'DT', 'units' : 'ouroboros'}
    #
    for mode_type in ['S']:

        # Get output directory.
        _, _, _, dir_output = \
            get_Ouroboros_out_dirs(run_info, mode_type)

        # Get list of modes.
        mode_info = load_eigenfreq_Ouroboros(run_info, mode_type, i_toroidal = None)
        n = mode_info['n']
        l = mode_info['l']
        f = mode_info['f']

        num_modes = len(n)

        for i in range(num_modes):
        #for i in [5]:

            #print('{:>5d} {:>1} {:>5d}'.format(n[i], mode_type, l[i]))

            # To convert from Ouroboros normalisation to Dahlen and Tromp
            # normalisation we must provide the mode frequency in
            # rad per s.
            f_rad_per_s = f[i]*1.0E-3*2.0*np.pi
            if normalisation_args['norm_func'] == 'DT':
                normalisation_args['omega'] = f_rad_per_s

            if mode_type == 'R':

                raise NotImplementedError

            elif mode_type == 'S': 

                gradient_wrapper(run_info, dir_output, mode_type, n[i], l[i], **normalisation_args)

            elif mode_type == 'T':

                raise NotImplementedError

            else:

                raise ValueError
    
    return

if __name__ == '__main__':

    main()
