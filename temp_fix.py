import argparse
import os

import numpy as np

from Ouroboros.common import read_Ouroboros_input_file, load_eigenfreq_Ouroboros, get_Ouroboros_out_dirs

def fix_wrapper(run_info, dir_output, mode_type, n, l, **normalisation_args):
    
    dir_eigenfunc = os.path.join(dir_output, 'eigenfunctions')
    file_eigenfunc = '{:>05d}_{:>05d}.npy'.format(n, l)
    path_eigenfunc = os.path.join(dir_eigenfunc, file_eigenfunc)

    r, U, V, Up, Vp, P = np.load(path_eigenfunc)
    Pp = np.zeros(U.shape)

    out_arr = np.array([r, U, V, Up, Vp, P, Pp])

    np.save(path_eigenfunc, out_arr)

    return

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_input_file", help = "File path (relative or absolute) to Ouroboros input file.")
    args = parser.parse_args()

    # Rename input arguments.
    path_input = args.path_to_input_file

    # Read input file.
    run_info = read_Ouroboros_input_file(path_input)

    # Calculate eigenfunction gradients before attenuation re-labelling.
    run_info['use_attenuation'] = False

    # Set normalisation of eigenfunctions (and therefore potential).
    # Use Mineos normalisation and Ouroboros units so output is consistent
    # with eigenfunctions.
    #normalisation_args = {'norm_func' : 'DT', 'units' : 'ouroboros'}
    normalisation_args = {'norm_func' : 'mineos', 'units' : 'ouroboros'}
    #
    for mode_type in ['S']:

        # Get output directory.
        _, _, _, dir_output = \
            get_Ouroboros_out_dirs(run_info, mode_type)

        # Get list of modes.
        mode_info = load_eigenfreq_Ouroboros(run_info, mode_type, i_toroidal = None)
        n = mode_info['n']
        l = mode_info['l']

        num_modes = len(n)

        for i in range(num_modes):

            fix_wrapper(run_info, dir_output, mode_type, n[i], l[i], **normalisation_args)

    return

if __name__ == '__main__':

    main()
