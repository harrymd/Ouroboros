'''
Count modes in a given frequency range.
'''

import argparse

import numpy as np

from Ouroboros.common import (get_r_fluid_solid_boundary, load_eigenfreq,
        load_model, read_input_file)

def main():

    # Parse input arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_input_file", help = "File path (relative or absolute) to Ouroboros input file.")
    parser.add_argument("freq_lower", type = float, help = "Lower frequency bound (mHz).")
    parser.add_argument("freq_upper", type = float, help = "Upper frequency bound (mHz).")
    args = parser.parse_args()

    # Rename input arguments.
    path_input = args.path_to_input_file
    freq_lower = args.freq_lower
    freq_upper = args.freq_upper

    print('Searching for modes in the frequency range {:>10.6f} to {:>10.6f} mHz.'.format(freq_lower, freq_upper))

    # Read the input file.
    run_info = read_input_file(path_input)

    # Find the number of solid regions.
    model = load_model(run_info['path_model'])
    i_fluid, r_solid_fluid_boundary, _ = get_r_fluid_solid_boundary(model['r'], model['v_s'])
    n_solid_regions = len(r_solid_fluid_boundary)

    # Load mode information for radial and spheroidal modes.
    mode_info = dict()
    if run_info['code'] == 'mineos':

        for mode_type in ['R', 'S', 'T', 'I']:

            # Load frequencies of modes.
            mode_info[mode_type] = load_eigenfreq(run_info, mode_type)

    elif run_info['code'] == 'ouroboros':

        for mode_type in ['R', 'S', 'T']:

            if mode_type == 'T':

                for i in range(n_solid_regions):

                    mode_str = '{:}{:>1d}'.format(mode_type, i)

                    # Load frequencies of modes.
                    mode_info[mode_str] = load_eigenfreq(run_info,
                                            mode_type, i_toroidal = i)

            else:

                mode_info[mode_type] = load_eigenfreq(run_info, mode_type)

    # Count modes.
    sum_n_modes = 0
    sum_multiplicities = 0

    print('\n')
    for mode_key in mode_info:
        
        cond =  ( (mode_info[mode_key]['f'] > freq_lower)
                & (mode_info[mode_key]['f'] < freq_upper))
        i = np.where(cond)[0]
    
        multiplicity = ((2*mode_info[mode_key]['l']) + 1)
        n_modes = len(i)
        multiplicities = np.sum(multiplicity[i]) 
        
        if n_modes > 0:

            maximum_l = np.max(mode_info[mode_key]['l'][i])
            print('Mode type {:>2}: {:>5d} modes (multiplicity: {:>5d}), max. l: {:>5d}'.format(mode_key, n_modes, multiplicities, maximum_l))

        else:

            print('Mode type {:>2}: {:>5d} modes (multiplicity: {:>5d})'.format(mode_key, n_modes, multiplicities))

        sum_n_modes = sum_n_modes + n_modes
        sum_multiplicities = sum_multiplicities + multiplicities

    print('Totals:       {:>5d} modes (multiplicity {:>5d})'.format(sum_n_modes, sum_multiplicities))

    return

if __name__ == '__main__':

    main()
