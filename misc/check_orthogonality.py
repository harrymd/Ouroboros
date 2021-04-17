import argparse
import itertools
import os

import numpy as np

from Ouroboros.common import (filter_mode_list, get_r_fluid_solid_boundary, 
                        interp_n_parts,
                        load_eigenfreq, load_eigenfunc, load_model, read_input_file)

def get_eigfunc_product(mode_type, r, rho, eigfunc_dict_A, eigfunc_dict_B, i_fluid_solid_boundary = None):

    if mode_type == 'S':
        
        assert i_fluid_solid_boundary is not None
        product = get_eigfunc_product_S(r, rho, eigfunc_dict_A, eigfunc_dict_B, i_fluid_solid_boundary)

    else:

        raise NotImplementedError

    return product

def get_eigfunc_product_S(r, rho, eigfunc_dict_A, eigfunc_dict_B, i_fluid_solid_boundary):

    # Unpack variables.
    U_A = eigfunc_dict_A['U']
    V_A = eigfunc_dict_A['V']
    #
    U_B = eigfunc_dict_B['U']
    V_B = eigfunc_dict_B['V']

    # Define function to be integrated. 
    function = rho * (r**2.0) * ((U_A * U_B) + (V_A * V_B))

    # Evaluate integral, treating each section separated.
    i = np.array([0, *i_fluid_solid_boundary, len(r)], dtype = np.int)
    n_segment = len(i) - 1
    #
    integral = 0.0
    for j in range(n_segment):
        
        i1 = i[j]
        i2 = i[j + 1]

        integral = integral + np.trapz(function[i1 : i2], x = r[i1 : i2])

    return integral

def main():

    # Read input arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_input_file", help = "File path (relative or absolute) to Ouroboros input file.")
    parser.add_argument("mode_type", choices = ['R', 'S', 'T'], help = 'Mode type (radial, spheroidal or toroidal).')
    args = parser.parse_args()

    # Rename input arguments.
    path_input  = args.path_to_input_file
    mode_type   = args.mode_type

    # Read the input file and command-line arguments.
    run_info = read_input_file(path_input)

    # Load the planetary model.
    model = load_model(run_info['path_model'])

    # Find the fluid-solid boundary points in the model.
    i_fluid_model, r_solid_fluid_boundary_model, i_fluid_solid_boundary_model =\
        get_r_fluid_solid_boundary(model['r'], model['v_s'])

    # Set normalisation arguments.
    norm_func = 'DT'
    eig_norm_units = 'SI'
    load_eigfunc_norm_args = {'norm_func' : norm_func, 'units' : eig_norm_units}

    # Load frequency information.
    mode_info = load_eigenfreq(run_info, mode_type)

    # Filter the mode list.
    mode_info = filter_mode_list({mode_type : mode_info}, f_lims = [0.0, 7.0])[mode_type]

    # Count the number of modes and make a list of pairs to compare.
    num_modes = len(mode_info['n'])
    pairs = itertools.combinations_with_replacement(list(range(num_modes)), 2)
    pairs = list(pairs)
    #num_pairs = (num_modes + 1)*num_modes//2
    num_pairs = len(pairs)

    # Prepare output array.
    product = np.zeros(num_pairs)

    # Loop over all the pairs.
    j_prev = -1
    k_prev = -1
    first_loop = True
    #for i, (j, k) in enumerate(pairs):
    for i in range(num_pairs):

        j, k = pairs[i]

        # Load first eigenfunction.
        if j != j_prev:

            f_j = mode_info['f'][j]
            load_eigfunc_norm_args['omega'] = f_j*1.0E-3*2.0*np.pi
            eigfunc_dict_j = load_eigenfunc(run_info, mode_type,
                                mode_info['n'][j],
                                mode_info['l'][j],
                                norm_args = load_eigfunc_norm_args)

        # Load second eigenfunction.
        if k != k_prev:

            f_k = mode_info['f'][k]
            load_eigfunc_norm_args['omega'] = f_k*1.0E-3*2.0*np.pi
            eigfunc_dict_k = load_eigenfunc(run_info, mode_type,
                                mode_info['n'][k],
                                mode_info['l'][k],
                                norm_args = load_eigfunc_norm_args)

        # Get density profile.
        if first_loop:

            # Get radial coordinate.
            r = eigfunc_dict_j['r']

            # Find indices of solid-fluid boundaries in the eigenfunction grid.
            i_fluid_solid_boundary = (np.where(np.diff(r) == 0.0))[0] + 1

            if run_info['code'] == 'mineos':

                #print("Warning: Assuming inner two discontinuities are CMB and ICB.")
                i_fluid_solid_boundary = i_fluid_solid_boundary[0:2]

            # Interpolate the density model.
            rho  = interp_n_parts(r, model['r'], model['rho'], i_fluid_solid_boundary, i_fluid_solid_boundary_model)

            # Turn off first loop.
            first_loop = False

        # Evaluate scalar product.
        print('{:>5d} of {:>5d}: scalar product of {:>3d} {:} {:>3d} with {:>3d} {:} {:>3d}:'.format(
                i + 1, num_pairs,
                mode_info['n'][j], mode_type, mode_info['l'][j],
                mode_info['n'][k], mode_type, mode_info['l'][k]), end = '')
        product[i] = get_eigfunc_product(mode_type, r, rho, eigfunc_dict_j, eigfunc_dict_k,
                        i_fluid_solid_boundary = i_fluid_solid_boundary)
        #print(' {:>+9.3e}'.format(product[i]))
        print(' {:>+6.3f}'.format(product[i]))

        # Prepare for next loop.
        j_prev = j
        k_prev = k
    
    # Writing output.
    out_fmt = '{:>5d} {:>5d} {:>5d} {:>5d} {:>+19.12e}\n'
    name_out = 'test_orthonormality.txt'
    path_out = os.path.join(run_info['dir_run'], name_out)
    print('Writing to {:}'.format(path_out))
    with open(path_out, 'w') as out_id:
        
        for i in range(num_pairs):
            
            j, k = pairs[i]
            out_id.write(out_fmt.format(
                mode_info['n'][j], mode_info['l'][j],
                mode_info['n'][k], mode_info['l'][k],
                product[i]))

    return

if __name__ == '__main__':

    main()
