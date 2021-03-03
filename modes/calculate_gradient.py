import argparse
import os

import numpy as np

from Ouroboros.common import (get_Ouroboros_out_dirs, 
                        load_eigenfreq_Ouroboros,
                        load_eigenfunc_Ouroboros, mkdir_if_not_exist,
                        read_Ouroboros_input_file)

def get_indices_of_discontinuities(r):
    
    i_discon = []
    n_r = len(r)
    for i in range(n_r - 1):

        if r[i] == r[i + 1]:
        
            i_discon.append(i)
            
    return i_discon
        
def radial_derivative(r, x):
    '''
    Numerical differentiation of quantity with respect to radius.
    '''
    
    n_r = len(r)
    i_d = get_indices_of_discontinuities(r)
    i_d.append(n_r)
    n_d = len(i_d)

    dxdr = np.zeros(n_r)
    
    i1 = 0
    for i2 in i_d:
        
        # Note: 'Distances must be scalars' error if NumPy version too low.
        dxdr[i1 : i2 + 1] = np.gradient(x[i1 : i2 + 1], r[i1 : i2 + 1])
        i1 = i2 + 1

    return dxdr

def gradient_wrapper(run_info, dir_output, mode_type, n, l, **normalisation_args):

    # Load eigenfunction.
    eigfunc_info = load_eigenfunc_Ouroboros(run_info, mode_type, n, l, **normalisation_args)
    r = eigfunc_info['r']
    r = r*1.0E-3 # Units of km.
    U = eigfunc_info['U']
    V = eigfunc_info['V']
    P = eigfunc_info['P']

    # Calculate derivatives.
    Up = radial_derivative(r, U)
    Vp = radial_derivative(r, V)
    Pp = radial_derivative(r, P)

    # Save.
    #dir_gradient = os.path.join(dir_output, 'gradient')
    #file_out = 'grad_{:>05d}_{:>05d}.npy'.format(n, l)
    #path_out = os.path.join(dir_gradient, file_out)
    #
    #mkdir_if_not_exist(dir_gradient)
    file_eigenfunc = '{:>05d}_{:>05d}.npy'.format(n, l)
    path_out = os.path.join(dir_output, 'eigenfunctions', file_eigenfunc)
    out_arr = np.array([r, U, V, Up, Vp, P, Pp])
    print("Saving to {:}".format(path_out))
    np.save(path_out, out_arr)

    return

def gradient_all_modes(run_info, mode_type):

    # Calculate eigenfunction gradients before attenuation re-labelling.
    run_info['use_attenuation'] = False

    # Set normalisation of eigenfunctions (and therefore potential).
    # Use Mineos normalisation and Ouroboros units so output is consistent
    # with eigenfunctions.
    #normalisation_args = {'norm_func' : 'DT', 'units' : 'ouroboros'}
    normalisation_args = {'norm_func' : 'mineos', 'units' : 'ouroboros'}
    #
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

        #print('{:>5d} {:>1} {:>5d}'.format(n[i], mode_type, l[i]))

        # To convert from Ouroboros normalisation to Dahlen and Tromp
        # normalisation we must provide the mode frequency in
        # rad per s.
        f_rad_per_s = f[i]*1.0E-3*2.0*np.pi
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

def main():

    # Read input arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_input_file", help = "File path (relative or absolute) to Ouroboros input file.")
    args = parser.parse_args()

    # Rename input arguments.
    path_input = args.path_to_input_file

    # Read input file.
    run_info = read_Ouroboros_input_file(path_input)
    
    # Loop over modes.
    for mode_type in run_info['mode_types']:

        gradient_all_modes(run_info, mode_type)

    return

if __name__ == '__main__':

    main()
