'''
Calculates gradient of eigenfunctions.
Uses the np.gradient function (second-order central differences at interior
points, first-order differences at the boundaries).
'''

import argparse
import os

import numpy as np

from Ouroboros.common import (get_n_solid_layers, get_Ouroboros_out_dirs, 
                        load_eigenfreq_Ouroboros,
                        load_eigenfunc_Ouroboros, load_model, mkdir_if_not_exist,
                        read_Ouroboros_input_file)

def get_indices_of_discontinuities(r):
    '''
    Finds indices of discontinuities, where two consecutive values of the
    radial coordinate are the same.
    '''
    
    # Loop over points.
    i_discon = []
    n_r = len(r)
    for i in range(n_r - 1):

        # Check if radius is the same.
        if r[i] == r[i + 1]:
        
            i_discon.append(i)
            
    return i_discon
        
def radial_derivative(r, x):
    '''
    Numerical differentiation of quantity with respect to radius.
    Takes into account radial discontinuities.
    '''
    
    # Find number of points and indices of discontinuities (including final
    # point) and number of discontinuities.
    n_r = len(r)
    i_d = get_indices_of_discontinuities(r)
    i_d.append(n_r)
    n_d = len(i_d)

    # Prepare output array.
    dxdr = np.zeros(n_r)
    
    # Differentiate each section separately.
    i1 = 0
    for i2 in i_d:
        
        # Note: 'Distances must be scalars' error if NumPy version too low.
        dxdr[i1 : i2 + 1] = np.gradient(x[i1 : i2 + 1], r[i1 : i2 + 1])
        i1 = i2 + 1

    return dxdr

def gradient_wrapper_R(run_info, dir_output, n, l, **normalisation_args):
    '''
    Calculate gradients for a single radial mode.
    '''

    mode_type = 'R'

    # Load eigenfunction.
    eigfunc_info = load_eigenfunc_Ouroboros(run_info, mode_type, n, l, **normalisation_args)
    r = eigfunc_info['r']
    r = r*1.0E-3 # Units of km.
    U = eigfunc_info['U']
    P = eigfunc_info['P']

    # Calculate derivatives.
    Up = radial_derivative(r, U)
    Pp = radial_derivative(r, P)

    # Save.
    file_eigenfunc = '{:>05d}_{:>05d}.npy'.format(n, l)
    path_out = os.path.join(dir_output, 'eigenfunctions', file_eigenfunc)
    out_arr = np.array([r, U, Up, P, Pp])
    print("Saving to {:}".format(path_out))
    np.save(path_out, out_arr)

    return

def gradient_wrapper_S(run_info, dir_output, n, l, **normalisation_args):
    '''
    Calculate gradients for a single spheroidal mode.
    '''

    mode_type = 'S'

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
    file_eigenfunc = '{:>05d}_{:>05d}.npy'.format(n, l)
    path_out = os.path.join(dir_output, 'eigenfunctions', file_eigenfunc)
    out_arr = np.array([r, U, Up, V, Vp, P, Pp])
    print("Saving to {:}".format(path_out))
    np.save(path_out, out_arr)

    return

def gradient_wrapper_T(run_info, dir_output, n, l, i_toroidal, **normalisation_args):
    '''
    Calculate gradients for a single toroidal mode.
    '''

    mode_type = 'T'

    # Load eigenfunction.
    eigfunc_info = load_eigenfunc_Ouroboros(run_info, mode_type, n, l, i_toroidal = i_toroidal, **normalisation_args)
    r = eigfunc_info['r']
    r = r*1.0E-3 # Units of km.
    W = eigfunc_info['W']

    # Calculate derivatives.
    Wp = radial_derivative(r, W)

    # Save.
    file_eigenfunc = '{:>05d}_{:>05d}.npy'.format(n, l)
    path_out = os.path.join(dir_output, 'eigenfunctions_{:>03d}'.format(i_toroidal),
                file_eigenfunc)
    out_arr = np.array([r, W, Wp])
    print("Saving to {:}".format(path_out))
    np.save(path_out, out_arr)

    return

def gradient_all_modes_R_or_S(run_info, mode_type, j_skip = None):
    '''
    Calculate gradient for all modes of a given type (spheroidal or radial).
    '''

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

    # Loop over all modes.
    num_modes = len(n)
    for i in range(num_modes):

        if (j_skip is None) or (not (i in j_skip)):

            #print('{:>5d} {:>1} {:>5d}'.format(n[i], mode_type, l[i]))

            # To convert from Ouroboros normalisation to Dahlen and Tromp
            # normalisation we must provide the mode frequency in
            # rad per s.
            f_rad_per_s = f[i]*1.0E-3*2.0*np.pi
            normalisation_args['omega'] = f_rad_per_s

            if mode_type == 'R':

                gradient_wrapper_R(run_info, dir_output, n[i], l[i], **normalisation_args)

            elif mode_type == 'S': 

                gradient_wrapper_S(run_info, dir_output, n[i], l[i], **normalisation_args)

    return

def gradient_all_modes_T(run_info, j_skip = None):
    '''
    Calculate gradient for all toroidal modes.
    Note that an additional loop is required to include all of the solid
    layers.
    '''

    mode_type = 'T'

    # Set normalisation of eigenfunctions (and therefore potential).
    # Use Mineos normalisation and Ouroboros units so output is consistent
    # with eigenfunctions.
    #normalisation_args = {'norm_func' : 'DT', 'units' : 'ouroboros'}
    normalisation_args = {'norm_func' : 'mineos', 'units' : 'ouroboros'}
    #
    # Get output directory.
    _, _, _, dir_output = \
        get_Ouroboros_out_dirs(run_info, mode_type)

    # Determine how many layers there are.
    model = load_model(run_info['path_model'])
    n_solid_layers = get_n_solid_layers(model)

    for i in range(n_solid_layers):

        # Get list of modes.
        mode_info = load_eigenfreq_Ouroboros(run_info, mode_type, i_toroidal = i)
        n = mode_info['n']
        l = mode_info['l']
        f = mode_info['f']

        num_modes = len(n)

        for j in range(num_modes):

            if (j_skip is None) or (not (j in j_skip)):

                #print('{:>5d} {:>1} {:>5d}'.format(n[j], mode_type, l[j]))

                # To convert from Ouroboros normalisation to Dahlen and Tromp
                # normalisation we must provide the mode frequency in
                # rad per s.
                f_rad_per_s = f[j]*1.0E-3*2.0*np.pi
                normalisation_args['omega'] = f_rad_per_s

                gradient_wrapper_T(run_info, dir_output, n[j], l[j], i, **normalisation_args)
    
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
