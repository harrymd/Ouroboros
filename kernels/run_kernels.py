'''
Calculates sensitivity kernels for normal modes.
'''

import argparse
import os

import numpy as np

from Ouroboros.common import (get_Ouroboros_out_dirs, get_r_fluid_solid_boundary,
                        get_path_adjusted_model,
                        interp_n_parts, load_eigenfreq, load_eigenfunc,
                        mkdir_if_not_exist, load_model, read_input_file)
from Ouroboros.kernels.kernels import get_kernels_T, get_kernels_S, gravitational_acceleration

def kernels_wrapper(run_info, mode_type, j_skip = None):
    '''
    Controls wrapper scripts for calculating all kernels.
    '''

    # Get path of model file.
    if run_info['attenuation'] == 'linear':

        model_path = get_path_adjusted_model(run_info)

    else:
        
        model_path = run_info['path_model']

    # Load the planetary model.
    model = load_model(model_path)

    # Calculate spheroidal or radial kernels.
    if mode_type in ['R', 'S']:

        kernels_wrapper_R_or_S(run_info, model, mode_type, j_skip = j_skip)

    # Calculate toroidal kernels.
    elif mode_type == 'T':

        kernels_wrapper_T(run_info, model, j_skip = j_skip)

    else:

        raise ValueError

    return

def kernels_wrapper_R_or_S(run_info, model, mode_type, j_skip = None):
    '''
    Calculate kernels for all radial or spheroidal modes.
    '''

    # Get a list of all the modes calculated by Ouroboros.
    mode_info = load_eigenfreq(run_info, mode_type) 
    n_list = mode_info['n']
    l_list = mode_info['l']
    f_mHz_list = mode_info['f']
    f_Hz_list = f_mHz_list*1.0E-3
    f_rad_per_s_list = f_Hz_list*2.0*np.pi
    num_modes = len(n_list)
    
    # Find the fluid-solid boundary points in the model.
    i_fluid_model, r_solid_fluid_boundary_model, i_fluid_solid_boundary_model =\
        get_r_fluid_solid_boundary(model['r'], model['v_s'])

    # Define normalisation arguments.
    norm_args = {'norm_func' : 'DT', 'units' : 'SI'}

    # Loop over modes.
    for i in range(num_modes):

        # Give shorter names to current n, l and f.
        n = n_list[i]
        l = l_list[i]
        omega_rad_per_s = f_rad_per_s_list[i]
        
        # To load eigenfunctions, we need the frequency for normalisation.
        norm_args['omega'] = omega_rad_per_s

        # Load eigenfunctions. On the first mode, also interpolate the model
        # properties at the eigenfunction grid points and calculate gravity.
        if i == 0:
            
            eigfunc_dict_0 = load_eigenfunc(run_info, mode_type, n, l, norm_args = norm_args)
            r = eigfunc_dict_0['r'] # m
            
            # Find indices of solid-fluid boundaries in the eigenfunction grid.
            i_fluid_solid_boundary = (np.where(np.diff(r) == 0.0))[0] + 1

            # Interpolate from the model grid to the output grid.
            rho  = interp_n_parts(r, model['r'], model['rho'],
                    i_fluid_solid_boundary, i_fluid_solid_boundary_model)
            v_p  = interp_n_parts(r, model['r'], model['v_p'],
                    i_fluid_solid_boundary, i_fluid_solid_boundary_model)
            v_s  = interp_n_parts(r, model['r'], model['v_s'],
                    i_fluid_solid_boundary, i_fluid_solid_boundary_model)
            i_fluid = np.where(v_s == 0.0)[0]

            if run_info['grav_switch'] in [1, 2]:

                # Calculate the gravitational acceleration.
                # (SI units m/s2)
                g_model      = gravitational_acceleration(model['r'], model['rho'])

                # Interpolate the gravitational acceleration at the nodal points.
                g = np.interp(r, model['r'], g_model)

            else:

                g = None

            # Find output directory.
            _, _, _, dir_out = get_Ouroboros_out_dirs(run_info, mode_type)
            dir_kernels = os.path.join(dir_out, 'kernels')
            mkdir_if_not_exist(dir_kernels)

        if (j_skip is None) or not (i in j_skip):
        
            # Make announcement.
            print('Calculating kernels for mode type {:} number {:>5d} of {:>5d}, n = {:>5d}, l = {:>5d}'.format(mode_type, i + 1, num_modes, n, l))

            # Load eigenfunction.
            eigfunc_dict = load_eigenfunc(run_info, mode_type, n, l, norm_args = norm_args)

            # Radial modes have no tangential component, but can otherwise
            # be treated the same as spheroidal modes.
            if mode_type == 'R':

                eigfunc_dict['V'] = np.zeros(eigfunc_dict['U'].shape)
                eigfunc_dict['Vp'] = np.zeros(eigfunc_dict['Up'].shape)
            
            # Calculate the kernels.
            K_ka, K_mu = get_kernels_S(r, eigfunc_dict['U'], eigfunc_dict['V'],
                            eigfunc_dict['Up'], eigfunc_dict['Vp'],
                            l, omega_rad_per_s, i_fluid = i_fluid)

            # Save the kernels.
            out_arr = np.array([r, K_ka, K_mu])
            name_out = 'kernels_{:>05d}_{:>05d}.npy'.format(n, l)
            path_out = os.path.join(dir_kernels, name_out)
            print('Saving to {:}'.format(path_out))
            np.save(path_out, out_arr)

    return

def kernels_wrapper_T(run_info, model, j_skip = None):
    '''
    Calculate kernels for all toroidal modes.
    '''

    mode_type = 'T'
    
    # Find the fluid-solid boundary points in the model.
    i_fluid_model, r_solid_fluid_boundary_model, i_fluid_solid_boundary_model =\
        get_r_fluid_solid_boundary(model['r'], model['v_s'])
    n_solid_regions = len(i_fluid_solid_boundary_model)

    # Define normalisation arguments.
    norm_args = {'norm_func' : 'DT', 'units' : 'SI'}

    # Loop over solid regions.
    for i_toroidal in range(n_solid_regions):
        
        # Get a list of all the modes calculated by Ouroboros.
        mode_info = load_eigenfreq(run_info, mode_type, i_toroidal = i_toroidal) 
        num_modes = len(mode_info['n'])

        # Loop over modes in this solid region.
        for i in range(num_modes):
            
            # Give shorter names to current n, l and f.
            n = mode_info['n'][i]
            l = mode_info['l'][i]
            f = mode_info['f'][i]
            omega_rad_per_s = f*1.0E-3*2.0*np.pi

            # To load eigenfunctions, we need the frequency for normalisation.
            norm_args['omega'] = omega_rad_per_s

            # Make announcement.
            mode_key = 'T{:}'.format(i_toroidal)
            print('Calculating kernels for mode type {:} number {:>5d} of \
                    {:>5d}, n = {:>5d}, l = {:>5d}'.format(mode_key, i + 1,
                        num_modes, n, l))

            # On the first mode, interpolate the model properties at the
            # eigenfunction grid points.
            if i == 0:

                # On the first mode, also interpolate the model
                # properties at the eigenfunction grid points and calculate gravity.
                eigfunc_dict = load_eigenfunc(run_info, mode_type, n, l,
                        i_toroidal = i_toroidal, norm_args = norm_args)
                r = eigfunc_dict['r']

                # Note for toroidal modes, no discontinuities need to be
                # accounted for in interpolation.
                rho = np.interp(r, model['r'], model['rho']) 
                v_s = np.interp(r, model['r'], model['v_s']) 

                # Also get output directory.
                _, _, _, dir_out = get_Ouroboros_out_dirs(run_info, mode_type)
                dir_kernels = os.path.join(dir_out, 'kernels_{:>03d}'
                                    .format(i_toroidal))
                mkdir_if_not_exist(dir_kernels)
            
            if (j_skip is None) or not (i in j_skip):

                # Load eigenfunction.
                eigfunc_dict = load_eigenfunc(run_info, mode_type, n, l,
                        i_toroidal = i_toroidal, norm_args = norm_args)

                # Calculate kernels.
                K_mu, K_rho = get_kernels_T(r, eigfunc_dict['W'],
                                eigfunc_dict['Wp'], l, omega_rad_per_s)

                # Save the kernels.
                out_arr = np.array([r, K_mu, K_rho])
                name_out = 'kernels_{:>05d}_{:>05d}.npy'.format(n, l)
                path_out = os.path.join(dir_kernels, name_out)
                print('Saving to {:}'.format(path_out))
                np.save(path_out, out_arr)

    return

def main():

    # Make announcement.
    print('run_kernels.py')

    # Parse input arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("path_input", help = "File path (relative or absolute) to Ouroboros input file.")
    input_args = parser.parse_args()
    path_input = input_args.path_input
    name_input = os.path.splitext(os.path.basename(path_input))[0]

    # Read the input file.
    run_info = read_input_file(path_input)

    for mode_type in run_info['mode_types']:

        kernels_wrapper(run_info, mode_type)
    
    return

if __name__ == '__main__':

    main()
