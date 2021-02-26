import argparse
import os

import numpy as np

from Ouroboros.common import get_Ouroboros_out_dirs, get_r_fluid_solid_boundary, interp_n_parts, load_eigenfreq_Ouroboros, load_eigenfunc_Ouroboros, mkdir_if_not_exist, load_model, read_Ouroboros_input_file
from Ouroboros.kernels.kernels import get_kernels_spheroidal, get_kernels_toroidal, gravitational_acceleration, potential

def kernels_wrapper(run_info, mode_type, model):

    if mode_type in ['R', 'S']:

        kernels_wrapper_R_or_S(run_info, model, mode_type)

    elif mode_type == 'T':

        kernels_wrapper_T(run_info, model)

    else:

        raise ValueError

    return

def kernels_wrapper_R_or_S(run_info, model, mode_type):

    # Get a list of all the modes calculated by Ouroboros.
    mode_info = load_eigenfreq_Ouroboros(run_info, mode_type) 
    n_list = mode_info['n']
    l_list = mode_info['l']
    f_mHz_list = mode_info['f']
    num_modes = len(n_list)
    
    # Find the fluid-solid boundary points in the model.
    i_fluid_model, r_solid_fluid_boundary_model, i_fluid_solid_boundary_model =\
        get_r_fluid_solid_boundary(model['r'], model['v_s'])

    for i in range(num_modes):
    #for i in [0]:
        
        # Give shorter names to current n, l and f.
        n = n_list[i]
        l = l_list[i]
        f_mHz = f_mHz_list[i]
        f_Hz = f_mHz/1.0E3
        omega_rad_per_s = f_Hz*(2.0*np.pi)

        # Make announcement.
        print('Calculating kernels for mode type {:} number {:>5d} of {:>5d}, n = {:>5d}, l = {:>5d}'.format(mode_type, i + 1, num_modes, n, l))

        # Load eigenfunctions. On the first mode, also interpolate the model
        # properties at the eigenfunction grid points and calculate gravity.
        if i == 0:
            
            # Load first eigenfunction.
            if mode_type == 'R':

                r, U = load_eigenfunc_Ouroboros(run_info, mode_type, n, l)
                V = np.zeros(U.shape)

            else:

                r, U, V = load_eigenfunc_Ouroboros(run_info, mode_type, n, l)

            ## Get planetary radius in km.
            #r_srf_km = r[-1]
            ## Calculate ratio for conversion to Mineos normalisation.
            #ratio = 1.0E-3*(r_srf_km**2.0)

            # Convert from km to m.
            r = r*1.0E3

            # Find indices of solid-fluid boundaries in the eigenfunction grid.
            i_fluid_solid_boundary = (np.where(np.diff(r) == 0.0))[0] + 1

            # Interpolate from the model grid to the output grid.
            rho  = interp_n_parts(r, model['r'], model['rho'], i_fluid_solid_boundary, i_fluid_solid_boundary_model)
            v_p  = interp_n_parts(r, model['r'], model['v_p'], i_fluid_solid_boundary, i_fluid_solid_boundary_model)
            v_s  = interp_n_parts(r, model['r'], model['v_s'], i_fluid_solid_boundary, i_fluid_solid_boundary_model)
            i_fluid = np.where(v_s == 0.0)[0]

            if run_info['grav_switch'] in [1, 2]:

                # Calculate the gravitational acceleration.
                # (SI units m/s2)
                g_model      = gravitational_acceleration(model['r'], model['rho'])

                # Interpolate the gravitational acceleration at the nodal points.
                g = np.interp(r, model['r'], g_model)

            else:

                g = None

        else:

            # Load eigenfunction.
            if mode_type == 'R':

                _, U = load_eigenfunc_Ouroboros(run_info, 'R', n, l)
                V = np.zeros(U.shape)

            else:

                _, U, V = load_eigenfunc_Ouroboros(run_info, 'S', n, l)

        ## Convert to Mineos normalisation.
        #U = U*ratio
        #V = V*ratio

        # Convert to D&T normalisation.
        U = U*omega_rad_per_s
        V = V*omega_rad_per_s

        if run_info['grav_switch'] == 2:

            # Calculate gravitational potential.
            P = potential(r, U, V, l, rho)

        else:

            P = None

        # Calculate the kernels.
        K_ka, K_mu, K_rho, K_alpha, K_beta, K_rhop = \
            get_kernels_spheroidal(omega_rad_per_s, r, U, V, l, v_p, v_s, run_info['grav_switch'],
                g = g,
                rho = rho,
                P = P,
                i_fluid = i_fluid)

        # Save the kernels.
        if i == 0:

            _, _, _, dir_out = get_Ouroboros_out_dirs(run_info, mode_type)
            dir_kernels = os.path.join(dir_out, 'kernels')
            mkdir_if_not_exist(dir_kernels)
    
        #out_arr = np.array([r, g, P, K_ka, K_mu, K_rho, K_alpha, K_beta, K_rhop])
        out_arr = np.array([r, K_ka, K_mu, K_rho, K_alpha, K_beta, K_rhop])
        name_out = 'kernels_{:>05d}_{:>05d}.npy'.format(n, l)
        path_out = os.path.join(dir_kernels, name_out)
        print('Saving to {:}'.format(path_out))
        np.save(path_out, out_arr)

    #import matplotlib.pyplot as plt

    #for v in [v_p, v_s, rho]:

    #    print(np.min(v), np.max(v))

    #fig = plt.figure()
    #ax = plt.gca()
    #ax.plot(g, r)
    #plt.show()
    
    #fig = plt.figure()
    #ax = plt.gca()

    #for E, E_label in zip([U, V, P], ['U', 'V', 'P']):

    #    ax.plot(E, r, label = E_label)

    ##ax.legend()

    ##plt.show()

    #for K in [K_ka, K_mu, K_rho, K_alpha, K_beta, K_rhop]:

    #    print(np.max(np.abs(K)))

    #fig, ax_arr  = plt.subplots(1, 3, figsize = (10.0, 6.0), sharey = True)
    #ax_arr[0].plot(K_ka, r)
    #ax_arr[1].plot(K_mu,  r)
    #ax_arr[2].plot(K_rho,  r)
    #plt.show()

    ##fig, ax_arr  = plt.subplots(1, 3, figsize = (10.0, 6.0))
    ##ax_arr[0].plot(K_alpha, r, label = 'vp')
    ##ax_arr[1].plot(K_beta,  r, label = 'vs')
    ##ax_arr[2].plot(K_rhop,  r, label = 'rho')
    ##plt.show()

    return

def kernels_wrapper_T(run_info, model):

    mode_type = 'T'
    
    # Find the fluid-solid boundary points in the model.
    i_fluid_model, r_solid_fluid_boundary_model, i_fluid_solid_boundary_model =\
        get_r_fluid_solid_boundary(model['r'], model['v_s'])
    n_solid_regions = len(i_fluid_solid_boundary_model)

    # Get planetary radius in km.
    r_srf_km = model['r'][-1]*1.0E-3
    # Calculate ratio for conversion to Mineos normalisation.
    ratio = 1.0E-3*(r_srf_km**2.0)

    for i_toroidal in range(n_solid_regions):
        
        # Get a list of all the modes calculated by Ouroboros.
        n_list, l_list, f_list = load_eigenfreq_Ouroboros(run_info, mode_type, i_toroidal = i_toroidal) 
        num_modes = len(n_list)

        for i in range(num_modes):
        #for i in [0]:
            
            # Give shorter names to current n, l and f.
            n = n_list[i]
            l = l_list[i]
            f = f_list[i]

            # Make announcement.
            mode_key = 'T{:}'.format(i_toroidal)
            print('Calculating kernels for mode type {:} number {:>5d} of {:>5d}, n = {:>5d}, l = {:>5d}'.format(mode_key, i + 1, num_modes, n, l))

            if i == 0:

                # Load eigenfunctions. On the first mode, also interpolate the model
                # properties at the eigenfunction grid points and calculate gravity.
                r, W = load_eigenfunc_Ouroboros(run_info, mode_type, n, l, i_toroidal = i_toroidal)

                rho = np.interp(r, model['r'], model['rho']) 
                #v_p = np.interp(r, model['r'], model['v_p']) 
                v_s = np.interp(r, model['r'], model['v_s']) 
            
            else:

                _, W = load_eigenfunc_Ouroboros(run_info, mode_type, n, l, i_toroidal = i_toroidal)


            # Convert to Mineos normalisation.
            W = W*ratio

            K_mu, K_rho, K_beta, K_rhop = \
                get_kernels_toroidal(f, r, W, l, rho, v_s)

        #import matplotlib.pyplot as plt

        #fig, ax_arr  = plt.subplots(1, 2, figsize = (6.0, 6.0), sharey = True)
        #ax_arr[0].plot(K_mu,  r)
        #ax_arr[1].plot(K_rho,  r)
        #plt.show()

        #fig, ax_arr  = plt.subplots(1, 3, figsize = (10.0, 6.0))
        #ax_arr[0].plot(K_alpha, r, label = 'vp')
        #ax_arr[1].plot(K_beta,  r, label = 'vs')
        #ax_arr[2].plot(K_rhop,  r, label = 'rho')
        #plt.show()

    return

def main():

    # Make announcement.
    print('run_kernels.py')

    # Parse input arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_input_file", help = "File path (relative or absolute) to Ouroboros input file.")
    input_args = parser.parse_args()
    Ouroboros_input_file = input_args.path_to_input_file
    name_input = os.path.splitext(os.path.basename(Ouroboros_input_file))[0]

    # Read the input file.
    Ouroboros_info = read_Ouroboros_input_file(Ouroboros_input_file)
    Ouroboros_info['use_attenuation'] = False

    # Load the planetary model.
    model = load_model(Ouroboros_info['path_model'])

    for mode_type in Ouroboros_info['mode_types']:
    ##for mode_type in ['S']:
    #for mode_type in ['R']:
    #for mode_type in ['T']:

        kernels_wrapper(Ouroboros_info, mode_type, model)
    
    return

if __name__ == '__main__':

    main()
