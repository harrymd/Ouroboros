import argparse
import os

import numpy as np

from common import get_r_fluid_solid_boundary, mkdir_if_not_exist, load_model, read_Ouroboros_input_file
from kernels import get_kernels_spheroidal, gravitational_acceleration, potential
from post.read_output import load_eigenfreq_Ouroboros, load_eigenfunc_Ouroboros

def interp_n_parts(r, r_model, x_model, i_fluid_solid_boundary, i_fluid_solid_boundary_model):
    '''
    Careful interpolation of model parameters, preserving the fluid-solid discontinuities.
    '''

    n_parts = len(i_fluid_solid_boundary) + 1
    assert n_parts == (len(i_fluid_solid_boundary_model) + 1)
    
    i_fluid_solid_boundary = list(i_fluid_solid_boundary)
    i_fluid_solid_boundary.insert(0, 0)
    i_fluid_solid_boundary.append(None)

    i_fluid_solid_boundary_model = list(i_fluid_solid_boundary_model)
    i_fluid_solid_boundary_model.insert(0, 0)
    i_fluid_solid_boundary_model.append(None)
    
    x_list = []
    for i in range(n_parts):

        i0 = i_fluid_solid_boundary[i]
        i1 = i_fluid_solid_boundary[i + 1]

        i0_model = i_fluid_solid_boundary_model[i]
        i1_model = i_fluid_solid_boundary_model[i + 1]

        x_i = np.interp(r[i0 : i1], r_model[i0_model : i1_model], x_model[i0_model : i1_model])
        x_list.append(x_i)

    x = np.concatenate(x_list)

    return x

def kernels_wrapper(run_info, mode_type, model):

    if mode_type == 'S':

        kernels_wrapper_S(run_info, model)

    return

def kernels_wrapper_S(run_info, model):

    # Get a list of all the modes calculated by Ouroboros.
    n_list, l_list, f_list = load_eigenfreq_Ouroboros(run_info, 'S') 
    num_modes = len(n_list)

    # Find the fluid-solid boundary points in the model.
    i_fluid_model, r_solid_fluid_boundary_model, i_fluid_solid_boundary_model =\
        get_r_fluid_solid_boundary(model['r'], model['v_s'])

    #for i in range(num_modes):
    for i in [0]:
        
        # Give shorter names to current n, l and f.
        n = n_list[i]
        l = l_list[i]
        f = f_list[i]

        # Make announcement.
        print('Calculating kernels for mode {:>5d} of {:>5d}, n = {:>5d}, l = {:>5d}'.format(i + 1, num_modes, n, l))

        # Load eigenfunctions. On the first mode, also interpolate the model
        # properties at the eigenfunction grid points and calculate gravity.
        if i == 0:

            # Load first eigenfunction.
            r, U, V = load_eigenfunc_Ouroboros(run_info, 'S', n, l)

            # Get planetary radius in km.
            r_srf_km = r[-1]
            # Calculate ratio for conversion to Mineos normalisation.
            ratio = 1.0E-3*(r_srf_km**2.0)

            # Convert from km to m.
            r = r*1.0E3

            # Find indices of solid-fluid boundaries in the eigenfunction grid.
            i_fluid_solid_boundary = (np.where(np.diff(r) == 0.0))[0] + 1

            # Interpolate from the model grid to the output grid.
            rho  = interp_n_parts(r, model['r'], model['rho'], i_fluid_solid_boundary, i_fluid_solid_boundary_model)
            v_p  = interp_n_parts(r, model['r'], model['v_p'], i_fluid_solid_boundary, i_fluid_solid_boundary_model)
            v_s  = interp_n_parts(r, model['r'], model['v_s'], i_fluid_solid_boundary, i_fluid_solid_boundary_model)
            i_fluid = np.where(v_s == 0.0)[0]

            # Calculate the gravitational acceleration.
            # (SI units m/s2)
            g_model      = gravitational_acceleration(model['r'], model['rho'])

            # Interpolate the gravitational acceleration at the nodal points.
            g = np.interp(r, model['r'], g_model)

        else:

            # Load eigenfunction.
            _, U, V = load_eigenfunc_Ouroboros(run_info, 'S', n, l)

        # Convert to Mineos normalisation.
        U = U*ratio
        V = V*ratio

        # Calculate gravitational potential.
        P = potential(r, U, V, l, rho)

        K_ka, K_mu, K_rho, K_alpha, K_beta, K_rhop = \
            get_kernels_spheroidal(f, r, U, V, l, g, rho, v_p, v_s, P, i_fluid = i_fluid)

    import matplotlib.pyplot as plt

    for v in [v_p, v_s, rho]:

        print(np.min(v), np.max(v))

    #fig = plt.figure()
    #ax = plt.gca()
    #ax.plot(g, r)
    #plt.show()
    
    fig = plt.figure()
    ax = plt.gca()

    for E, E_label in zip([U, V, P], ['U', 'V', 'P']):

        ax.plot(E, r, label = E_label)

    #ax.legend()

    #plt.show()

    for K in [K_ka, K_mu, K_rho, K_alpha, K_beta, K_rhop]:

        print(np.max(np.abs(K)))

    fig, ax_arr  = plt.subplots(1, 3, figsize = (10.0, 6.0), sharey = True)
    ax_arr[0].plot(K_ka, r)
    ax_arr[1].plot(K_mu,  r)
    ax_arr[2].plot(K_rho,  r)
    plt.show()

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

    # Load the planetary model.
    model = load_model(Ouroboros_info['path_model'])

    ##for mode_type in Ouroboros_info['mode_types']:
    for mode_type in ['S']:

        kernels_wrapper(Ouroboros_info, mode_type, model)
    
    return

if __name__ == '__main__':

    main()