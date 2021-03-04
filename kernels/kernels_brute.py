import argparse
from functools import partial
import itertools
import multiprocessing
import os
import time

import numpy as np
from scipy.linalg import eigh

from Ouroboros.common import mkdir_if_not_exist, read_Ouroboros_input_file, get_Ouroboros_out_dirs
from Ouroboros.modes.compute_modes import prep_fem, build_matrices_radial_or_spheroidal, build_matrices_toroidal_and_solve, process_eigen_radial_or_spheroidal, process_eigen_toroidal

# -----------------------------------------------------------------------------
def kernel_prep(run_info, dir_output, mode_type, param_switch, d_p_over_p): 

    # Unpack input dictionary.
    if run_info['use_attenuation']:

        model_path = get_path_adjusted_model(run_info)

    else:

        model_path  = run_info['path_model']
    #l_min, l_max    = run_info['l_lims']
    #n_min, n_max    = run_info['n_lims']
    num_elmt        = run_info['n_layers']
    grav_switch        = run_info['grav_switch']
    
    # Create kernel dictionary.
    dir_kernel = os.path.join(dir_output, 'kernels_brute')
    mkdir_if_not_exist(dir_kernel)

    # Get the switch string.
    if mode_type in ['R', 'S']:

        grav_switch_str_dict = {0 : 'noGP', 1 : 'G', 2 : 'GP'}
        grav_switch_str = grav_switch_str_dict[grav_switch]
        switch = '{:}_{:}'.format(mode_type, grav_switch_str)

    else:

        grav_switch_str = None
        switch = 'T'

    # Set up the model and various finite-element parameters.
    (model, vs, count_thick, thickness, essen,
    invV, invV_p, invV_V, invV_P,
    order, order_p, order_V, order_P,
    Dr, Dr_p, Dr_V, Dr_P,
    x, x_V, x_P, VX,
    rho, radius,
    block_type, brk_radius, brk_num, layers,
    dir_eigenfunc, path_eigenvalues) = \
        prep_fem(model_path, dir_output, num_elmt, switch)

    # Calculate an array of the parameter change.
    if param_switch == 'ka':

        d_p = d_p_over_p * model.ka
        file_d_p = 'd_ka.npy'

    elif param_switch == 'mu':

        d_p = d_p_over_p * model.mu
        file_d_p = 'd_mu.npy'

    elif param_switch == 'rho':

        d_p = d_p_over_p * model.rho
        file_d_p = 'd_rho.npy'

    else:

        raise ValueError
    
    path_d_p = os.path.join(dir_kernel, file_d_p)
    print('Saving {:}'.format(path_d_p))
    np.save(path_d_p, d_p)
    
    # Calculate array of layer thickness.
    file_VX = 'VX.npy'
    path_VX = os.path.join(dir_kernel, file_VX)
    print('Saving {:}'.format(path_VX))
    np.save(path_VX, VX)

    return (model, vs, count_thick, thickness, essen,
            invV, invV_p, invV_V, invV_P,
            order, order_p, order_V, order_P,
            Dr, Dr_p, Dr_V, Dr_P,
            x, x_V, x_P, VX,
            rho, radius,
            block_type, brk_radius, brk_num, layers,
            dir_eigenfunc, path_eigenvalues,
            d_p, switch, grav_switch_str,
            dir_kernel)

def kernel_wrapper(run_info, mode_type, param_switch, parallel):

    print('Calculating sensitivity to parameter {:} of mode type {:}'.format(param_switch, mode_type))

    # Start timer.
    start_time = time.time()
    
    # Create output directory chain.
    dir_model, dir_run, dir_g, dir_type = get_Ouroboros_out_dirs(run_info, mode_type)
    for dir_ in [run_info['dir_output'], dir_model, dir_run, dir_g, dir_type]:
    
        if dir_ is not None:
            
            mkdir_if_not_exist(dir_)
    
    ## Create the input file.
    #write_input_file(dir_type, name_model, grav_switch, mode_type, n_lims, l_lims, n_layers)
    
    if mode_type == 'R':
    
        if parallel:
            
            print('Radial modes have only one l-value so parallel processing has no effect. Reverting to ordinary processing.')
    
        kernel_radial(run_info, dir_type, param_switch)
    
    elif mode_type == 'S':
    
        if parallel:
            
            #kernel_spheroidal_parallel(run_info, dir_type, param_switch)
            kernel_spheroidal_single_parallel_wrapper(run_info, dir_type, param_switch)

    #mode_type = 'S'

    ## Unpack input dictionary.
    #l_min, l_max    = run_info['l_lims']
    #n_min, n_max    = run_info['n_lims']
    #num_elmt        = run_info['n_layers']

    ## Prepare grid.
    #(model, vs, count_thick, thickness, essen,
    # invV, invV_p, invV_V, invV_P,
    # order, order_p, order_V, order_P,
    # Dr, Dr_p, Dr_V, Dr_P,
    # x, x_V, x_P, VX,
    # rho, radius,
    # block_type, brk_radius, brk_num, layers,
    # dir_eigenfunc, path_eigenvalues,
    # d_p, switch, grav_switch_str,
    # dir_kernel) = \
    #         kernel_prep(run_info, dir_output, mode_type, param_switch, d_p_over_p)

    ## Loop over angular order.
    #if l_min == 0:

    #    l_min_loop = 1

    #else:

    #    l_min_loop = l_min

    ##for l in range(l_min_loop, l_max):
    #for l in range(l_min_loop, l_min_loop + 1):

    #    if mode_type == 'S':

    #        if parallel:

    #            kernel_radial_or_spheroidal_single_parallel(
    #                n_min, n_max, l_min, l_max, model, count_thick, thickness,
    #                invV, invV_p, invV_V, invV_P,
    #                order, order_p, order_V, order_P,
    #                Dr, Dr_p, Dr_V, Dr_P,
    #                rho, radius,
    #                brk_radius, brk_num, layers, block_type,
    #                essen, x, x_V, num_elmt,
    #                d_p,
    #                switch, param_switch, grav_switch_str,
    #                dir_kernel, l)

    #        else:
    #
    #            kernel_spheroidal(run_info, dir_type, param_switch)
    #
    #    elif mode_type == 'T':

    #        if parallel:
    #    
    #            kernel_toroidal_parallel(run_info, dir_type, param_switch)

    #        else:

    #            kernel_toroidal(run_info, dir_type, param_switch)
    #    
    #    else:
    #    
    #        raise ValueError('Mode type {:} not recognised'.format(mode_type))
    #
    #end_time = time.time()
    #time_elapsed = end_time - start_time
    #print('Time used: {:.2f} seconds.'.format(time_elapsed)) 

    return

# -----------------------------------------------------------------------------
def kernel_spheroidal(run_info, dir_output, param_switch, d_p_over_p = 1.0E-2):

    # Set mode type string for spheroidal modes.
    mode_type = 'S'

    # Unpack input dictionary.
    l_min, l_max    = run_info['l_lims']
    n_min, n_max    = run_info['n_lims']
    num_elmt        = run_info['n_layers']

    # Prepare grid.
    (model, vs, count_thick, thickness, essen,
     invV, invV_p, invV_V, invV_P,
     order, order_p, order_V, order_P,
     Dr, Dr_p, Dr_V, Dr_P,
     x, x_V, x_P, VX,
     rho, radius,
     block_type, brk_radius, brk_num, layers,
     dir_eigenfunc, path_eigenvalues,
     d_p, switch, grav_switch_str,
     dir_kernel) = \
             kernel_prep(run_info, dir_output, mode_type, param_switch, d_p_over_p)

    # Loop over angular order.
    if l_min == 0:

        l_min_loop = 1

    else:

        l_min_loop = l_min

    for l in range(l_min_loop, l_max + 1):

        kernel_radial_or_spheroidal_single(
            n_min, n_max, l_min, l_max, model, count_thick, thickness,
            invV, invV_p, invV_V, invV_P,
            order, order_p, order_V, order_P,
            Dr, Dr_p, Dr_V, Dr_P,
            rho, radius,
            brk_radius, brk_num, layers, block_type,
            essen, x, x_V, num_elmt,
            d_p,
            switch, param_switch, grav_switch_str,
            dir_kernel, l)

    return

def kernel_spheroidal_single_parallel_wrapper(run_info, dir_output, param_switch, d_p_over_p = 1.0E-2):

    mode_type = 'S'

    # Unpack input dictionary.
    l_min, l_max    = run_info['l_lims']
    n_min, n_max    = run_info['n_lims']
    num_elmt        = run_info['n_layers']

    # Prepare grid.
    (model, vs, count_thick, thickness, essen,
     invV, invV_p, invV_V, invV_P,
     order, order_p, order_V, order_P,
     Dr, Dr_p, Dr_V, Dr_P,
     x, x_V, x_P, VX,
     rho, radius,
     block_type, brk_radius, brk_num, layers,
     dir_eigenfunc, path_eigenvalues,
     d_p, switch, grav_switch_str,
     dir_kernel) = \
             kernel_prep(run_info, dir_output, mode_type, param_switch, d_p_over_p)

    # Loop over angular order.
    if l_min == 0:

        l_min_loop = 1

    else:

        l_min_loop = l_min

    for l in range(l_min_loop, l_max + 1):
    #for l in range(l_min_loop, l_min_loop + 1):
        
        kernel_radial_or_spheroidal_single_parallel(
            n_min, n_max, l_min, l_max, model, count_thick, thickness,
            invV, invV_p, invV_V, invV_P,
            order, order_p, order_V, order_P,
            Dr, Dr_p, Dr_V, Dr_P,
            rho, radius,
            brk_radius, brk_num, layers, block_type,
            essen, x, x_V, num_elmt,
            d_p,
            switch, param_switch, grav_switch_str,
            dir_kernel, l)

    return

def kernel_spheroidal_parallel(run_info, dir_output, param_switch, d_p_over_p = 1.0E-2):

    mode_type = 'S'

    # Unpack input dictionary.
    l_min, l_max    = run_info['l_lims']
    n_min, n_max    = run_info['n_lims']
    num_elmt        = run_info['n_layers']

    # Prepare grid.
    (model, vs, count_thick, thickness, essen,
     invV, invV_p, invV_V, invV_P,
     order, order_p, order_V, order_P,
     Dr, Dr_p, Dr_V, Dr_P,
     x, x_V, x_P, VX,
     rho, radius,
     block_type, brk_radius, brk_num, layers,
     dir_eigenfunc, path_eigenvalues,
     d_p, switch, grav_switch_str,
     dir_kernel) = \
             kernel_prep(run_info, dir_output, mode_type, param_switch, d_p_over_p)

    # Loop over angular order.
    if l_min == 0:

        l_min_loop = 1

    else:

        l_min_loop = l_min

    l_span = list(range(l_min_loop, l_max + 1))

    n_processes = multiprocessing.cpu_count()
    #n_processes = 2
    print('Creating pool with {:d} processes.'.format(n_processes))
    with multiprocessing.Pool(processes = n_processes) as pool:
    
        # Note that the partial() function is used to meet the requirement of pool.map() of a pickleable function with a single input.
        kernel_spheroidal_single_partial = \
                partial(
                    kernel_spheroidal_single,
                        n_min, n_max, l_min, l_max, model, count_thick, thickness,
                        invV, invV_p, invV_V, invV_P,
                        order, order_p, order_V, order_P,
                        Dr, Dr_p, Dr_V, Dr_P,
                        rho, radius,
                        brk_radius, brk_num, layers, block_type,
                        essen, x, x_V, num_elmt,
                        d_p,
                        switch, param_switch, grav_switch_str,
                        dir_kernel)

        pool.map(kernel_spheroidal_single_partial, l_span)

def kernel_radial_or_spheroidal_single(
        n_min, n_max, l_min, l_max, model, count_thick, thickness,
        invV, invV_p, invV_V, invV_P,
        order, order_p, order_V, order_P,
        Dr, Dr_p, Dr_V, Dr_P,
        rho, radius,
        brk_radius, brk_num, layers, block_type,
        essen, x, x_V, num_elmt,
        d_p,
        switch, param_switch, grav_switch_str,
        dir_kernel, l):

    if l == 0:

        print('kernel_radial (switch = {:})'.format(grav_switch_str))

    else:

        print('kernel_spheroidal (switch = {:}): l = {:>5d} (from {:>5d} to {:>5d})'.format(grav_switch_str, l, l_min, l_max))
    
    # Get the reference frequencies (for the unperturbed model). ----------

    # Construct the matrices A and B.
    A_singularity, B_singularity, A0_inv,                   \
    E_singularity, B_eqv_pressure, block_type, block_len =  \
        build_matrices_radial_or_spheroidal(
            l, model, count_thick,
            invV, invV_p, invV_V, invV_P,
            order, order_p, order_V, order_P,
            Dr, Dr_p, Dr_V, Dr_P,
            rho, radius,
            block_type, brk_radius, brk_num, layers, switch)
    
    # Find the eigenvalues and eigenvectors. 
    eigvals, eigvecs = eigh(A_singularity, B_singularity)
    
    # Convert to mHz, remove essential spectrum, renormalise and save.            
    omega_ref, n_min_r = process_eigen_radial_or_spheroidal(
        l, eigvals, eigvecs,
        count_thick, thickness, essen, layers,
        n_min, n_max, order, order_V,
        x, x_V,
        block_type, block_len, A0_inv, E_singularity, B_eqv_pressure,
        None, None, switch, save = False)
    
    # For each element, apply a perturbation to the model and 
    # calculate the new frequency.
    n_freqs = len(omega_ref)
    omega_ptb = np.zeros((num_elmt, n_freqs))
    for i in range(num_elmt):
    #for i in range(1):
        
        print('element: {:>5d}'.format(i)) # Apply the perturbation to the i_th layer.

        if param_switch == 'ka':

            model.ka[i] = model.ka[i] + d_p[i]

        elif param_switch == 'mu':

            if model.mu[i] == 0.0:

                omega_ptb[i, :] = 0.0    
                continue

            else:

                model.mu[i] = model.mu[i] + d_p[i]

        elif param_switch == 'rho':

            model.rho[i] = model.rho[i] + d_p[i]

        # Construct the matrices A and B for the i_th perturbed model.
        A_singularity, B_singularity, A0_inv,                   \
        E_singularity, B_eqv_pressure, block_type, block_len =  \
            build_matrices_radial_or_spheroidal(
                l, model, count_thick,
                invV, invV_p, invV_V, invV_P,
                order, order_p, order_V, order_P,
                Dr, Dr_p, Dr_V, Dr_P,
                rho, radius,
                block_type, brk_radius, brk_num, layers, switch)

        # Find the eigenvalues and eigenvectors. 
        eigvals, eigvecs = eigh(A_singularity, B_singularity)
        
        # Calculate the eigenfrequencies for the i_th perturbed model.
        omega_ptb[i, :], _ = process_eigen_radial_or_spheroidal(
            l, eigvals, eigvecs,
            count_thick, thickness, essen, layers,
            n_min, n_max, order, order_V,
            x, x_V,
            block_type, block_len, A0_inv, E_singularity, B_eqv_pressure,
            None, None, switch, save = False)

        # Remove the perturbation from the i_th layer.
        if param_switch == 'ka':

            model.ka[i] = model.ka[i] - d_p[i]

        elif param_switch == 'mu':

            model.mu[i] = model.mu[i] - d_p[i]

        elif param_switch == 'rho':

            model.rho[i] = model.rho[i] - d_p[i]

    # Save.
    # Loop over radial order.
    for i_n, n in enumerate(range(n_min_r, n_max + 1)): 
        
        file_omega_ptb = 'omega_ptb_{:}_{:>05d}_{:>05d}.txt'.format(param_switch, n, l)
        path_omega_ptb = os.path.join(dir_kernel, file_omega_ptb) 
    
        with open(path_omega_ptb, 'w') as out_id:
            
            # Write a header with the reference frequency.
            out_id.write('# Ref: {:>18.14f}\n'.format(omega_ref[i_n]))
            
            # Loop over the layers of the model.
            for i in range(num_elmt):
            #for i in range(2):

                out_id.write('{:18.14f}\n'.format(omega_ptb[i, i_n]))

    return

def kernel_radial_or_spheroidal_single_parallel(
        n_min, n_max, l_min, l_max, model, count_thick, thickness,
        invV, invV_p, invV_V, invV_P,
        order, order_p, order_V, order_P,
        Dr, Dr_p, Dr_V, Dr_P,
        rho, radius,
        brk_radius, brk_num, layers, block_type,
        essen, x, x_V, num_elmt,
        d_p,
        switch, param_switch, grav_switch_str,
        dir_kernel, l):

    if l == 0:

        print('kernel_radial (switch = {:})'.format(grav_switch_str))

    else:

        print('kernel_spheroidal (switch = {:}): l = {:>5d} (from {:>5d} to {:>5d})'.format(grav_switch_str, l, l_min, l_max))
    
    # Get the reference frequencies (for the unperturbed model). ----------

    print('Calculating reference frequencies.')

    # Construct the matrices A and B.
    A_singularity, B_singularity, A0_inv,                   \
    E_singularity, B_eqv_pressure, block_type, block_len =  \
        build_matrices_radial_or_spheroidal(
            l, model, count_thick,
            invV, invV_p, invV_V, invV_P,
            order, order_p, order_V, order_P,
            Dr, Dr_p, Dr_V, Dr_P,
            rho, radius,
            block_type, brk_radius, brk_num, layers, switch)
    
    # Find the eigenvalues and eigenvectors. 
    eigvals, eigvecs = eigh(A_singularity, B_singularity)
    
    # Convert to mHz, remove essential spectrum, renormalise and save.            
    omega_ref, n_min_r = process_eigen_radial_or_spheroidal(
        l, eigvals, eigvecs,
        count_thick, thickness, essen, layers,
        n_min, n_max, order, order_V,
        x, x_V,
        block_type, block_len, A0_inv, E_singularity, B_eqv_pressure,
        None, None, switch, save = False)
    
    # For each element, apply a perturbation to the model and 
    # calculate the new frequency.
    n_freqs = len(omega_ref)
    omega_ptb = np.zeros((num_elmt, n_freqs))

    i_list = np.array(list(range(num_elmt)), dtype = np.int)

    # Solve in parallel.
    n_processes = multiprocessing.cpu_count()
    print('Creating pool with {:d} processes.'.format(n_processes))
    with multiprocessing.Pool(processes = n_processes) as pool:

        kernel_spheroidal_single_layer_partial = \
            partial(
                kernel_spheroidal_single_layer,
                param_switch, l, model, count_thick, invV, invV_p, invV_P, invV_V, order, order_p, order_V, order_P, Dr, Dr_p, Dr_V, Dr_P, rho, radius, block_type, block_len, brk_radius, brk_num, layers, switch, thickness, essen, n_min, n_max, x, x_V, d_p)

        results = pool.map(kernel_spheroidal_single_layer_partial, i_list)
    
    for i in i_list:

        omega_ptb[i, :] = results[i]

    # Save.
    # Loop over radial order.
    for i_n, n in enumerate(range(n_min_r, n_max + 1)): 
        
        file_omega_ptb = 'omega_ptb_{:}_{:>05d}_{:>05d}.txt'.format(param_switch, n, l)
        path_omega_ptb = os.path.join(dir_kernel, file_omega_ptb) 
    
        with open(path_omega_ptb, 'w') as out_id:
            
            # Write a header with the reference frequency.
            out_id.write('# Ref: {:>18.14f}\n'.format(omega_ref[i_n]))
            
            # Loop over the layers of the model.
            for i in range(num_elmt):
            #for i in range(2):

                out_id.write('{:18.14f}\n'.format(omega_ptb[i, i_n]))

    return

def kernel_spheroidal_single_layer(param_switch, l, model, count_thick, invV, invV_p, invV_P, invV_V, order, order_p, order_V, order_P, Dr, Dr_p, Dr_V, Dr_P, rho, radius, block_type, block_len, brk_radius, brk_num, layers, switch, thickness, essen, n_min, n_max, x, x_V, d_p, i):

    print('element: {:>5d}'.format(i)) # Apply the perturbation to the i_th layer.

    if param_switch == 'ka':

        model.ka[i] = model.ka[i] + d_p[i]

    elif param_switch == 'mu':

        if model.mu[i] == 0.0:

            omega_ptb = 0.0    
            return omega_ptb

        else:

            model.mu[i] = model.mu[i] + d_p[i]

    elif param_switch == 'rho':

        model.rho[i] = model.rho[i] + d_p[i]

    # Construct the matrices A and B for the i_th perturbed model.
    A_singularity, B_singularity, A0_inv,                   \
    E_singularity, B_eqv_pressure, block_type, block_len =  \
        build_matrices_radial_or_spheroidal(
            l, model, count_thick,
            invV, invV_p, invV_V, invV_P,
            order, order_p, order_V, order_P,
            Dr, Dr_p, Dr_V, Dr_P,
            rho, radius,
            block_type, brk_radius, brk_num, layers, switch)

    # Find the eigenvalues and eigenvectors. 
    eigvals, eigvecs = eigh(A_singularity, B_singularity)
    
    # Calculate the eigenfrequencies for the i_th perturbed model.
    omega_ptb, _ = process_eigen_radial_or_spheroidal(
        l, eigvals, eigvecs,
        count_thick, thickness, essen, layers,
        n_min, n_max, order, order_V,
        x, x_V,
        block_type, block_len, A0_inv, E_singularity, B_eqv_pressure,
        None, None, switch, save = False)

    # Remove the perturbation from the i_th layer.
    if param_switch == 'ka':

        model.ka[i] = model.ka[i] - d_p[i]

    elif param_switch == 'mu':

        model.mu[i] = model.mu[i] - d_p[i]

    elif param_switch == 'rho':

        model.rho[i] = model.rho[i] - d_p[i]

    return omega_ptb

# -----------------------------------------------------------------------------
def kernel_radial(run_info, dir_output, param_switch, d_p_over_p = 1.0E-2):
    
    # Set mode type string and angular order for radial modes.
    mode_type = 'R'
    l = 0

    # Unpack input dictionary.
    l_min = None
    l_max = None
    n_min, n_max    = run_info['n_lims']
    num_elmt        = run_info['n_layers']

    # Prepare grid.
    (model, vs, count_thick, thickness, essen,
     invV, invV_p, invV_V, invV_P,
     order, order_p, order_V, order_P,
     Dr, Dr_p, Dr_V, Dr_P,
     x, x_V, x_P, VX,
     rho, radius,
     block_type, brk_radius, brk_num, layers,
     dir_eigenfunc, path_eigenvalues,
     d_p, switch, grav_switch_str,
     dir_kernel) = \
             kernel_prep(run_info, dir_output, mode_type, param_switch, d_p_over_p)

    # Calculate brute-force kernel.
    kernel_radial_or_spheroidal_single(
         n_min, n_max, l_min, l_max, model, count_thick, thickness,
         invV, invV_p, invV_V, invV_P,
         order, order_p, order_V, order_P,
         Dr, Dr_p, Dr_V, Dr_P,
         rho, radius,
         brk_radius, brk_num, layers, block_type,
         essen, x, x_V, num_elmt,
         d_p,
         switch, param_switch, grav_switch_str,
         dir_kernel, l)

    return

# -----------------------------------------------------------------------------
def kernel_toroidal(run_info, dir_output, param_switch, d_p_over_p = 1.0E-2):

    # Set mode type string for toroidal modes.
    mode_type = 'T'

    # Unpack input dictionary.
    l_min, l_max    = run_info['l_lims']
    n_min, n_max    = run_info['n_lims']
    num_elmt        = run_info['n_layers']

    # Prepare grid.
    (model, vs, count_thick, thickness, essen,
     invV, invV_p, invV_V, invV_P,
     order, order_p, order_V, order_P,
     Dr, Dr_p, Dr_V, Dr_P,
     x, x_V, x_P, VX,
     rho, radius,
     block_type, brk_radius, brk_num, layers,
     dir_eigenfunc, path_eigenvalues,
     d_p, switch, grav_switch_str,
     dir_kernel) = \
             kernel_prep(run_info, dir_output, mode_type, param_switch, d_p_over_p)

    # Get the start and end points of each solid region.
    is_solid = model.mu > 0.0
    # https://stackoverflow.com/questions/53265826
    i_discon = [list(g)[0][0] for _, g in itertools.groupby(enumerate(is_solid), key=lambda x: x[-1])]
    n_discon = len(i_discon)
    i_discon.append(num_elmt)
    solid = model.mu[0] > 0.0

    i_start = []
    i_end = []
    j_solid_layers = []
    for j in range(n_discon):
        
        if solid:

            j_solid_layers.append(j)
            i_start.append(i_discon[j])
            i_end.append(i_discon[j + 1])

        solid = not(solid)

    dirs_layers = [os.path.join(dir_kernel, '{:}'.format(j)) for j in j_solid_layers]
    for dir_layers in dirs_layers:
        mkdir_if_not_exist(dir_layers)

    # Loop over angular order.
    if l_min == 0:

        l_min_loop = 1

    else:

        l_min_loop = l_min

    for l in range(l_min_loop, l_max + 1):

        kernel_toroidal_single(
            n_min, n_max, l_min, l_max, model, count_thick, thickness,
            j_solid_layers,
            invV, order, Dr,
            x,
            d_p,
            param_switch,
            dir_kernel,
            i_start, i_end,
            l)

    return

def kernel_toroidal_single(
        n_min, n_max, l_min, l_max, model, count_thick, thickness,
        j_solid_layers,
        invV, order, Dr,
        x,
        d_p,
        param_switch,
        dir_kernel,
        i_start, i_end, l):

    print('kernel_toroidal: l = {:>5d} (from {:>5d} to {:>5d})'.format(l, l_min, l_max))

    # Calculate asymptotic wavenumber.
    k = np.sqrt(l*(l + 1.0))
    model.set_k(k)

    for layer_i, j in enumerate(j_solid_layers):
    #for layer_i, j in enumerate([0]):
    #for layer_i, j in enumerate([2]):
    #for layer_i, j in zip([1], [2]):

        #print('Layer {:}'.format(j))

        dir_layer = os.path.join(dir_kernel, '{:}'.format(j))
    
        # Get the reference frequencies (for the unperturbed model). ----------

        # Construct the matrices and find the eigenvalues and eigenvectors for the
        # reference model.
        eigvals, eigvecs = build_matrices_toroidal_and_solve(model, count_thick, j, invV, order, Dr)

        # Get eigenvalues in mHz.
        omega_ref = process_eigen_toroidal(l, eigvals, eigvecs, n_min, n_max, count_thick, thickness, order, x, j, None, None, save = False)
        n_freqs = len(omega_ref)
        
        # For each element, apply a perturbation to the model and 
        # calculate the new frequency.
        num_elmt_layer = i_end[layer_i] - i_start[layer_i]
        omega_ptb = np.zeros((num_elmt_layer, n_freqs))

        for ii, i in enumerate(range(i_start[layer_i], i_end[layer_i])):
        #for ii, i in enumerate(range(i_start[layer_i], i_start[layer_i] + 2)):
            
            #print('element: {:>5d}'.format(i)) # Apply the perturbation to the i_th layer.

            if param_switch == 'ka':
                
                model.ka[i] = model.ka[i] + d_p[i]

            elif param_switch == 'mu':

                if model.mu[i] == 0.0:
                    
                    raise ValueError
                    omega_ptb[i, :] = 0.0    
                    continue

                else:

                    model.mu[i] = model.mu[i] + d_p[i]

            elif param_switch == 'rho':

                model.rho[i] = model.rho[i] + d_p[i]

            # Construct the matrices and find the eigenvalues and eigenvectors for the
            # reference model.
            eigvals, eigvecs = build_matrices_toroidal_and_solve(model, count_thick, j, invV, order, Dr)

            # Store the perturbed eigenvalues.
            omega_ptb[ii, :] = process_eigen_toroidal(l, eigvals, eigvecs, n_min, n_max, count_thick, thickness, order, x, j, None, None, save = False)
            
            # Remove the perturbation from the i_th layer.
            if param_switch == 'ka':

                model.ka[i] = model.ka[i] - d_p[i]

            elif param_switch == 'mu':

                model.mu[i] = model.mu[i] - d_p[i]

            elif param_switch == 'rho':

                model.rho[i] = model.rho[i] - d_p[i]

        # Save.
        # Loop over radial order.
        for i_n, n in enumerate(range(n_min, n_max + 1)): 

            file_omega_ptb = 'omega_ptb_{:}_{:>05d}_{:>05d}.txt'.format(param_switch, n, l)
            path_omega_ptb = os.path.join(dir_layer, file_omega_ptb)
            print('Writing to {:}'.format(path_omega_ptb))
        
            with open(path_omega_ptb, 'w') as out_id:
                
                # Write a header with the reference frequency.
                out_id.write('# Ref: {:>18.14f}\n'.format(omega_ref[i_n]))
                
                # Loop over the layers of the model.
                for i in range(num_elmt_layer):
                    
                    out_id.write('{:18.14f}\n'.format(omega_ptb[i, i_n]))

    return

def kernel_toroidal_parallel(run_info, dir_output, param_switch, d_p_over_p = 1.0E-2):

    mode_type = 'T'

    # Unpack input dictionary.
    l_min, l_max    = run_info['l_lims']
    n_min, n_max    = run_info['n_lims']
    num_elmt        = run_info['n_layers']

    # Prepare grid.
    (model, vs, count_thick, thickness, essen,
     invV, invV_p, invV_V, invV_P,
     order, order_p, order_V, order_P,
     Dr, Dr_p, Dr_V, Dr_P,
     x, x_V, x_P, VX,
     rho, radius,
     block_type, brk_radius, brk_num, layers,
     dir_eigenfunc, path_eigenvalues,
     d_p, switch, grav_switch_str,
     dir_kernel) = \
             kernel_prep(run_info, dir_output, mode_type, param_switch, d_p_over_p)

    # Get the start and end points of each solid region.
    is_solid = model.mu > 0.0
    # https://stackoverflow.com/questions/53265826
    i_discon = [list(g)[0][0] for _, g in itertools.groupby(enumerate(is_solid), key=lambda x: x[-1])]
    n_discon = len(i_discon)
    i_discon.append(num_elmt)
    solid = model.mu[0] > 0.0

    i_start = []
    i_end = []
    j_solid_layers = []
    for j in range(n_discon):
        
        if solid:

            j_solid_layers.append(j)
            i_start.append(i_discon[j])
            i_end.append(i_discon[j + 1])

        solid = not(solid)

    dirs_layers = [os.path.join(dir_kernel, '{:}'.format(j)) for j in j_solid_layers]
    for dir_layers in dirs_layers:
        mkdir_if_not_exist(dir_layers)

    # Loop over angular order.
    if l_min == 0:

        l_min_loop = 1

    else:

        l_min_loop = l_min

    l_span = list(range(l_min_loop, l_max + 1))

    n_processes = multiprocessing.cpu_count()
    #n_processes = 2
    print('Creating pool with {:d} processes.'.format(n_processes))
    with multiprocessing.Pool(processes = n_processes) as pool:
    
        # Note that the partial() function is used to meet the requirement of pool.map() of a pickleable function with a single input.
        kernel_toroidal_single_partial = \
                partial(
                    kernel_toroidal_single,
                        n_min, n_max, l_min, l_max, model, count_thick, thickness,
                        j_solid_layers,
                        invV, order, Dr,
                        x,
                        d_p,
                        param_switch,
                        dir_kernel,
                        i_start, i_end)

        pool.map(kernel_toroidal_single_partial, l_span)

# -----------------------------------------------------------------------------
def main():

    # Make announcement.
    print('Calculating brute-force kernels.')

    # Parse input arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_input_file", help = "File path (relative or absolute) to Ouroboros input file.")
    parser.add_argument("variable", choices = ['ka', 'mu', 'rho', 'all'], help = 'Variable for which the sensitivity kernels will be calculated (bulk modulus, ka; shear modulus, mu; density, rho; all variables, all')
    parser.add_argument("--parallel", action = 'store_true', help = 'Give this flag to run in parallel.')
    parser.add_argument("--all_grav_switches", action = 'store_true', help = 'Give this flag to override gravity switch in input file and calculate kernels for all three gravity switches.')
    input_args = parser.parse_args()
    Ouroboros_input_file = input_args.path_to_input_file
    name_input = os.path.splitext(os.path.basename(Ouroboros_input_file))[0]
    param_switch = input_args.variable
    parallel = input_args.parallel
    all_grav_switches = input_args.all_grav_switches

    # Read the input file.
    Ouroboros_info = read_Ouroboros_input_file(Ouroboros_input_file)

    if Ouroboros_info['use_attenuation']:

        create_adjusted_model(Ouroboros_info)

    ## Set the 'grav_switch' string: 0 -> noG, 1 -> G, 2 -> GP.
    #grav_switch_strs = ['noGP', 'G', 'GP']
    #grav_switch_str = grav_switch_strs[Ouroboros_info['grav_switch']]

    for mode_type in Ouroboros_info['mode_types']:
    #for mode_type in ['S']:
    #for mode_type in ['R']:
    #for mode_type in ['T']:

        if mode_type in ['R', 'S']:

            if param_switch == 'all':

                param_switch_list = ['ka', 'mu', 'rho']

            else:

                param_switch_list = [param_switch]

            if all_grav_switches:

                grav_switch_list = [0, 1, 2]

            else:

                grav_switch_list = [Ouroboros_info['grav_switch']]

        else:

            if param_switch == 'all':

                param_switch_list = ['mu', 'rho']

            else:

                param_switch_list = [param_switch]

            grav_switch_list = [Ouroboros_info['grav_switch']]

        for grav_switch in grav_switch_list:

            Ouroboros_info['grav_switch'] = grav_switch

            for param_switch_i in param_switch_list:

                kernel_wrapper(Ouroboros_info, mode_type, param_switch_i, parallel)

    return

if __name__ == '__main__':

    main()

# -----------------------------------------------------------------------------
def write_input_file(dir_out, name_model, grav_switch, mode_type, n_lims, l_lims, n_layers):
    
    name_input_file = 'input.txt' 
    path_input_file = os.path.join(dir_out, name_input_file)

    print('Writing input file: {:}'.format(path_input_file))

    grav_switch_dict = { 0 : -1, 1 : 0, 2 : 1}
    grav_switch_val= grav_switch_dict[grav_switch]

    mode_type_dict = {'R': 1, 'T' : 2, 'S' : 3}
    mode_type_val = mode_type_dict[mode_type]

    lines = ['{:}.txt'.format(name_model),
                'eigenvalues.txt',
                '{:d}'.format(grav_switch_val),
                '{:d}'.format(mode_type_val),
                '{:d} {:d} {:d} {:d}'.format(l_lims[0], l_lims[1], n_lims[0], n_lims[1]),
                '{:d}'.format(n_layers)]

    with open(path_input_file, 'w') as out_id:
        
        for line in lines:

            out_id.write(line + '\n')

    return
