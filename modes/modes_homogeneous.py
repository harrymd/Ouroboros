'''
Calculate modes (frequencies and eigenfunctions) of homogeneous spherical
planet. See [1], sections 8.7.4 and 8.8.7.

Reference

[1] Dahlen and Tromp (1998) 'Theoretical global seismology'.
'''

import argparse
import os

import numpy as np
from scipy.special import spherical_jn as spherical_bessel

from Ouroboros.common import (mkdir_if_not_exist,
        read_Ouroboros_homogeneous_input_file)
from Ouroboros.misc.root_finding import roots

def compute_modes_homogeneous_T(dir_output_master, r, rho, beta, num_samples, n_max, l_max, om_max, root_tolerance_mHz):

    print('Searching for toroidal modes...')

    # Define root function for arbitrary l-value.
    # [1] eq. 8.120.
    func = lambda x, l: (l - 1.0) * spherical_bessel(l, x) - \
                                x * spherical_bessel(l + 1, x)

    # Prepare output arrays.
    l_list = []
    n_list = []
    x_list = []

    # Find tolerance and  upper limit of root search in dimensionless variable.
    x_max = (om_max * r) / beta
    root_tolerance_rad_per_s = (root_tolerance_mHz * 1.0E-3 * 2.0 * np.pi)
    root_tolerance = (root_tolerance_rad_per_s * r) / beta

    # Loop over l-values.
    # Note: No modes with l = 0.
    for l in range(1, l_max + 1):

        # Get constructive inference condition for this value of l.
        func_l = lambda x: func(x, l)

        # Find the roots.
        x_l = roots(func_l, 0.0, x_max, eps = root_tolerance,
                    n_roots_max = n_max + 2, verbose = False)

        # Get the non-zero roots.
        x_l = np.array(x_l)
        i_non_zero = np.where(x_l > (10.0 * root_tolerance))[0]

        # Store the results.
        x_l = x_l[i_non_zero]
        num_l = len(x_l)
        l_list.extend([l] * num_l)
        n_list_i = np.array(list(range(num_l)), dtype = np.int)
        # Account for missing modes with l = 0 and l = 1.
        if (l == 0) or (l == 1):

            n_list_i = n_list_i + 1

        n_list.extend(n_list_i)
        x_list.extend(x_l)

        print('l = {:>5d}, found {:>5d} modes'.format(l, num_l))

    # Convert to NumPy array. 
    l_list = np.array(l_list, dtype = np.int)
    n_list = np.array(n_list, dtype = np.int)
    x_list = np.array(x_list)
    num_modes = len(l_list)

    # Convert from dimensionless variable x to frequency.
    om_list = ((x_list * beta) / r)
    f_list_mHz = 1.0E3 * (om_list / (2.0 * np.pi))

    # Save eigenvalues.
    # Prepare output directory.
    dir_output = os.path.join(dir_output_master, 'T')
    mkdir_if_not_exist(dir_output)
    path_eigenvalues = os.path.join(dir_output, 'eigenvalues_000.txt')
    with open(path_eigenvalues, 'w') as out_id:

        # Loop over modes.
        for i in range(num_modes):

            out_id.write('{:>10d} {:>10d} {:>19.12e} {:>19.12e} {:>19.12e}\n'\
                        .format(n_list[i], l_list[i],
                            f_list_mHz[i], f_list_mHz[i], 0.0))

    # Calculate the eigenfunctions.
    # Prepare output directory.
    dir_eigenfunctions = os.path.join(dir_output, 'eigenfunctions_000')
    mkdir_if_not_exist(dir_eigenfunctions)
    #
    # Define a grid of points.
    r_samples = np.linspace(0.0, r, num = num_samples)
    #
    # Loop over modes.

    for i in range(num_modes):
        
        # Define Bessel function for this l-value and evaluate it.
        func_l = lambda x: spherical_bessel(l_list[i], x)
        x_samples = (om_list[i] * r_samples) / beta
        W = func_l(x_samples)

        # Calculate the norm of the eigenfunction and apply the
        # Ouroboros normalisation.
        # Note: Factor of 1.0E-3 converts from m to km.
        k = np.sqrt(l * (l + 1.0))
        prefactor = (rho * ((om_list[i] * k) ** 2.0))
        #prefactor = (rho * ((k * (om_list[i] ** 1.5)) ** 2.0))
        integrand = ((W * r_samples * 1.0E-3) ** 2.0)
        integral = prefactor * np.trapz(integrand, x = r_samples)
        W = (W / np.sqrt(integral))

        # Save the eigenfunction in Ouroboros format.
        file_eigenfunc = '{:>05d}_{:>05d}.npy'.format(n_list[i], l_list[i])
        path_eigenfunc = os.path.join(dir_eigenfunctions, file_eigenfunc)
        Wp_dummy = np.zeros(len(W))
        out_arr = np.array([(r_samples * 1.0E-3), W, Wp_dummy])
        np.save(path_eigenfunc, out_arr)

    return

def main(): 

    # Read input arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("path_input", help = "File path (relative or absolute) to input file.")
    #
    args = parser.parse_args()

    # Rename input arguments.
    path_input = args.path_input    

    # Read input file (model parameters in SI units).
    run_info = read_Ouroboros_homogeneous_input_file(path_input)
    # Unpack variables.
    dir_output          = run_info['dir_output']
    r                   = run_info['r']
    mu                  = run_info['mu']
    kappa               = run_info['kappa']
    rho                 = run_info['rho']
    n_max               = run_info['n_max']
    l_max               = run_info['l_max']
    f_max_mHz           = run_info['f_max_mHz']
    root_tolerance_mHz  = run_info['root_tol_mHz']
    num_samples         = run_info['num_samples']

    # Create output directory.
    mkdir_if_not_exist(dir_output)

    # Get derived quantities (SI units).
    # alpha     P-wave speed (m/s).
    # beta      S-wave speed (m/s). 
    # f_max_Hz  Upper limit of search in frequency domain (Hz). 
    # om_max    Upper limit of search in frequency domain (rad/s).
    beta = np.sqrt(mu / rho)
    #
    f_max_Hz    = (f_max_mHz * 1.0E-3)
    om_max      = (2.0 * np.pi * f_max_Hz)

    # Compute the toroidal modes.
    compute_modes_homogeneous_T(dir_output, r, rho, beta, num_samples, n_max,
            l_max, om_max, root_tolerance_mHz)

if __name__ == '__main__':

    main()
