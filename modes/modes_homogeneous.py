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
from Ouroboros.constants import G
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
        # Note: Factor of 1.0E-3 converts from m to km and from kg/m3 to g/cm3.
        k = np.sqrt(l_list[i] * (l_list[i] + 1.0))
        prefactor = (rho * 1.0E-3 * ((om_list[i] * k) ** 2.0))
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

def compute_modes_homogeneous_S(dir_output_master, a, rho, ka, mu, alpha, beta, n_max, l_max, om_max, root_tolerance_mHz):
    
    raise NotImplementedError
    print('Searching for spheroidal modes...')

    # Define root function for arbitrary l-value.
    # [1] eq. 8.184.
    func = lambda om, l: determinant_S(om, rho, ka, mu, alpha, beta, l, a)

    # Prepare output arrays.
    l_list = []
    n_list = []
    om_list = []

    # Find tolerance and  upper limit of root search in angular frequency.
    root_tolerance_rad_per_s = (root_tolerance_mHz * 1.0E-3 * 2.0 * np.pi)

    # Loop over l-values.
    # Note: Skip radial modes with l = 0.
    for l in range(1, l_max + 1):

        # Get constructive inference condition for this value of l.
        func_l = lambda om: func(om, l)

        # Find the roots.
        om_l = roots(func_l, 0.0, om_max, eps = root_tolerance_rad_per_s,
                    n_roots_max = n_max + 2, verbose = False)

        # Get the non-zero roots.
        om_l = np.array(om_l)
        i_non_zero = np.where(om_l > (10.0 * root_tolerance_rad_per_s))[0]

        # Store the results.
        om_l = om_l[i_non_zero]
        num_l = len(om_l)
        l_list.extend([l] * num_l)
        n_list_i = np.array(list(range(num_l)), dtype = np.int)
        ## Account for missing modes with l = 0 and l = 1.
        #if (l == 0) or (l == 1):

        #    n_list_i = n_list_i + 1

        n_list.extend(n_list_i)
        om_list.extend(om_l)

        print('l = {:>5d}, found {:>5d} modes'.format(l, num_l))

    # Convert to NumPy array. 
    l_list  = np.array(l_list, dtype = np.int)
    n_list  = np.array(n_list, dtype = np.int)
    om_list = np.array(om_list)
    num_modes = len(l_list)

    # Convert from dimensionless variable x to frequency.
    f_list_mHz = 1.0E3 * (om_list / (2.0 * np.pi))

    # Save eigenvalues.
    # Prepare output directory.
    dir_output = os.path.join(dir_output_master, 'GP', 'S')
    mkdir_if_not_exist(dir_output)
    path_eigenvalues = os.path.join(dir_output, 'eigenvalues.txt')
    with open(path_eigenvalues, 'w') as out_id:

        # Loop over modes.
        for i in range(num_modes):

            out_id.write('{:>10d} {:>10d} {:>19.12e} {:>19.12e} {:>19.12e}\n'\
                        .format(n_list[i], l_list[i],
                            f_list_mHz[i], f_list_mHz[i], 0.0))

    return

def determinant_S(om, rho, ka, mu, alpha, beta, l, a):
    '''
    Calculate determinant in [1] eq. 8.184.
    '''

    R1, R2, R3, S1, S2, S3, B1, B2, B3 = \
            S_linear_indep_solutions(om, rho, ka, mu, alpha, beta, l, a)

    M = np.array([[R1, R2, R3], [S1, S2, S3], [B1, B2, B3]])

    det = np.linalg.det(M)

    return det

def S_linear_indep_solutions(om, rho, ka, mu, alpha, beta, l, a):
    '''
    Evaluate [1] eq. 8.176–8.178 and 8.182–8.183 at r = a.
    '''

    # Calculate gamma variable.
    k = np.sqrt(l * (l + 1.0))
    gam_pos, gam_neg = gamma(om, alpha, beta, rho, k)

    # Calculate zeta and xi variables.
    zeta_pos, xi_pos = S_zeta_and_xi(om, rho, beta, gam_pos, l)
    zeta_neg, xi_neg = S_zeta_and_xi(om, rho, beta, gam_neg, l)

    R1 = S_linear_indep_R12(gam_pos, ka, mu, zeta_pos, xi_pos, l, a) 
    R2 = S_linear_indep_R12(gam_neg, ka, mu, zeta_neg, xi_neg, l, a) 
    R3 = S_linear_indep_R3(mu, l, a)

    S1 = S_linear_indep_S12(gam_pos, ka, mu, zeta_pos, xi_pos, l, a)
    S2 = S_linear_indep_S12(gam_neg, ka, mu, zeta_neg, xi_neg, l, a)
    S3 = S_linear_indep_S3(mu, l, a)

    B1 = S_linear_indep_B12(gam_pos, rho, zeta_pos, l, a)
    B2 = S_linear_indep_B12(gam_neg, rho, zeta_neg, l, a)
    B3 = S_linear_indep_B3(om, rho, beta, a)

    return R1, R2, R3, S1, S2, S3, B1, B2, B3

def gamma(om, alpha, beta, rho, k):
    '''
    [1] eq. 8.179.
    '''

    G1 = (16.0 * np.pi * G * rho) / 3.0

    a = (om / beta) ** 2.0
    b = ((om ** 2.0) + G1) / (alpha ** 2.0)

    A = (0.5 * a)
    B = (0.5 * b)

    c1 = (a - b) ** 2.0
    c2 = ((8.0 * np.pi * G * k * rho) / (3.0 * alpha * beta)) ** 2.0
    C  = 0.5 * np.sqrt(c1 + c2)

    gam2_pos = (A + B + C)
    gam2_neg = (A + B - C)

    #assert (gam2_pos > 0.0) and (gam2_neg > 0.0)

    gam_pos = np.sqrt(gam2_pos)
    gam_neg = np.sqrt(gam2_neg)

    return gam_pos, gam_neg

def S_linear_indep_R12(gamma, ka, mu, zeta, xi, l, r):
    '''
    [1] eq. 8.176.
    '''
    
    k2 = (l * (l + 1.0))
    gam2 = (gamma ** 2.0)

    Ra = -1.0 * (((ka + (4.0 * mu / 3.0)) * zeta * gam2) +
                ((2.0 * l * (l + 1.0) * mu * xi)/(r ** 2.0))) \
                * spherical_bessel(l, gamma * r)

    Rb = -1.0 * ((2.0 * mu * ((2.0 * zeta) + k2) * gamma)/r) \
                * spherical_bessel(l + 1, gamma * r)

    R = (Ra + Rb)

    return R

def S_linear_indep_S12(gamma, ka, mu, zeta, xi, l, r):
    '''
    [1] eq. 8.177.
    '''

    k = np.sqrt(l * (l + 1.0))
    gam2 = (gamma ** 2.0)

    Sa = (k * mu * (gam2 + (2.0 * (l - 1.0) * xi / r ** 2.0))
            * spherical_bessel(l, gamma * r))
    
    Sb = ((-1.0 * k * mu * (zeta + 1.0) * gamma / r)
            * spherical_bessel(l + 1, gamma * r))

    S = (Sa + Sb)

    return S

def S_linear_indep_B12(gamma, rho, zeta, l, r):
    '''
    [1] eq. 8.178.
    '''
    
    k2 = (l * (l + 1.0))
    
    B = ((-4.0 * np.pi * G * rho / r) * (k2 - (l + 1.0) * zeta) 
        * spherical_bessel(l, gamma * r)) 

    return B

def S_linear_indep_R3(mu, l, r):
    '''
    [1] eq. 8.182a.
    '''

    R = 2.0 * l * (l + 1.0) * mu * (r ** (l - 2.0))
    
    return R

def S_linear_indep_S3(mu, l, r):
    '''
    [1] eq. 8.182b.
    '''
    
    k = np.sqrt(l * (l + 1.0))
    S = 2.0 * k * (l - 1.0) * mu * (r ** (l - 2.0)) 

    return S 

def S_linear_indep_B3(om, rho, l, r):
    '''
    [1] eq. 8.183b.
    '''

    B = ((((2.0 * l) + 1) * (om ** 2.0))
            - ((8.0 / 3.0) * np.pi * G * l * (l - 1.0) * rho)) \
                    * (r ** (l - 1.0))

    return B

def S_zeta_and_xi(om, rho, beta, gamma, l):
    '''
    [1] eq. 8.180.
    '''

    zeta = (3.0 * (beta ** 2.0) * (gamma ** 2.0 - (om / beta) ** 2.0)) / \
            (4.0 * np.pi * G * rho)

    xi = zeta - (l + 1.0)
    
    return zeta, xi

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
    alpha = np.sqrt((kappa + (4.0 * mu / 3.0))/ rho)
    beta = np.sqrt(mu / rho)
    #
    f_max_Hz    = (f_max_mHz * 1.0E-3)
    om_max      = (2.0 * np.pi * f_max_Hz)

    # Compute the toroidal modes.
    compute_modes_homogeneous_T(dir_output, r, rho, beta, num_samples, n_max,
            l_max, om_max, root_tolerance_mHz)

    ## Compute the spheroidal modes.
    #compute_modes_homogeneous_S(dir_output, r, rho, kappa, mu, alpha, beta,
    #        n_max, l_max, om_max, root_tolerance_mHz)

    return

if __name__ == '__main__':

    main()
