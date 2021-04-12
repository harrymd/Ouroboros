import argparse
import os
from shutil import copyfile

import numpy as np

from Ouroboros.common import get_path_adjusted_model, load_eigenfreq_Ouroboros, load_kernel, load_model_full, mkdir_if_not_exist, get_Ouroboros_out_dirs, read_Ouroboros_input_file

def dispersion_correction(x, Q_x, omega, omega_ref):
    '''
    Dahlen and Tromp (1998) eq. 9.50-9.51

    x           Shear or bulk modulus.
    Q_x         Shear or bulk Q-factor.
    omega       Frequency (rad/s).
    omega_ref   Reference frequency (rad/s).
    '''

    d_x = (2.0 * x * np.log(omega / omega_ref)) / (np.pi * Q_x)

    i_bad = np.where(Q_x == 0.0)[0]
    d_x[i_bad] = 0.0

    return d_x

def relabel_modes(dir_output, n_orig, l, f_old, f_new, Q_new):

    l_vals = np.sort(np.unique(l))

    num_modes = len(f_new)
    n_old_list = np.zeros(num_modes, dtype = np.int)
    n_new_list = np.zeros(num_modes, dtype = np.int)
    l_list = np.zeros(num_modes, dtype = np.int)
    f_old_list = np.zeros(num_modes)
    f_new_list = np.zeros(num_modes)
    Q_new_list = np.zeros(num_modes)
    
    k = 0
    for l_i in l_vals:

        i = np.where(l == l_i)[0]
        
        n_orig_i = n_orig[i]
        n_min_orig_i = np.min(n_orig_i)
        
        num_i = len(i)
        n_new_i = np.array(range(n_min_orig_i, n_min_orig_i + num_i), dtype = np.int)
        l_new_i = np.zeros(num_i, dtype = np.int) + l_i

        j = np.argsort(f_new[i])
        n_orig_i_sorted = n_orig_i[j]
        f_new_i_sorted = f_new[i][j]
        f_old_i_sorted = f_old[i][j]
        Q_new_i_sorted = Q_new[i][j]

        l_list    [k : k + num_i] = l_i
        n_new_list[k : k + num_i] = n_new_i
        n_old_list[k : k + num_i] = n_orig_i_sorted
        f_new_list[k : k + num_i] = f_new_i_sorted
        f_old_list[k : k + num_i] = f_old_i_sorted
        Q_new_list[k : k + num_i] = Q_new_i_sorted

        k = k + num_i
    
    #fmt = '{:>5d} {:>5d} {:>5d} {:>8.3f} {:}'
    #fmt = '{:>5d} {:>5d} {:>5d} {:>5d}'
    #dir_eigenfunctions_old = os.path.join(dir_output, 'eigenfunctions')
    #dir_eigenfunctions_new = os.path.join(dir_output, 'eigenfunctions_relabelled')
    #mkdir_if_not_exist(dir_eigenfunctions_new)
    #with open(path_relabel_list, 'w') as out_id:

    for k in range(num_modes):

            #print(fmt.format(n_old_list[k], n_new_list[k], l_list[k], f_new_list[k], n_old_list[k] == n_new_list[k]))



        if n_new_list[k] != n_old_list[k]:

            print('{:>3d} S {:>3d} ({:>6.2f}) --> {:>3d} S {:>3d} ({:>6.2f})'.format(n_old_list[k], l_list[k], f_old_list[k],  n_new_list[k], l_list[k], f_new_list[k]))

            #name_eigfunc_old = '{:>05d}_{:>05d}.npy'.format(n_old_list[k], l_list[k])
            #path_eigfunc_old = os.path.join(dir_eigenfunctions_old, name_eigfunc_old)
            ##
            #name_eigfunc_new = '{:>05d}_{:>05d}.npy'.format(n_new_list[k], l_list[k])
            #path_eigfunc_new = os.path.join(dir_eigenfunctions_new, name_eigfunc_new)
            #
            ##print('cp {:} {:}'.format(path_eigfunc_old, path_eigfunc_new))
            #copyfile(path_eigfunc_old, path_eigfunc_new)

    #path_eigvals_new = os.path.join(dir_output, 'eigenvalues_relabelled.txt')
    path_eigvals = os.path.join(dir_output, 'eigenvaluesd.txt')
    print("Writing {:}".format(path_eigvals_new))
    #fmt = '{:>10d} {:>10d} {:>10d} {:>10d} {:>19.12e} {:>19.12e} {:>19.12e}'
    fmt = '{:>10d} {:>10d} {:>10d} {:>10d} {:>19.12e} {:>19.12e} {:>19.12e}'
    with open(path_eigvals_new, 'w') as out_id:

        for k in range(num_modes):

            out_id.write(fmt.format(n_old_list[k], l_list[k], n_new_list[k], l_list[k], f_old_list[k], f_new_list[k], Q_new_list[k]) + '\n') 

    # Re-scale the eigenfunctions.

    return

def calculate_Q_wrapper(run_info, model, mode_type, n, l, f_rad_per_s, neglect_ka = False):

    # Load kernel.
    r, K_ka, K_mu = load_kernel(run_info, mode_type, n, l, units = 'SI')

    # Interpolate the model.
    model_interpolated = dict()
    model_interpolated['T_ref'] = model['T_ref']
    model_interpolated['f_ref_rad_per_s'] = 2.0*np.pi/model_interpolated['T_ref']
    model_interpolated['r'] = r
    for key in ['mu', 'ka', 'Q_mu', 'Q_ka']:

        model_interpolated[key] = np.interp(r, model['r'], model[key])

    model  = model_interpolated

    ## Calculate dispersion corrections.
    #d_mu = dispersion_correction(model['mu'], model['Q_mu'],
    #            f_rad_per_s, model['f_ref_rad_per_s'])
    #model['mu'] = model['mu'] + d_mu

    #d_ka = dispersion_correction(model['ka'], model['Q_ka'],
    #            f_rad_per_s, model['f_ref_rad_per_s'])
    #model['ka'] = model['ka'] + d_ka
    
    verbose = False
    if verbose:

        print('max(Q_mu): {:>10.3e}'.format(np.max(np.abs(model['Q_mu']))))
        print('max(Q_ka): {:>10.3e}'.format(np.max(np.abs(model['Q_ka']))))
        print('max(mu): {:>10.3e}'.format(np.max(np.abs(model['mu']))))
        print('max(ka): {:>10.3e}'.format(np.max(np.abs(model['ka']))))
        print('max(K_mu): {:>10.3e}'.format(np.max(np.abs(K_mu))))
        print('max(K_ka): {:>10.3e}'.format(np.max(np.abs(K_ka))))

    # Evaluate the integral (D&T 1998 eq. 9.54).
    I_mu = (model['mu']*K_mu/model['Q_mu'])
    i_fluid = np.where(model['mu'] == 0.0)[0]
    I_mu[i_fluid] = 0.0
    
    if neglect_ka:

        integrand = I_mu
    
    else:

        I_ka = (model['ka']*K_ka/model['Q_ka']) 
        integrand = I_mu + I_ka

    integral = np.trapz(integrand, x = r)
    inv_Q = 2.0*integral/f_rad_per_s
    Q = 1.0/inv_Q
    
    # Fudge -- need to check.
    Q = Q/(2.0*np.pi)
    
    plotting = True
    if plotting:

        import matplotlib.pyplot as plt

        fig, ax_arr = plt.subplots(1, 5, sharey = True, figsize = (14.0, 8.5), constrained_layout = True)

        ax = ax_arr[0]
        ax.plot(integrand, r, label = 'Integrand (sum)')
        ax.plot(I_mu, r, label = 'Integrand (mu)')
        ax.plot(I_ka, r, label = 'Integrand (ka)')
        #ax.plot(I_ka*10.0, r, label = 'Integrand (ka x 10)')

        ax = ax_arr[1]
        ax.plot(model['mu'], r, label = 'mu')
        ax.plot(model['ka'], r, label = 'ka')

        ax = ax_arr[2]
        ax.plot(K_mu, r, label = 'K_mu')
        ax.plot(K_ka, r, label = 'K_ka')

        ax = ax_arr[3]
        ax.plot(model['mu']*K_mu, r, label = 'mu*K_mu')
        ax.plot(model['ka']*K_ka, r, label = 'ka*K_ka')

        ax = ax_arr[4]
        ax.plot(1.0/model['Q_mu'], r, label = '1.0/Q_mu')
        ax.plot(1.0/model['Q_ka'], r, label = '1.0/Q_ka')

        for ax in ax_arr:

            ax.legend()
        
        plt.show()

        import sys
        sys.exit()

    return Q

def apply_atten_correction_all_modes(run_info, mode_type):

    # Load the planetary model.
    model_path = get_path_adjusted_model(run_info)
    model = load_model_full(model_path)
    # Throw away unnecessary items.
    keys_to_keep = ['T_ref', 'r', 'mu', 'ka', 'Q_mu', 'Q_ka']
    model_new = {key : model[key] for key in keys_to_keep}
    model = model_new

    omega_ref_Hz = 1.0/model['T_ref']
    omega_ref_rad_per_s = 2.0*np.pi*omega_ref_Hz

    #neglect_ka = True
    neglect_ka = False
    
    out_fmt = '{:>5d} {:>5d} {:>19.12e} {:>12.6e} {:>19.12e} {:>19.12e}'

    # Override attenuation flag.
    #run_info['use_attenuation'] = False

    # Get output directory.
    #_, _, _, dir_output = \
    #    get_Ouroboros_out_dirs(run_info, mode_type)
    #path_out = os.path.join(dir_output, 'Q_values.txt')

    # Load mode database.
    mode_info = load_eigenfreq_Ouroboros(run_info, mode_type)
    n = mode_info['n']
    l = mode_info['l']
    omega_uncorrected_mHz = mode_info['f_0']

    omega_rad_per_s = omega_uncorrected_mHz*1.0E-3*2.0*np.pi
    num_modes = len(n)
    Q = np.zeros(num_modes)
    
    single_mode = False
    if single_mode:

        n_q =  8 
        l_q =  2 
        j = np.where((n == n_q) & (l == l_q))[0][0]
        calculate_Q_wrapper(run_info, model, mode_type, n_q, l_q, omega_rad_per_s[j], neglect_ka = neglect_ka) 

        return

    # Loop over modes.
    for i in range(num_modes):
        
        print('Calculating Q for mode {:>5d} {:} {:>5d}'.format(n[i], mode_type, l[i]))
        Q[i] = calculate_Q_wrapper(run_info, model, mode_type, n[i], l[i], omega_rad_per_s[i], neglect_ka = neglect_ka)

    # Calculate frequency shift.
    # Dahlen and Tromp (9.55).
    omega_ratio = omega_rad_per_s/omega_ref_rad_per_s
    delta_omega_rad_per_s = omega_rad_per_s*np.log(omega_ratio)/(np.pi*Q)
    # Fudge -- need to work out.
    #delta_omega = delta_omega*3.0
    omega_rad_per_s_new = omega_rad_per_s + delta_omega_rad_per_s
    omega_new_mHz = omega_rad_per_s_new*1.0E3/(2.0*np.pi)

    # Re-write the eigenvalue file.
    # Generate the name of the output directory based on the Ouroboros
    # parameters.
    _, _, _, dir_eigval  = get_Ouroboros_out_dirs(run_info, mode_type)
    path_eigenvalues = os.path.join(dir_eigval, 'eigenvalues.txt')
    fmt = '{:>10d} {:>10d} {:>19.12e} {:>19.12e} {:>19.12e}\n'
    print('Re-writing {:}'.format(path_eigenvalues))
    with open(path_eigenvalues, 'w') as out_id:

        for i in range(num_modes):

            out_id.write(fmt.format(n[i], l[i], omega_uncorrected_mHz[i], omega_new_mHz[i], Q[i]))

    return

def main():

    # Parse input arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_input_file", help = "File path (relative or absolute) to Ouroboros input file.")
    input_args = parser.parse_args()
    path_input = input_args.path_to_input_file

    # Read the input file.
    run_info = read_Ouroboros_input_file(path_input)

    # Apply attenuation correction.
    for mode_type in run_info['mode_types']:

        apply_atten_correction_all_modes(run_info, mode_type)
    
    return

if __name__ == '__main__':

    main()
