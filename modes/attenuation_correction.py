import argparse
import os
from shutil import copyfile

import numpy as np

from Ouroboros.common import load_eigenfreq_Ouroboros, load_model_full, mkdir_if_not_exist, get_Ouroboros_out_dirs, read_Ouroboros_input_file

def relabel_modes(dir_output, n_orig, l, f_new, Q_new):

    l_vals = np.sort(np.unique(l))

    num_modes = len(f_new)
    n_old_list = np.zeros(num_modes, dtype = np.int)
    n_new_list = np.zeros(num_modes, dtype = np.int)
    l_list = np.zeros(num_modes, dtype = np.int)
    f_new_list = np.zeros(num_modes)
    Q_new_list = np.zeros(num_modes)
    
    k = 0
    for l_i in l_vals:

        i = np.where(l == l_i)[0]
        
        n_orig_i = n_orig[i]
        print(n_orig_i)
        n_min_orig_i = np.min(n_orig_i)
        
        num_i = len(i)
        n_new_i = np.array(range(n_min_orig_i, n_min_orig_i + num_i), dtype = np.int)
        l_new_i = np.zeros(num_i, dtype = np.int) + l_i

        j = np.argsort(f_new[i])
        n_orig_i_sorted = n_orig_i[j]
        f_new_i_sorted = f_new[i][j]
        Q_new_i_sorted = Q_new[i][j]

        l_list    [k : k + num_i] = l_i
        n_new_list[k : k + num_i] = n_new_i
        n_old_list[k : k + num_i] = n_orig_i_sorted
        f_new_list[k : k + num_i] = f_new_i_sorted
        Q_new_list[k : k + num_i] = Q_new_i_sorted

        k = k + num_i
    
    fmt = '{:>5d} {:>5d} {:>5d} {:>8.3f} {:}'
    dir_eigenfunctions_old = os.path.join(dir_output, 'eigenfunctions')
    dir_eigenfunctions_new = os.path.join(dir_output, 'eigenfunctions_relabelled')
    mkdir_if_not_exist(dir_eigenfunctions_new)
    for k in range(num_modes):

        #print(fmt.format(n_old_list[k], n_new_list[k], l_list[k], f_new_list[k], n_old_list[k] == n_new_list[k]))

        if n_new_list[k] != n_old_list[k]:

            print('{:>3d} S {:>3d} --> {:>3d} S {:>3d}'.format(n_new_list[k], l_list[k],  n_old_list[k], l_list[k]))

        name_eigfunc_old = '{:>05d}_{:>05d}.npy'.format(n_old_list[k], l_list[k])
        path_eigfunc_old = os.path.join(dir_eigenfunctions_old, name_eigfunc_old)
        #
        name_eigfunc_new = '{:>05d}_{:>05d}.npy'.format(n_new_list[k], l_list[k])
        path_eigfunc_new = os.path.join(dir_eigenfunctions_new, name_eigfunc_new)
        
        #print('cp {:} {:}'.format(path_eigfunc_old, path_eigfunc_new))
        copyfile(path_eigfunc_old, path_eigfunc_new)

    path_eigvals_new = os.path.join(dir_output, 'eigenvalues_relabelled.txt')
    print("Writing {:}".format(path_eigvals_new))
    fmt = '{:>10d} {:>10d} {:>18.12f} {:>19.12e}'
    with open(path_eigvals_new, 'w') as out_id:

        for k in range(num_modes):

            out_id.write(fmt.format(n_new_list[k], l_list[k], f_new_list[k], Q_new_list[k]) + '\n') 

    return

def calculate_Q_wrapper(dir_kernels, model, mode_type, n, l, f_rad_per_s, neglect_ka = False):

    # Load kernel.
    name_kernel_file = 'kernels_{:>05d}_{:>05d}.npy'.format(n, l)
    path_kernel = os.path.join(dir_kernels, name_kernel_file)
    kernel_arr = np.load(path_kernel)
    # Unpack the array (r in m).
    r, K_ka, K_mu, K_rho, K_alpha, K_beta, K_rhop = kernel_arr
    # Convert to SI units.
    K_ka = K_ka*1.0E-6
    K_mu = K_mu*1.0E-6

    # Interpolate the model.
    model_interpolated = dict()
    model_interpolated['T_ref'] = model['T_ref']
    model_interpolated['r'] = r
    for key in ['mu', 'ka', 'Q_mu', 'Q_ka']:

        model_interpolated[key] = np.interp(r, model['r'], model[key])
    model  = model_interpolated
    
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
    
    plotting = False
    if plotting:

        import matplotlib.pyplot as plt

        fig, ax_arr = plt.subplots(1, 4, sharey = True, figsize = (14.0, 8.5), constrained_layout = True)

        ax = ax_arr[0]
        ax.plot(integrand, r, label = 'sum')
        ax.plot(I_mu, r, label = 'mu')
        ax.plot(I_ka, r, label = 'ka')

        ax = ax_arr[1]
        ax.plot(model['mu'], r, label = 'mu')
        ax.plot(model['ka'], r, label = 'ka')

        ax = ax_arr[2]
        #ax.plot(k_mu, r, label = 'k_mu')
        #ax.plot(k_ka, r, label = 'k_ka')
        ax.plot(model['mu']*K_mu, r, label = 'mu*K_mu')
        ax.plot(model['ka']*K_ka, r, label = 'ka*K_ka')

        ax = ax_arr[3]
        ax.plot(1.0/model['Q_mu'], r, label = '1.0/Q_mu')
        ax.plot(1.0/model['Q_ka'], r, label = '1.0/Q_ka')

        for ax in ax_arr:

            ax.legend()
        
        plt.show()

    return Q

def main():

    # Parse input arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_input_file", help = "File path (relative or absolute) to Ouroboros input file.")
    input_args = parser.parse_args()
    path_input = input_args.path_to_input_file

    # Read the input file.
    run_info = read_Ouroboros_input_file(path_input)

    # Load the planetary model.
    model = load_model_full(run_info['path_model'])
    # Throw away unnecessary items.
    keys_to_keep = ['T_ref', 'r', 'mu', 'ka', 'Q_mu', 'Q_ka']
    model_new = {key : model[key] for key in keys_to_keep}
    model = model_new

    f_ref = 1.0/model['T_ref']
    f_ref_rad_per_s = 2.0*np.pi*f_ref

    neglect_ka = False
    
    out_fmt = '{:>5d} {:>5d} {:>19.12e} {:>12.6e} {:>19.12e} {:>19.12e}'

    # Override attenuation flag.
    run_info['use_attenuation'] = False

    for mode_type in ['S']:

        # Get output directory.
        _, _, _, dir_output = \
            get_Ouroboros_out_dirs(run_info, mode_type)
        dir_kernels = os.path.join(dir_output, 'kernels')
        path_out = os.path.join(dir_output, 'Q_values.txt')

        # Load mode database.
        mode_info = load_eigenfreq_Ouroboros(run_info, mode_type)
        n = mode_info['n']
        l = mode_info['l']
        f = mode_info['f']

        f_rad_per_s = f*1.0E-3*2.0*np.pi
        num_modes = len(n)
        Q = np.zeros(num_modes)
        
        single_mode = False
        if single_mode:

            n_q =  9 
            l_q =  2 
            j = np.where((n == n_q) & (l == l_q))[0][0]
            calculate_Q_wrapper(dir_kernels, model, mode_type, n_q, n_q, f_rad_per_s[j], neglect_ka = neglect_ka) 

            return

        # Loop over modes.
        for i in range(num_modes):
            
            print('Calculating Q for mode {:>5d} {:} {:>5d}'.format(n[i], mode_type, l[i]))
            Q[i] = calculate_Q_wrapper(dir_kernels, model, mode_type, n[i], l[i], f_rad_per_s[i], neglect_ka = neglect_ka)

        # Calculate frequency shift.
        # Dahlen and Tromp (9.55).
        omega_ratio = f_rad_per_s/f_ref_rad_per_s
        delta_omega = f_rad_per_s*np.log(omega_ratio)/(np.pi*Q)
        # Fudge -- need to work out.
        #delta_omega = delta_omega*3.0
        omega_new = f_rad_per_s + delta_omega
        delta_omega_microHz = delta_omega*1.0E6/(2.0*np.pi)
        omega_new_mHz = omega_new*1.0E3/(2.0*np.pi)

        relabel_modes(dir_output, n, l, omega_new_mHz, Q)

        #with open(path_out, 'w') as out_id:

        #    for i in range(num_modes):

        #        out_id.write(out_fmt.format(n[i], l[i], f[i], Q[i],
        #                        delta_omega_microHz[i], omega_new_mHz[i]) + '\n')
        

    return

if __name__ == '__main__':

    main()
