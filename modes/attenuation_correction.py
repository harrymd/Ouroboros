'''
Scripts for correcting models and eigenfrequencies due to the effects of
attenuation.
'''

import argparse
import os
from shutil import copyfile

import numpy as np

from Ouroboros.common import (get_path_adjusted_model, load_eigenfreq_Ouroboros,
        load_kernel, load_model_full, mkdir_if_not_exist,
        get_Ouroboros_out_dirs, read_Ouroboros_input_file)

def create_adjusted_model(run_info):
    '''
    Anelasticity causes dispersion so the elastic parameters are effectively
    reduced at frequencies below the reference frequency of the model.
    We use the constant-Q equations from
    (Equations in Dahlen and Tromp, 1998, p. 350.)
    '''

    # Load the reference model.
    model = load_model_full(run_info['path_model'])

    # Get target and reference frequencies.
    f_target_rad_per_s = run_info['f_target_mHz']*1.0E-3*2.0*np.pi
    f_ref_Hz = (1.0/model['T_ref'])
    f_ref_mHz = f_ref_Hz*1.0E3
    f_ref_rad_per_s = f_ref_Hz*2.0*np.pi
    
    # Calculate dispersion corrections.
    d_mu = dispersion_correction(model['mu'], model['Q_mu'],
                f_target_rad_per_s, f_ref_rad_per_s)
    mu_new = model['mu'] + d_mu

    d_ka = dispersion_correction(model['ka'], model['Q_ka'],
                f_target_rad_per_s, f_ref_rad_per_s)
    ka_new = model['ka'] + d_ka

    # Calculate P- and S-wave speeds.
    # (Equations in Dahlen and Tromp, 1998, p. 350.)
    alpha_new = np.sqrt((ka_new + (4.0/3.0)*mu_new)/model['rho'])
    beta_new = np.sqrt(mu_new/model['rho'])
    
    # Store new model.
    model_new = dict()
    # Some parameters do not change.
    var_unchanged = ['r', 'rho', 'Q_mu', 'Q_ka', 'n_layers', 'i_icb', 'i_cmb']
    for var in var_unchanged:

        model_new[var] = model[var]

    # Store other parameters.
    # Note: Here assume isotropic wavespeeds.
    model_new['v_pv'] = alpha_new
    model_new['v_ph'] = alpha_new
    model_new['v_sv'] = beta_new
    model_new['v_sh'] = beta_new
    model_new['eta']  = np.zeros(model['n_layers']) + 1.0
    #
    model_new['f_ref_Hz'] = run_info['f_target_mHz']*1.0E-3

    # Get path out and header string.
    path_out = get_path_adjusted_model(run_info)

    #
    header_str = 'Model {:} adjusted from freq. {:>8.2f} to {:>8.2f} mHz' \
                    .format(run_info['name_model'], f_ref_mHz,
                            run_info['f_target_mHz'])
    assert len(header_str) <= 80

    # Write the new model.
    write_model(model_new, path_out, header_str)

    return

def dispersion_correction(x, Q_x, omega, omega_ref):
    '''
    Dahlen and Tromp (1998) eq. 9.50-9.51

    x           Shear or bulk modulus.
    Q_x         Shear or bulk Q-factor.
    omega       Frequency (rad/s).
    omega_ref   Reference frequency (rad/s).
    '''

    # Equation 9.50/9.51.
    d_x = (2.0 * x * np.log(omega / omega_ref)) / (np.pi * Q_x)

    # No perturbation for zero-Q regions (usually this indicates infinite
    # Q, i.e. no attenuation, so no correction is necessary).
    i_bad = np.where(Q_x == 0.0)[0]
    d_x[i_bad] = 0.0

    return d_x

def calculate_Q_wrapper(run_info, model, mode_type, n, l, f_rad_per_s, neglect_ka = False):
    '''
    Calculates the Q-factor of a mode using D&T 1998 eq. 9.54.
    '''

    # Load kernel.
    r, K_ka, K_mu = load_kernel(run_info, mode_type, n, l, units = 'SI')

    # Interpolate the model at the eigenfunction sample points.
    model_interpolated = dict()
    model_interpolated['T_ref'] = model['T_ref']
    model_interpolated['f_ref_rad_per_s'] = 2.0*np.pi/model_interpolated['T_ref']
    model_interpolated['r'] = r
    for key in ['mu', 'ka', 'Q_mu', 'Q_ka']:

        model_interpolated[key] = np.interp(r, model['r'], model[key])

    model  = model_interpolated

    # Evaluate the integral (D&T 1998 eq. 9.54).
    # Define the integrand.
    I_mu = (model['mu']*K_mu/model['Q_mu'])
    i_fluid = np.where(model['mu'] == 0.0)[0]
    I_mu[i_fluid] = 0.0
    
    if neglect_ka:

        integrand = I_mu
    
    # Optionally, include the kappa part of the integrand.
    else:

        I_ka = (model['ka']*K_ka/model['Q_ka']) 
        integrand = I_mu + I_ka

    # Integrate using the trapezium rule.
    integral = np.trapz(integrand, x = r)
    inv_Q = 2.0*integral/f_rad_per_s
    Q = 1.0/inv_Q
    Q = Q/(2.0*np.pi)
    
    # If requested, visualise the integration.
    plotting = False 
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

    return Q

def apply_atten_correction_all_modes(run_info, mode_type):
    '''
    A wrapper for applying an attenuation correction to the mode frequencies.
    '''

    # Load the planetary model.
    model_path = get_path_adjusted_model(run_info)
    model = load_model_full(model_path)
    # Throw away unnecessary items.
    keys_to_keep = ['T_ref', 'r', 'mu', 'ka', 'Q_mu', 'Q_ka']
    model_new = {key : model[key] for key in keys_to_keep}
    model = model_new

    # Convert period to rad/s.
    omega_ref_Hz = 1.0/model['T_ref']
    omega_ref_rad_per_s = 2.0*np.pi*omega_ref_Hz

    # Decide whether to neglect bulk attenuation.
    #neglect_ka = True
    neglect_ka = False
    
    # Define out file format.
    out_fmt = '{:>5d} {:>5d} {:>19.12e} {:>12.6e} {:>19.12e} {:>19.12e}'

    # Load mode database.
    mode_info = load_eigenfreq_Ouroboros(run_info, mode_type)
    n = mode_info['n']
    l = mode_info['l']
    omega_uncorrected_mHz = mode_info['f_0']
    omega_rad_per_s = omega_uncorrected_mHz*1.0E-3*2.0*np.pi

    # Prepare output array.
    num_modes = len(n)
    Q = np.zeros(num_modes)
    
    # Testing a single mode.
    single_mode = False
    if single_mode:

        n_q =  8 
        l_q =  2 
        j = np.where((n == n_q) & (l == l_q))[0][0]
        calculate_Q_wrapper(run_info, model, mode_type, n_q, l_q, omega_rad_per_s[j], neglect_ka = neglect_ka) 

        return

    # Loop over modes.
    for i in range(num_modes):
        
        print('Calculating Q for mode {:>5d} {:} {:>5d}'.format(
                n[i], mode_type, l[i]))
        Q[i] = calculate_Q_wrapper(run_info, model, mode_type, n[i], l[i],
                omega_rad_per_s[i], neglect_ka = neglect_ka)

    # Calculate frequency shift.
    # Dahlen and Tromp (9.55).
    omega_ratio = omega_rad_per_s/omega_ref_rad_per_s
    delta_omega_rad_per_s = omega_rad_per_s*np.log(omega_ratio)/(np.pi*Q)
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

            out_id.write(fmt.format(n[i], l[i], omega_uncorrected_mHz[i],
                        omega_new_mHz[i], Q[i]))

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
