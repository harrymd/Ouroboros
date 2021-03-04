import os
import argparse
from shutil import copyfile
import time

import numpy as np

from Ouroboros.common import get_Ouroboros_out_dirs, get_path_adjusted_model, read_Ouroboros_input_file, mkdir_if_not_exist, load_model_full
from Ouroboros.modes.compute_modes import radial_modes, spheroidal_modes, toroidal_modes
from Ouroboros.modes.calculate_potential import potential_all_modes
from Ouroboros.modes.calculate_gradient import gradient_all_modes

def dispersion_correction(x, Q_x, omega, omega_ref):
    '''
    Dahlen and Tromp (1998) eq. 9.50-9.51

    x           Shear or bulk modulus.
    Q_x         Shear or builk Q-factor.
    omega       Frequency (rad/s).
    omega_ref   Reference frequency (rad/s).
    '''

    d_x = (2.0 * x * np.log(omega / omega_ref)) / (np.pi * Q_x)

    i_bad = np.where(Q_x == 0.0)[0]
    d_x[i_bad] = 0.0

    return d_x

def create_adjusted_model(run_info):
    '''
    Anelasticity causes dispersion so the elastic parameters are effectively
    reduced at frequencies below the reference frequency of the model.
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
    write_adjusted_model(model_new, path_out, header_str)

    return

def write_adjusted_model(model, path_out, header_str):

    print("Writing to {:}".format(path_out))

    out_fmt = '{:>7.0f}. {:>8.2f} {:>8.2f} {:>8.2f} {:>8.1f} {:>8.1f} {:>8.2f} {:>8.2f} {:>8.5f}\n'

    with open(path_out, 'w') as out_id:

        out_id.write(header_str + '\n')
        out_id.write('{:>4d} {:>8.5f} {:>3d}\n'.format(0, model['f_ref_Hz'], 1))
        out_id.write('{:>6d} {:>3d} {:>3d}\n'.format(model['n_layers'], model['i_icb'],
                        model['i_cmb']))
        
        for i in range(model['n_layers']):

            out_id.write(out_fmt.format(
                model['r'][i], model['rho'][i], model['v_pv'][i], model['v_sv'][i],
                model['Q_ka'][i], model['Q_mu'][i], model['v_ph'][i],
                model['v_sh'][i], model['eta'][i]))

    return

def main():
    '''
    Control script for running RadialPNM.
    Run from the command line using
    python3 main.py
    in the directory containing this script.
    '''
    
    # Make announcement.
    print('Ouroboros')

    # Parse input arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_input_file", help = "File path (relative or absolute) to Ouroboros input file.")
    input_args = parser.parse_args()
    Ouroboros_input_file = input_args.path_to_input_file
    name_input = os.path.splitext(os.path.basename(Ouroboros_input_file))[0]

    # Read the input file.
    Ouroboros_info = read_Ouroboros_input_file(Ouroboros_input_file)

    if Ouroboros_info['use_attenuation']:

        create_adjusted_model(Ouroboros_info)

    # Set the 'g_switch' string: 0 -> noG, 1 -> G, 2 -> GP.
    g_switch_strs = ['noGP', 'G', 'GP']
    g_switch_str = g_switch_strs[Ouroboros_info['grav_switch']]

    Ouroboros_info['dirs_type'] = dict()

    for mode_type in Ouroboros_info['mode_types']:
        
        # Start timer.
        start_time = time.time()

        # Set the 'switch' string, e.g. 'S_GP'.
        if mode_type == 'T':

            switch = None 

        else:

            switch = '{:}_{:}'.format(mode_type, g_switch_str)

        Ouroboros_info['switch'] = switch

        dir_model, dir_run, dir_g, dir_type = get_Ouroboros_out_dirs(Ouroboros_info, mode_type)
        for dir_ in [dir_model, dir_run, dir_g, dir_type]:
            
            if dir_ is not None:
                
                mkdir_if_not_exist(dir_)

        ## Create the type directory if it doesn't exist.
        #if mode_type in ['R', 'S']:

        #    dir_g = os.path.join(dir_run, 'grav_{:1d}'.format(Ouroboros_info['g_switch']))
        #    mkdir_if_not_exist(dir_g)
        #    dir_type = os.path.join(dir_g, mode_type)

        #else:

        #    dir_type = os.path.join(dir_run, mode_type)

        #mkdir_if_not_exist(dir_type)
        Ouroboros_info['dirs_type'][mode_type] = dir_type

        # Copy the input file to the output directory.
        copyfile(Ouroboros_input_file, os.path.join(dir_type, name_input))

        # Run the code.
        if mode_type == 'T':

            toroidal_modes(Ouroboros_info)
        
        elif mode_type == 'R':
            
            radial_modes(Ouroboros_info)

        elif mode_type == 'S':

            spheroidal_modes(Ouroboros_info)

        else:

            raise ValueError

        elapsed_time = time.time() - start_time
        path_time = os.path.join(dir_type, 'time.txt')
        print('Time used: {:>.3f} s, saving time info to {:}'.format(elapsed_time, path_time))
        with open(path_time, 'w') as out_id:

            out_id.write('{:>.3f} seconds'.format(elapsed_time))
        
        # Calculate potential.
        potential_all_modes(Ouroboros_info, mode_type)

        # Calculate gradient.
        gradient_all_modes(Ouroboros_info, mode_type)

    return

if __name__ == '__main__':

    main()
