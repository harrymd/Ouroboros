'''
Calculates the mean excitation coefficients of a given mode due to a 
given source.
'''

import argparse
from glob import glob
import os

import numpy as np
import pandas

from Ouroboros.common import read_input_file, read_Ouroboros_summation_input_file
from Ouroboros.misc.cmt_io import read_mineos_cmt
from Ouroboros.summation.run_summation import (calculate_source_coefficients_spheroidal,
        get_eigenfunc, get_output_dir_info, get_surface_gravity, load_mode_info,
        mode_int_to_type, mode_type_to_int)

def calculate_mean_coeff_spheroidal(source_coeffs, l, Ur):
    '''
    Calculates the mean excitation coefficient for a given spheroidal mode.
    See Ouroboros/doc/summation_notes.pdf
    '''

    # Unpack source coeffs.
    A0 = source_coeffs['A0']
    A1 = source_coeffs['A1']
    B1 = source_coeffs['B1']
    A2 = source_coeffs['A2']
    B2 = source_coeffs['B2']

    # Evaluate terms dependent on source coefficients.
    t0 = 2.0*(A0**2.0)
    t1 = l*(l + 1)*((A1**2.0) + (B1**2.0))
    t2 = (l - 1)*l*(l + 1)*(l + 2)*((A2**2.0) + (B2**2.0))
    #
    summ = t0 + t1 + t2

    # Evaluate pre-factors.
    p0 = ((2.0*l) + 1)*np.abs(Ur)/(8.0*(np.pi**(3.0/2.0)))
    p1 = (2.0*np.pi)/((2.0*l) + 1)

    # Evaluate mean coefficient.
    integral = p1*summ
    coeff = p0*np.sqrt(integral)

    return coeff

def calculate_mean_excitation(run_info, summation_info, overwrite = False, response_correction_params = None, path_suffix = ''):
    '''
    Calculate the mean excitation coefficients of all modes due to a given
    source.
    '''

    # Receiver depth (km).
    z_receiver = 0.0

    # Get name of output file.
    name_coeffs_data_frame = 'coeffs_mean{:}.pkl'.format(path_suffix)
    name_modes_data_frame = 'modes_mean{:}.pkl'.format(path_suffix)
    #
    path_coeffs_data_frame = os.path.join(summation_info['dir_output'], name_coeffs_data_frame)
    path_modes_data_frame = os.path.join(summation_info['dir_output'], name_modes_data_frame)
    #
    paths = [path_coeffs_data_frame, path_modes_data_frame]
    #
    paths_exist = [os.path.exists(path) for path in paths]

    if all(paths_exist) and (not overwrite):

        print('Mean coefficient output files already exist.')

        print('Loading {:}'.format(path_coeffs_data_frame))
        coeffs = pandas.read_pickle(path_coeffs_data_frame)

        print('Loading {:}'.format(path_modes_data_frame))
        modes = pandas.read_pickle(path_modes_data_frame)

        return coeffs, modes

    # Load CMT information.
    cmt = read_mineos_cmt(summation_info['path_cmt'])
    print('Event: {:>10}, lon. : {:>+8.2f}, lat.  {:>+8.2f}, depth {:>9.2f} km'.format(cmt['ev_id'],
            cmt['lon_centroid'], cmt['lat_centroid'], cmt['depth_centroid']))

    # Load mode information.
    mode_info = load_mode_info(run_info, summation_info)
    if summation_info['path_mode_list'] is not None:

        mode_info = filter_mode_list(mode_info, summation_info['path_mode_list'])

    num_modes_dict = dict()
    for mode_type in summation_info['mode_types']:

        num_modes_dict[mode_type] = len(mode_info[mode_type]['f'])

    num_modes_total = sum([num_modes_dict[mode_type] for mode_type in num_modes_dict])

    # Calculate maximum l-value. 
    l_max = np.max([np.max(mode_info[mode_type]['l']) for mode_type in mode_info])
    print('Maximum l-value: {:>5d}'.format(l_max))

    # Create output arrays.
    type_list    = np.zeros(num_modes_total, dtype = np.int)
    n_list       = np.zeros(num_modes_total, dtype = np.int)
    l_list       = np.zeros(num_modes_total, dtype = np.int)
    f_list       = np.zeros(num_modes_total, dtype = np.float)
    Q_list       = np.zeros(num_modes_total, dtype = np.float)
    A_r_list     = np.zeros(num_modes_total, dtype = np.float) 
    #A_Theta_list = np.zeros(num_modes_total, dtype = np.float)
    #A_Phi_list   = np.zeros(num_modes_total, dtype = np.float)

    i_offset = 0
    for mode_type in summation_info['mode_types']:

        num_modes = num_modes_dict[mode_type]
        print('Mode type: {:>3}, mode count: {:>5d}'.format(mode_type, num_modes))

        for i in range(num_modes):
            
            # Unpack.
            n = mode_info[mode_type]['n'][i]
            l = mode_info[mode_type]['l'][i]
            f = mode_info[mode_type]['f'][i]
            f_rad_per_s = f*1.0E-3*2.0*np.pi
            Q = mode_info[mode_type]['Q'][i]

            if i == (num_modes - 1):

                str_end = '\n'

            else:

                str_end = '\r'

            print('Mode: {:>5d} of {:>5d}, n = {:>5d}, l = {:>5d}, f = {:>7.3f} mHz'.format(
                    i + 1, num_modes, n, l, f), end = str_end)
            
            # Load the eigenfunction information interpolated
            # at the source and receiver locations.
            # Also get the planet radius.
            eigfunc_source, eigfunc_receiver, r_planet = \
                get_eigenfunc(run_info, mode_type,
                    n,
                    l,
                    f_rad_per_s,
                    cmt['depth_centroid']*1.0E3, # km to m.
                    z_receiver = 0.0,
                    response_correction_params = response_correction_params)
            
            if (i == 0) & (i_offset == 0):
            
                # Calculate radial coordinate of event (km).
                cmt['r_centroid'] = r_planet*1.0E-3 - cmt['depth_centroid']
            
            # Calculate the coefficients.
            if mode_type == 'S':

                # Excitation coefficients determined by source location.
                source_coeffs = \
                    calculate_source_coefficients_spheroidal(
                        l, f_rad_per_s, cmt, eigfunc_source,
                        summation_info['pulse_type'])
                
                # Mean excitation coefficient.
                mean_coeff = calculate_mean_coeff_spheroidal(source_coeffs, l,
                                eigfunc_receiver['U'])

            else:

                raise NotImplementedError

            # Store output.
            type_list[i + i_offset]     = mode_type_to_int[mode_type]
            n_list[i + i_offset]        = n
            l_list[i + i_offset]        = l 
            f_list[i + i_offset]        = f
            Q_list[i + i_offset]        = Q 

            A_r_list[i + i_offset]       = mean_coeff
            #A_Theta_list[i + i_offset]  = coeffs['Theta']
            #A_Phi_list[i + i_offset]    = coeffs['Phi']

        i_offset = i_offset + i + 1

    # Store mode data.
    mode_data_frame = pandas.DataFrame(
            {'type' : [mode_int_to_type[x] for x in type_list],
            'n'     : n_list,
            'l'     : l_list,
            'f'     : f_list,
            'Q'     : Q_list},
            columns = ['type', 'n', 'l', 'f', 'Q'])

    # Store coefficient data.
    coeff_data_frame = pandas.DataFrame(
            {'A_r' : A_r_list})

    # Save output.
    print('Saving coefficients to {:}'.format(path_coeffs_data_frame))
    coeff_data_frame.to_pickle(path_coeffs_data_frame)
    #
    print('Saving mode information to {:}'.format(path_modes_data_frame))
    mode_data_frame.to_pickle(path_modes_data_frame)

    return coeff_data_frame, mode_data_frame 

def calculate_mean_excitation_multi_event(run_info, summation_info, overwrite = False, response_correction_params = None):
    '''
    Calculates the mean excitation coefficients for more than one event.
    In the summation input, specify path_cmt as a directory, and give
    each event file in the format *.txt.
    '''

    cmt_list_regex = os.path.join(summation_info['path_cmt'], '*.txt')
    cmt_path_list = glob(cmt_list_regex)
    cmt_path_list.sort()

    n_events = len(cmt_path_list)
    for i in range(n_events):
        
        print(i)
        summation_info['path_cmt'] = cmt_path_list[i]

        calculate_mean_excitation(run_info, summation_info,
                    overwrite = overwrite,
                    response_correction_params = response_correction_params,
                    path_suffix = '_{:>05d}'.format(i))

    return

def main():
    '''
    Main script for calculating mean excitation coefficient.
    '''

    # Parse input arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("path_mode_input", help = "File path (relative or absolute) to Ouroboros mode input file.")
    parser.add_argument("path_summation_input", help = "File path (relative or absolute) to Ouroboros summation input file.")
    parser.add_argument("--multi_event", action = 'store_true', help = 'Loop over a list of events.')
    #
    input_args = parser.parse_args()
    #
    path_mode_input = input_args.path_mode_input
    path_summation_input = input_args.path_summation_input
    multi_event = input_args.multi_event

    # Read the mode input file.
    run_info = read_input_file(path_mode_input)

    # Read the summation input file.
    summation_info = read_Ouroboros_summation_input_file(path_summation_input)

    # Get output directory information.
    run_info, summation_info = get_output_dir_info(run_info, summation_info)

    # If necessary, calculate the surface gravity.
    if summation_info['correct_response']:

        g = get_surface_gravity(run_info)
        response_correction_params = dict()
        response_correction_params['g'] = g

    else:

        response_correction_params = None

    if multi_event:

        # Calculate mean excitation.
        calculate_mean_excitation_multi_event(run_info, summation_info,
                    overwrite = False,
                    response_correction_params = response_correction_params)
        
    else:

        # Calculate mean excitation.
        calculate_mean_excitation(run_info, summation_info,
                    overwrite = False,
                    response_correction_params = response_correction_params)

    return

if __name__ == '__main__':

    main()
