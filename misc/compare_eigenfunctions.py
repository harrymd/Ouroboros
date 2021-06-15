'''
Compares eigenfunctions from two different calculations.
'''

import argparse
import os

import numpy as np

from Ouroboros.common import (align_mode_lists, get_Mineos_out_dirs,
        get_Ouroboros_out_dirs, get_r_fluid_solid_boundary,
        interp_n_parts, load_eigenfreq, load_eigenfunc, load_model,
        read_input_file)

def get_eigfunc_comparison_out_path(run_info_A, run_info_B, mode_type):
    '''
    Find the path to the output file of the comparison.
    '''

    if run_info_A['code'] == 'ouroboros':

        _, _, _, dir_output = get_Ouroboros_out_dirs(run_info_A, mode_type)

    elif run_info_A['code'] == 'mineos':

        _, dir_output = get_Mineos_out_dirs(run_info_A)

    else:

        raise ValueError

    file_out = 'eigfunc_compare_with_{:}_{:}_{:>05d}_{:>05d}_{:1d}.txt'.format(
                run_info_B['code'], run_info_B['name_model'], run_info_B['n_lims'][1],
                run_info_B['l_lims'][1], run_info_B['grav_switch'])
    path_out = os.path.join(dir_output, file_out)

    return path_out

def comparison_wrapper(run_info_A, run_info_B, mode_type, n, l, om_A, om_B, norm_args, i_toroidal = None):
    '''
    Handle the comparison for different mode types.
    '''
    
    if mode_type == 'S':

        rms = comparison_wrapper_S(run_info_A, run_info_B, n, l, om_A, om_B,
                                    norm_args)

    elif mode_type == 'R':

        rms = comparison_wrapper_R(run_info_A, run_info_B, n, om_A, om_B,
                                    norm_args)

    elif mode_type == 'T':

        rms = comparison_wrapper_T(run_info_A, run_info_B, n, l, om_A, om_B,
                                    norm_args, i_toroidal = i_toroidal)

    elif mode_type == 'I':

        raise NotImplementedError

    else:

        raise ValueError

    return rms

def check_sign_S(r, U, V):
    '''
    Use a sign convention to determine the sign of a spheroidal mode.
    '''

    # Calculate sign.
    iU = np.trapz(r*U, x = r)
    #iV = np.trapz(r*V, x = r)

    sign = np.sign(iU)

    return sign

def check_sign_P(r, P):
    '''
    Use a sign convention to determine the sign of a potential eigenfunction.
    '''

    # Calculate sign.
    iP = np.trapz(P, x = r)

    sign = np.sign(iP)

    return sign

def check_sign_R(r, U):
    '''
    Use a sign convention to determine the sign of a radial eigenfunction.
    '''

    # Calculate sign.
    iU = np.trapz(r*U, x = r)

    sign = np.sign(iU)

    return sign

def check_sign_T(r, W):
    '''
    Use a sign convention to determine the sign of a toroidal mode.
    '''

    # Calculate sign.
    iW = np.trapz(r*W, x = r)

    sign = np.sign(iW)

    return sign

def comparison_wrapper_S(run_info_A, run_info_B, n, l, om_A, om_B, norm_args):
    '''
    Compare a spheroidal mode from two different calculations.
    '''

    # Load eigenfunctions.
    norm_args['omega'] = om_A
    eigfunc_dict_A = load_eigenfunc(run_info_A, 'S', n, l, norm_args = norm_args)
    #
    norm_args['omega'] = om_B
    eigfunc_dict_B = load_eigenfunc(run_info_B, 'S', n, l, norm_args = norm_args)

    # Check sign and equalise.
    sign_A = check_sign_S(eigfunc_dict_A['r'], eigfunc_dict_A['U'], eigfunc_dict_A['V'])
    sign_B = check_sign_S(eigfunc_dict_B['r'], eigfunc_dict_B['U'], eigfunc_dict_B['V'])
    if sign_A != sign_B:

        for var in ['U', 'V']:

            eigfunc_dict_B[var] = eigfunc_dict_B[var]*-1.0

    # Assume there are only two major discontinuities and they are the first
    # ones.
    i_discon_A = np.where(np.diff(eigfunc_dict_A['r']) == 0.0)[0] + 1
    i_discon_B = np.where(np.diff(eigfunc_dict_B['r']) == 0.0)[0] + 1

    if len(i_discon_A) > 2:

        i_discon_A = i_discon_A[0:2]

    if len(i_discon_B) > 2:

        i_discon_B = i_discon_B[0:2]

    # Re-grid.
    n_pts_A = len(eigfunc_dict_A['r'])
    n_pts_B = len(eigfunc_dict_B['r'])
    if n_pts_A >= n_pts_B:
        
        for var in ['U', 'V']:

            eigfunc_dict_B[var] = interp_n_parts(eigfunc_dict_A['r'],
                                    eigfunc_dict_B['r'], eigfunc_dict_B[var],
                                    i_discon_A, i_discon_B)

        eigfunc_dict_B['r'] = eigfunc_dict_A['r']

    else: 

        for var in ['U', 'V']:

            eigfunc_dict_A[var] = interp_n_parts(eigfunc_dict_B['r'],
                                    eigfunc_dict_A['r'], eigfunc_dict_A[var],
                                    i_discon_B, i_discon_A)

        eigfunc_dict_A['r'] = eigfunc_dict_B['r']

    # Calculate the RMS difference.
    r_range = eigfunc_dict_A['r'][-1] - eigfunc_dict_A['r'][0]
    
    U_diff = eigfunc_dict_A['U'] - eigfunc_dict_B['U']
    rms_diff_U = np.sqrt(np.trapz(U_diff**2.0, x = eigfunc_dict_A['r'])/r_range)

    V_diff = eigfunc_dict_A['V'] - eigfunc_dict_B['V']
    rms_diff_V = np.sqrt(np.trapz(V_diff**2.0, x = eigfunc_dict_A['r'])/r_range)

    rms_diff = rms_diff_U + rms_diff_V

    rms_U_A = np.sqrt(np.trapz( eigfunc_dict_A['U']**2.0,
                                x = eigfunc_dict_A['r'])/r_range)
    rms_V_A = np.sqrt(np.trapz( eigfunc_dict_A['V']**2.0,
                                x = eigfunc_dict_A['r'])/r_range)
    rms_A = rms_U_A + rms_V_A

    rms_U_B = np.sqrt(np.trapz( eigfunc_dict_B['U']**2.0,
                                x = eigfunc_dict_B['r'])/r_range)
    rms_V_B = np.sqrt(np.trapz( eigfunc_dict_B['V']**2.0,
                                x = eigfunc_dict_B['r'])/r_range)
    rms_B = rms_U_B + rms_V_B
    
    return rms_diff, rms_A, rms_B

def comparison_wrapper_R(run_info_A, run_info_B, n, om_A, om_B, norm_args):
    '''
    Compare a radial mode from two different calculations.
    '''
    
    mode_type = 'R'
    l = 0

    # Load eigenfunctions.
    norm_args['omega'] = om_A
    eigfunc_dict_A = load_eigenfunc(run_info_A, mode_type, n, l,
                                    norm_args = norm_args)
    #
    norm_args['omega'] = om_B
    eigfunc_dict_B = load_eigenfunc(run_info_B, mode_type, n, l,
                                    norm_args = norm_args)

    # Check sign and equalise.
    sign_A = check_sign_R(eigfunc_dict_A['r'], eigfunc_dict_A['U'])
    sign_B = check_sign_R(eigfunc_dict_B['r'], eigfunc_dict_B['U'])
    if sign_A != sign_B:

        for var in ['U']:

            eigfunc_dict_B[var] = eigfunc_dict_B[var]*-1.0

    # Assume there are only two major discontinuities and they are the first
    # ones.
    i_discon_A = np.where(np.diff(eigfunc_dict_A['r']) == 0.0)[0] + 1
    i_discon_B = np.where(np.diff(eigfunc_dict_B['r']) == 0.0)[0] + 1

    if len(i_discon_A) > 2:

        i_discon_A = i_discon_A[0:2]

    if len(i_discon_B) > 2:

        i_discon_B = i_discon_B[0:2]

    # Re-grid.
    n_pts_A = len(eigfunc_dict_A['r'])
    n_pts_B = len(eigfunc_dict_B['r'])
    if n_pts_A >= n_pts_B:
        
        for var in ['U']:

            eigfunc_dict_B[var] = interp_n_parts(eigfunc_dict_A['r'],
                                    eigfunc_dict_B['r'], eigfunc_dict_B[var],
                                    i_discon_A, i_discon_B)

        eigfunc_dict_B['r'] = eigfunc_dict_A['r']

    else: 

        for var in ['U']:

            eigfunc_dict_A[var] = interp_n_parts(eigfunc_dict_B['r'],
                                    eigfunc_dict_A['r'], eigfunc_dict_A[var],
                                    i_discon_B, i_discon_A)

        eigfunc_dict_A['r'] = eigfunc_dict_B['r']

    # Calculate the RMS difference.
    r_range = eigfunc_dict_A['r'][-1] - eigfunc_dict_A['r'][0]
    
    U_diff = eigfunc_dict_A['U'] - eigfunc_dict_B['U']
    rms_diff_U = np.sqrt(np.trapz(U_diff**2.0, x = eigfunc_dict_A['r'])/r_range)

    rms_diff = rms_diff_U

    rms_U_A = np.sqrt(np.trapz( eigfunc_dict_A['U']**2.0,
                                x = eigfunc_dict_A['r'])/r_range)
    rms_A = rms_U_A

    rms_U_B = np.sqrt(np.trapz( eigfunc_dict_B['U']**2.0,
                                x = eigfunc_dict_B['r'])/r_range)
    rms_B = rms_U_B

    return rms_diff, rms_A, rms_B

def comparison_wrapper_T(run_info_A, run_info_B, n, l, om_A, om_B, norm_args, i_toroidal):
    '''
    Compare a spheroidal mode from two different calculations.
    '''

    mode_type = 'T'

    # Load eigenfunctions.
    norm_args['omega'] = om_A
    eigfunc_dict_A = load_eigenfunc(run_info_A, mode_type, n, l,
                        norm_args = norm_args, i_toroidal = i_toroidal)
    #
    norm_args['omega'] = om_B
    eigfunc_dict_B = load_eigenfunc(run_info_B, mode_type, n, l,
                        norm_args = norm_args, i_toroidal = i_toroidal)

    # Check sign and equalise.
    sign_A = check_sign_T(eigfunc_dict_A['r'], eigfunc_dict_A['W'])
    sign_B = check_sign_T(eigfunc_dict_B['r'], eigfunc_dict_B['W'])
    #
    if sign_A != sign_B:

        eigfunc_dict_B['W'] = (eigfunc_dict_B['W'] * -1.0)

    # Get locations of discontinuities.
    model_A = load_model(run_info_A['path_model'])
    _, _, i_discon_B = \
            get_r_fluid_solid_boundary(model_A['r'], model_A['v_s'])

    model_B = load_model(run_info_B['path_model'])
    _, _, i_discon_A = \
            get_r_fluid_solid_boundary(model_B['r'], model_B['v_s'])

    assert len(i_discon_A) == len(i_discon_B), \
            'Number of discontinuities must be the same for both models.'

    # Re-grid.
    n_pts_A = len(eigfunc_dict_A['r'])
    n_pts_B = len(eigfunc_dict_B['r'])
    if n_pts_A >= n_pts_B:
        
        for var in ['W']:

            eigfunc_dict_B[var] = interp_n_parts(eigfunc_dict_A['r'],
                                    eigfunc_dict_B['r'], eigfunc_dict_B[var],
                                    i_discon_A, i_discon_B)

        eigfunc_dict_B['r'] = eigfunc_dict_A['r']

    else: 

        for var in ['W']:

            eigfunc_dict_A[var] = interp_n_parts(eigfunc_dict_B['r'],
                                    eigfunc_dict_A['r'], eigfunc_dict_A[var],
                                    i_discon_B, i_discon_A)

        eigfunc_dict_A['r'] = eigfunc_dict_B['r']

    # Calculate the RMS difference.
    r_range = eigfunc_dict_A['r'][-1] - eigfunc_dict_A['r'][0]
    
    W_diff = eigfunc_dict_A['W'] - eigfunc_dict_B['W']
    rms_diff_W = np.sqrt(np.trapz(W_diff**2.0, x = eigfunc_dict_A['r'])/r_range)

    rms_diff = rms_diff_W

    rms_W_A = np.sqrt(np.trapz( eigfunc_dict_A['W']**2.0,
                                x = eigfunc_dict_A['r'])/r_range)
    rms_A = rms_W_A

    rms_W_B = np.sqrt(np.trapz( eigfunc_dict_B['W']**2.0,
                                x = eigfunc_dict_B['r'])/r_range)
    rms_B = rms_W_B

    return rms_diff, rms_A, rms_B

def main():

    # Read input arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("path_input_A", help = "File path (relative or absolute) to first input file.")
    parser.add_argument("path_input_B", help = "File path (relative or absolute) to second input file.")
    parser.add_argument("--i_toroidal", type = int, help = "Specify layer number (required for toroidal modes).")
    #
    args = parser.parse_args()
    #
    path_input_A = args.path_input_A
    path_input_B = args.path_input_B
    i_toroidal = args.i_toroidal

    # Read input files.
    run_info_A = read_input_file(path_input_A)
    run_info_B = read_input_file(path_input_B)

    # Find common mode types.
    mode_types = [mode_type for mode_type in run_info_A['mode_types'] if mode_type in run_info_B['mode_types']]
    assert len(mode_types) > 0, 'The two input files have no mode types in common.'

    # Define normalisation to be used for comparison.
    #norm_args = {'norm_func' : 'DT', 'units' : 'mineos'}
    norm_args = {'norm_func' : 'mineos', 'units' : 'mineos'}

    # Define format of output file.
    out_fmt = '{:>5d} {:>5d} {:>19.12f} {:>19.12f} {:>18.12e} {:>18.12e} {:>18.12e}\n'

    # Loop over mode types.
    for mode_type in ['I']:

        if mode_type in mode_types:

            mode_types.remove(mode_type)

    for mode_type in mode_types:

        # Found output path.
        path_out = get_eigfunc_comparison_out_path(run_info_A, run_info_B, mode_type)

        # Load mode lists.
        mode_info_A = load_eigenfreq(run_info_A, mode_type, i_toroidal = i_toroidal)
        mode_info_B = load_eigenfreq(run_info_B, mode_type, i_toroidal = i_toroidal)

        # Find common modes.
        n, l, i_align_A, i_align_B = align_mode_lists(mode_info_A['n'],
                                        mode_info_A['l'], mode_info_B['n'],
                                        mode_info_B['l'])
        f_A = mode_info_A['f'][i_align_A]
        f_B = mode_info_B['f'][i_align_B]

        # Convert from mHz to rad/s.
        om_A = f_A*2.0*np.pi*1.0E-3
        om_B = f_B*2.0*np.pi*1.0E-3

        # Loop over common modes.
        num_modes = len(l)
        rms_diff = np.zeros(num_modes)
        rms_A = np.zeros(num_modes)
        rms_B = np.zeros(num_modes)
        for i in range(num_modes):

            rms_diff[i], rms_A[i], rms_B[i] = \
                comparison_wrapper(run_info_A, run_info_B, mode_type,
                                n[i], l[i], om_A[i], om_B[i], norm_args,
                                i_toroidal = i_toroidal)

        # Save.
        print('Writing to {:}'.format(path_out))
        with open(path_out, 'w') as out_id:

            for i in range(num_modes):

                out_id.write(out_fmt.format(n[i], l[i], f_A[i], f_B[i], rms_diff[i], rms_A[i], rms_B[i]))

    return

if __name__ == '__main__':

    main()
