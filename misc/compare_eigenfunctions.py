import argparse
import os

import numpy as np

from Ouroboros.common import align_mode_lists, get_Mineos_out_dirs, get_Ouroboros_out_dirs, interp_n_parts, load_eigenfreq, load_eigenfunc, read_input_file

def get_eigfunc_comparison_out_path(run_info_A, run_info_B, mode_type):

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

def comparison_wrapper(run_info_A, run_info_B, mode_type, n, l, om_A, om_B, norm_args):
    
    if mode_type == 'S':

        rms = comparison_wrapper_S(run_info_A, run_info_B, n, l, om_A, om_B, norm_args)

    else:

        raise NotImplementedError

    return rms

def check_sign_S(r, U, V):

    # Calculate sign.
    iU = np.trapz(r*U, x = r)
    iV = np.trapz(r*V, x = r)

    #aU = np.abs(iU)
    #aV = np.abs(iV)

    #ratio = aU/aV

    #UV_angle = np.arctan2(iV, iU)

    #iS = (iU + iV)

    sign = np.sign(iU)

    return sign

def check_sign_P(r, P):

    # Calculate sign.
    iP = np.trapz(P, x = r)

    sign = np.sign(iP)

    return sign

def comparison_wrapper_S(run_info_A, run_info_B, n, l, om_A, om_B, norm_args):

    # Load eigenfunctions.
    norm_args['omega'] = om_A
    eigfunc_dict_A = load_eigenfunc(run_info_A, 'S', n, l, norm_args = norm_args)
    #
    norm_args['omega'] = om_B
    eigfunc_dict_B = load_eigenfunc(run_info_B, 'S', n, l, norm_args = norm_args)

    #for key in ['r', 'U', 'V']:

    #    print(key, np.all(eigfunc_dict_B[key] == eigfunc_dict_A[key]))

    # Check sign and equalise.
    sign_A = check_sign_S(eigfunc_dict_A['r'], eigfunc_dict_A['U'], eigfunc_dict_A['V'])
    sign_B = check_sign_S(eigfunc_dict_B['r'], eigfunc_dict_B['U'], eigfunc_dict_B['V'])
    if sign_A != sign_B:

        for var in ['U', 'V']:

            eigfunc_dict_B[var] = eigfunc_dict_B[var]*-1.0

    #for key in ['r', 'U', 'V']:

    #    print(key, np.all(eigfunc_dict_B[key] == eigfunc_dict_A[key]))

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

            #eigfunc_dict_B[var] = np.interp(eigfunc_dict_A['r'],
            #                        eigfunc_dict_B['r'], eigfunc_dict_B[var])

            eigfunc_dict_B[var] = interp_n_parts(eigfunc_dict_A['r'],
                                    eigfunc_dict_B['r'], eigfunc_dict_B[var],
                                    i_discon_A, i_discon_B)

        eigfunc_dict_B['r'] = eigfunc_dict_A['r']

    else: 

        for var in ['U', 'V']:

            #eigfunc_dict_A[var] = np.interp(eigfunc_dict_B['r'],
            #                        eigfunc_dict_A['r'], eigfunc_dict_A[var])

            eigfunc_dict_A[var] = interp_n_parts(eigfunc_dict_B['r'],
                                    eigfunc_dict_A['r'], eigfunc_dict_A[var],
                                    i_discon_B, i_discon_A)

        eigfunc_dict_A['r'] = eigfunc_dict_B['r']

    #for key in ['r', 'U', 'V']:

    #    print(key, np.all(eigfunc_dict_B[key] == eigfunc_dict_A[key]))

    r_range = eigfunc_dict_A['r'][-1] - eigfunc_dict_A['r'][0]
    
    U_diff = eigfunc_dict_A['U'] - eigfunc_dict_B['U']
    rms_U = np.sqrt(np.trapz(U_diff**2.0, x = eigfunc_dict_A['r'])/r_range)

    V_diff = eigfunc_dict_A['V'] - eigfunc_dict_B['V']
    rms_V = np.sqrt(np.trapz(V_diff**2.0, x = eigfunc_dict_A['r'])/r_range)

    rms = rms_U + rms_V
    
    #print(n, l, np.max(np.abs(U_diff)), np.max(np.abs(V_diff)))
    
    plot = False
    if plot:

        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = plt.gca()
        
        c_U = 'b'
        c_V = 'r'
        ls_A = '-'
        ls_B = ':'
        #ax.plot(eigfunc_dict_A['r'], eigfunc_dict_A['U'], label = 'U (A)', color = c_U, linestyle = ls_A)
        #ax.plot(eigfunc_dict_A['r'], eigfunc_dict_A['V'], label = 'V (A)', color = c_V, linestyle = ls_A)

        #ax.plot(eigfunc_dict_B['r'], eigfunc_dict_B['U'], label = 'U (B)', color = c_U, linestyle = ls_B)
        #ax.plot(eigfunc_dict_B['r'], eigfunc_dict_B['V'], label = 'V (B)', color = c_V, linestyle = ls_B)

        ax.plot(eigfunc_dict_B['r'], eigfunc_dict_A['U'] - eigfunc_dict_B['U'], label = 'U (A - B)', color = c_U, linestyle = ls_B)
        ax.plot(eigfunc_dict_B['r'], eigfunc_dict_A['V'] - eigfunc_dict_B['V'], label = 'V (A - B)', color = c_V, linestyle = ls_B)

        ax.legend()

        plt.show()

    return rms

def main():

    # Read input arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("path_input_A", help = "File path (relative or absolute) to first input file.")
    parser.add_argument("path_input_B", help = "File path (relative or absolute) to second input file.")
    #
    args = parser.parse_args()
    #
    path_input_A = args.path_input_A
    path_input_B = args.path_input_B

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
    out_fmt = '{:>5d} {:>5d} {:>19.12f} {:>19.12f} {:>18.12e}\n'

    # Loop over mode types.
    for mode_type in mode_types:

        # Found output path.
        path_out = get_eigfunc_comparison_out_path(run_info_A, run_info_B, mode_type) 

        # Load mode lists.
        mode_info_A = load_eigenfreq(run_info_A, mode_type)
        mode_info_B = load_eigenfreq(run_info_B, mode_type)

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
        rms = np.zeros(num_modes)
        for i in range(num_modes):
        #for i in [0]:

            rms[i] = comparison_wrapper(run_info_A, run_info_B, mode_type, n[i], l[i],
                                om_A[i], om_B[i], norm_args)

        # Save.
        print('Writing to {:}'.format(path_out))
        with open(path_out, 'w') as out_id:

            for i in range(num_modes):

                out_id.write(out_fmt.format(n[i], l[i], f_A[i], f_B[i], rms[i]))

    return

if __name__ == '__main__':

    main()
