import argparse
from glob import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas

from Ouroboros.common import load_eigenfunc, load_eigenfreq, read_input_file, read_Ouroboros_summation_input_file
from Ouroboros.misc.cmt_io import read_mineos_cmt
from Ouroboros.summation.run_summation import get_output_dir_info

def read_cmt_files(path_list):

    n_events = len(path_list)

    depth = np.zeros(n_events)
    moment = np.zeros(n_events)
    
    for i in range(n_events):

        cmt = read_mineos_cmt(path_list[i])

        depth[i] = cmt['depth_centroid']
        moment[i] = cmt['scalar_moment']

    event_info = dict()
    event_info['depth'] = depth
    event_info['moment'] = moment
        
    return event_info

def read_coeffs(summation_info, cmt_path_list, mode_list):

    num_events = len(cmt_path_list)
    num_modes = len(mode_list)

    coeffs = np.zeros((num_events, num_modes))

    for i in range(num_events):

        path_cmt = cmt_path_list[i]
        name_cmt = os.path.basename(path_cmt).split('.')[0]

        if i == 0:

            # Load mode data.
            path_modes = os.path.join(summation_info['dir_output'], 'modes_mean_{:}.pkl'.format(name_cmt))
            print('Loading {:}'.format(path_modes))
            modes = pandas.read_pickle(path_modes)

            j_list = []
            for k in range(num_modes):

                n, l = mode_list[k]

                j = np.where((modes['n'] == n) & (modes['l'] == l))[0][0]
                j_list.append(j)

        j_list = np.array(j_list, dtype = np.int)

        name_coeffs = 'coeffs_mean_{:}.pkl'.format(name_cmt)
        path_coeffs = os.path.join(summation_info['dir_output'], name_coeffs)
        coeffs_i = pandas.read_pickle(path_coeffs)
        coeffs_i = np.array(coeffs_i)

        for k in range(num_modes):

            coeffs[i, k] = coeffs_i[j_list[k]]

    return coeffs

def main():

    # Parse input arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("path_mode_input", help = "File path (relative or absolute) to Ouroboros mode input file.")
    parser.add_argument("path_summation_input", help = "File path (relative or absolute) to Ouroboros summation input file.")
    #
    input_args = parser.parse_args()
    #
    path_mode_input = input_args.path_mode_input
    path_summation_input = input_args.path_summation_input

    # Read the mode input file.
    run_info = read_input_file(path_mode_input)

    # Read the summation input file.
    summation_info = read_Ouroboros_summation_input_file(path_summation_input)

    # Get output directory information.
    run_info, summation_info = get_output_dir_info(run_info, summation_info)

    # Get list of CMT files.
    cmt_list_regex = os.path.join(summation_info['path_cmt'], '*.txt')
    cmt_path_list = glob(cmt_list_regex)
    cmt_path_list.sort()

    # Get list of CMT information.
    event_info = read_cmt_files(cmt_path_list)

    # Get list of excitation coefficients.
    mode_list = [[0, 48], [1, 31], [2, 25], [3, 25]]
    coeffs = read_coeffs(summation_info, cmt_path_list, mode_list)

    coeffs = np.abs(coeffs)

    i_mode_0 = 3
    i_mode_1 = 0

    n0, l0 = mode_list[i_mode_0]
    n1, l1 = mode_list[i_mode_1]
    
    fig = plt.figure(figsize = (5.0, 8.0), constrained_layout = True)
    ax = plt.gca()

    ratio = coeffs[:, i_mode_0]/coeffs[:, i_mode_1]

    ax.scatter(ratio, event_info['depth'], alpha = 0.6, c = 'purple')

    font_size_label = 13
    font_size_title = 15
    ax.set_xlabel('Excitation ratio', fontsize = font_size_label)
    ax.set_ylabel('Depth (km)', fontsize = font_size_label)
    
    x_lim = ax.get_xlim()
    ax.set_xlim([0.0, x_lim[1]])
    ax.set_ylim([0.0, 750.0])
    ax.invert_yaxis()

    title_str = 'Ratio $_{{{:d}}}S_{{{:d}}}$ / $_{{{:d}}}S_{{{:d}}}$'.format(n0, l0, n1, l1)
    ax.set_title(title_str, fontsize = font_size_title)

    name_out = 'excitation_ratio_depth_profile_{:>05d}_{:>05d}_{:>05d}_{:>05d}.png'.format(n0, l0, n1, l1)

    path_out = os.path.join(summation_info['dir_output'], name_out)
    print('Writing to {:}'.format(path_out))
    plt.savefig(path_out, dpi = 300)

    plt.show()

    return

def plot_excitation_depth_profile():
    
    i_mode = 1 
    n, l = mode_list[i_mode]

    # Get frequency information.
    mode_type = 'S'
    mode_info = load_eigenfreq(run_info, mode_type, n_q = n, l_q = l)
    f = mode_info['f']

    # Get normalisation arguments.
    f_rad_per_s = f*1.0E-3*2.0*np.pi
    normalisation_args = {'norm_func' : 'DT', 'units' : 'mineos'}
    normalisation_args['omega'] = f_rad_per_s

    # Get eigenfunction information.
    eigfunc_dict = load_eigenfunc(run_info, mode_type, n, l,
            norm_args = normalisation_args)
    eigfunc_dict['r'] = eigfunc_dict['r']*1.0E-3 # Convert to km.
    
    coeff_scale = 1.0E36
    fig = plt.figure(figsize = (5.0, 8.0), constrained_layout = True)
    ax = plt.gca()

    ax.scatter(coeff_scale*coeffs[:, i_mode]/event_info['moment'], event_info['depth'], alpha = 0.6, c = 'orange')

    ax_eig = ax.twiny()
    z = (eigfunc_dict['r'][-1] - eigfunc_dict['r'])
    ax_eig.plot(np.abs(eigfunc_dict['U']),
            z, label = '$|U|$', c = 'r')
    ax_eig.plot(np.abs(eigfunc_dict['V']),
            z, label = '$|V|$', c = 'b')
    ax_eig.plot(np.sqrt((eigfunc_dict['U']**2.0 + eigfunc_dict['V']**2.0)),
            z, label = '$(U^{2} + V^{2})^{1/2}$', c = 'g')

    ax_eig.legend(loc = 'lower right')

    font_size_label = 13
    font_size_title = 15
    ax.set_xlabel('Excitation divided by moment', fontsize = font_size_label)
    ax.set_ylabel('Depth (km)', fontsize = font_size_label)
    ax_eig.set_xlabel('Eigenfunctions', fontsize = font_size_label)
    
    x_lim = ax.get_xlim()
    ax.set_xlim([0.0, x_lim[1]])
    ax.set_ylim([0.0, 750.0])
    ax.invert_yaxis()

    x_lim = ax_eig.get_xlim()
    ax_eig.set_xlim([0.0, x_lim[1]])
    #ax_eig.set_xlim([0.0, 0.005])
    
    title_str = '$_{{{:d}}}S_{{{:d}}}$'.format(n, l)
    ax.set_title(title_str, fontsize = font_size_title)

    name_out = 'excitation_depth_profile_{:>05d}_{:>05d}.png'.format(n, l)
    path_out = os.path.join(summation_info['dir_output'], name_out)
    print('Writing to {:}'.format(path_out))
    plt.savefig(path_out, dpi = 300)

    plt.show()

    return

if __name__ == '__main__':

    main()
