import argparse
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from common import load_eigenfreq_Mineos, load_eigenfreq_Ouroboros, mkdir_if_not_exist, read_Mineos_input_file, read_Ouroboros_input_file, get_Ouroboros_out_dirs
#from stoneley.code.common.Mineos import load_eigenfreq_Mineos

def plot_dispersion_wrapper(run_info, mode_type, ax = None, save = True, show = True, i_toroidal = None, f_lims = 'auto', l_lims = 'auto', colors = ['k', 'k'], apply_atten_correction = False):

    # Get frequency information.
    if run_info['use_mineos']:
        
        assert not apply_atten_correction, 'Mineos modes not compatible with attenuation correction.'

        if i_toroidal is not None:

            raise NotImplementedError('Dispersion plotting for Mineos toroidal modes not implemented yet.')
        
        #name_run = '{:>05d}_{:>05d}_{:>1d}'.format(run_info['l_lims'][1], run_info['n_lims'][1], run_info['g_switch'])
        #dir_run = os.path.join(run_info['dir_output'], run_info['model'], name_run)
        #path_minos_bran = os.path.join(dir_run, 'minos_bran_out_{:}.txt'.format(mode_type))
        #mode_data = read_minos_bran_output(path_minos_bran)
        #print(mode_data)
        #n = mode_data['n']
        #l = mode_data['l']
        #f = mode_data['w'] # Freq in mHz.

        n, l, f = load_eigenfreq_Mineos(run_info, mode_type)

    else:

        if apply_atten_correction:
            
            n, l, f, Q = load_eigenfreq_Ouroboros(run_info, mode_type, i_toroidal = i_toroidal, return_Q = True)

        else:

            n, l, f = load_eigenfreq_Ouroboros(run_info, mode_type, i_toroidal = i_toroidal)

    # Try to also load radial modes.
    if mode_type == 'S':

        try:

            if run_info['use_mineos']:

                #path_minos_bran_R = os.path.join(dir_run, 'minos_bran_out_{:}.txt'.format('R'))
                #mode_data_R = read_minos_bran_output(path_minos_bran_R)
                #n_R = mode_data_R['n']
                #l_R = mode_data_R['l']
                #f_R = mode_data_R['w'] # Freq in mHz.
                #nlf_radial = (n_R, l_R, f_R)
                nlf_radial = load_eigenfreq_Mineos(run_info, 'R')

            else:

                nlf_radial = load_eigenfreq_Ouroboros(run_info, 'R')
    
        except OSError:

            print('No radial mode data found; plot will not include radial modes.')
            nlf_radial = None

    else:

        nlf_radial = None

    alpha = 0.5

    if ax is None:

        fig = plt.figure(figsize = (9.5, 5.5))
        ax  = plt.gca()
     
    ax = plot_dispersion(n, l, f, ax = ax, show = False, color = colors[0], c_scatter = colors[1], alpha = alpha, add_legend = False, nlf_radial = nlf_radial, f_lims = f_lims, l_lims = l_lims)

    #ax.set_ylim([0.0, 14.0])
    #ax.set_xlim([0.0, 60.0])

    if save:
        
        fig_name = 'dispersion.png'
        if run_info['use_mineos']:

            dir_out = run_info['dir_output'] 

        else:

            _, _, _, dir_type = get_Ouroboros_out_dirs(run_info, mode_type)
            dir_out = dir_type

        dir_plot = os.path.join(dir_out, 'plots')
        mkdir_if_not_exist(dir_plot)
        fig_path = os.path.join(dir_plot, fig_name)
        print('Saving figure to {:}'.format(fig_path))
        plt.savefig(fig_path, dpi = 300, bbox_inches = 'tight')

    if show:
    
        plt.show()

    return ax

def plot_dispersion(n, l, f, ax = None, l_lims = 'auto', f_lims = 'auto', x_label = 'Angular order, $\ell$', y_label = 'Frequency / mHz', title = None, h_lines = None, path_fig = None, show = True, color = 'k', c_scatter = 'k', alpha = 1.0, label = None, add_legend = False, nlf_radial = None, sizes = None, colors = None):

    n_list = sorted(list(set(n)))
    
    if ax is None:
    
        fig = plt.figure()
        ax  = plt.gca()
    
    for ni in n_list:
        
        i = (n == ni)
        
        ax.plot(l[i], f[i], ls = '-', color = color, lw = 1, alpha = alpha)

    if colors is not None:

        c_scatter = colors
        
    if sizes is None:
        sizes = 1
    ax.scatter(l, f, s = sizes, c = c_scatter, alpha = alpha, zorder = 10)

    if nlf_radial is not None:

        n_radial, l_radial, f_radial = nlf_radial
        ax.scatter(l_radial, f_radial, s = sizes, c = c_scatter, alpha = alpha, zorder = 10)
    
    if label is not None:

        ax.plot([], [], linestyle = '-', marker = '.', color = color, alpha = alpha, label = label)

    font_size_label = 18
    
    if l_lims != 'auto':

        ax.set_xlim(l_lims)

    ax.xaxis.set_major_locator(MaxNLocator(integer = True))

    if f_lims is not None:

        if f_lims == 'auto':

            f_min = 0.0
            f_max = np.max(f)
            buff = (f_max - f_min)*0.05
            f_lims = [f_min, f_max + buff]

        ax.set_ylim(f_lims)

    if x_label is not None:

        ax.set_xlabel(x_label, fontsize = font_size_label)

    if y_label is not None:

        ax.set_ylabel(y_label, fontsize = font_size_label)

    if title is not None:
        
        ax.set_title(title, fontsize = font_size_label)
    
    if h_lines is not None:

        for h_line in h_lines:

            ax.axhline(h_line, color = 'k', linestyle = ':')

    if add_legend:

        plt.legend()

    plt.tight_layout()

    if path_fig is not None:
        
        plt.savefig(path_fig,
            dpi         = 300,
            transparent = True,
            bbox_inches = 'tight')
            
    if show:
        
        plt.show()
        
    return ax

def align_mode_lists(n_0, l_0, f_0, n_1, l_1, f_1):

    num_modes_0 = len(n_0)
    f_1_sorted_by_0 = np.zeros(num_modes_0)

    for i in range(num_modes_0):

        j = np.where((l_1 == l_0[i]) & (n_1 == n_0[i]))[0]
        
        assert len(j) < 2
        if len(j) == 1:

            f_1_sorted_by_0[i] = f_1[j]

        else:

            f_1_sorted_by_0[i] = np.nan

    i_good = np.where(~np.isnan(f_1_sorted_by_0))[0]
    n = n_0[i_good]
    l = l_0[i_good]
    f_0 = f_0[i_good]
    f_1 = f_1_sorted_by_0[i_good]

    return n, l, f_0, f_1, i_good

def plot_differences(run_info_0, run_info_1, mode_type, i_toroidal = None, apply_atten_correction = False, f_lims = None, l_lims = None, save = True, show = True):
    
    mode_info_0 = load_eigenfreq_Ouroboros(run_info_0, mode_type, i_toroidal = i_toroidal)
    n_0 = mode_info_0['n']
    l_0 = mode_info_0['l']
    f_0 = mode_info_0['f']



    n_1, l_1, f_1 = load_eigenfreq_Mineos(run_info_1, mode_type)

    n, l, f_0, f_1, _ = align_mode_lists(n_0, l_0, f_0, n_1, l_1, f_1)

    f_mean = (f_0 + f_1)/2.0
    i_f = np.where((f_mean > f_lims[0]) & (f_mean < f_lims[1]))[0]
    f_diff = (f_0 - f_1)
    abs_f_diff = np.abs(f_diff)
    frac_abs_f_diff = abs_f_diff/f_mean
    min_frac_abs_f_diff = np.min(frac_abs_f_diff[i_f])
    max_frac_abs_f_diff = np.max(frac_abs_f_diff[i_f])
    range_frac_abs_f_diff = max_frac_abs_f_diff - min_frac_abs_f_diff
    s_min =  3.0
    s_max = 50.0
    s_range = s_max - s_min
    s_mid = (s_min + s_max)/2.0
    sizes = s_min + s_range*(frac_abs_f_diff - min_frac_abs_f_diff)/range_frac_abs_f_diff


    colors = []
    color_pos = 'b'
    color_neg = 'r'
    for i in range(len(f_1)):

        if f_diff[i] > 0.0:

            colors.append(color_pos)

        else:

            colors.append(color_neg)

    ax = None
    if ax is None:

        fig = plt.figure(figsize = (9.5, 5.5))
        ax  = plt.gca()

    ax = plot_dispersion(n, l, f_mean, l_lims = l_lims, f_lims = f_lims, ax = ax, sizes = sizes, show = False, colors = colors)

    ax.scatter([], [], c = 'k', s = s_min, label = '{:>.3f} %'.format(min_frac_abs_f_diff*1.0E2))
    ax.scatter([], [], c = 'k', s = s_max, label = '{:>.3f} %'.format(max_frac_abs_f_diff*1.0E2))
    ax.scatter([], [], c = color_pos, s = s_mid, label = 'Positive')
    ax.scatter([], [], c = color_neg, s = s_mid, label = 'Negative')
    plt.legend(title = 'Freq. diff.')

    if save:
        
        fig_name = 'dispersion_differences.png'

        _, _, _, dir_type = get_Ouroboros_out_dirs(run_info_0, mode_type)
        dir_out = dir_type

        dir_plot = os.path.join(dir_out, 'plots')
        mkdir_if_not_exist(dir_plot)
        fig_path = os.path.join(dir_plot, fig_name)
        print('Saving figure to {:}'.format(fig_path))
        plt.savefig(fig_path, dpi = 300, bbox_inches = 'tight')

    if show:
    
        plt.show()

    return

def main():

    # Read input arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_input_file", help = "File path (relative or absolute) to Ouroboros input file.")
    parser.add_argument("--toroidal", dest = "layer_number", help = "Plot toroidal modes for the solid shell given by LAYER_NUMBER (0 is outermost solid shell). Default is to plot spheroidal modes.", type = int)
    parser.add_argument("--use_mineos", action = "store_true", help = "Plot only Mineos modes (default: only Ouroboros) ")
    parser.add_argument("--f_lims", type = float, nargs = 2, help = "Specify frequency limits (mHz) of plot axes (default: limits are found automatically).")
    parser.add_argument("--l_lims", type = float, nargs = 2, help = "Specify angular order of plot axes (default: limits are found automatically).")
    parser.add_argument("--path_input_mineos_comparison", help = "File path to Mineos input file for comparison.")
    parser.add_argument("--attenuation_correction", action = 'store_true', help = 'Before plotting, apply attenuation correction calculated with calculate_Q.py.')
    parser.add_argument("--plot_differences", action = 'store_true', help = 'Plot differences between mode frequencies.')
    args = parser.parse_args()

    # Rename input arguments.
    path_input = args.path_to_input_file
    i_toroidal = args.layer_number
    use_mineos = args.use_mineos
    path_input_mineos = args.path_input_mineos_comparison
    apply_atten_correction =  args.attenuation_correction
    f_lims = args.f_lims
    l_lims = args.l_lims
    if f_lims is None:
        f_lims = 'auto'
    if l_lims is None:
        l_lims = 'auto'

    # Read the input file.
    if use_mineos:

        # Read Mineos input file.
        run_info = read_Mineos_input_file(path_input)

        # Store whether Mineos is being used.
        run_info['use_mineos'] = use_mineos

    elif path_input_mineos is not None:

        # Read Mineos input file.
        run_info_mineos = read_Mineos_input_file(path_input_mineos)

        # Store whether Mineos is being used.
        run_info_mineos['use_mineos'] = True

        # Read Ouroboros input file.
        run_info = read_Ouroboros_input_file(path_input)

        # Store whether Mineos is being used.
        run_info['use_mineos'] = False

    else:
        
        # Read Ouroboros input file.
        run_info = read_Ouroboros_input_file(path_input)

        # Store whether Mineos is being used.
        run_info['use_mineos'] = use_mineos

    # Set mode type string.
    if i_toroidal is not None:

        mode_type = 'T'
    
    elif run_info['mode_types'] == ['R']:
        
        print('Cannot plot dispersion diagram for only radial modes. Try including spheroidal modes in input file.')

    else:

        mode_type = 'S'

    # Plot the dispersion diagram.
    if plot_differences:
        
        assert path_input_mineos is not None
        plot_differences(run_info, run_info_mineos, mode_type, f_lims = f_lims, l_lims = l_lims, apply_atten_correction = apply_atten_correction)

    else:

        if path_input_mineos is not None:

            ax = plot_dispersion_wrapper(run_info_mineos, mode_type, i_toroidal = i_toroidal, f_lims = f_lims, l_lims = l_lims, show = False, save = False,
                        colors = ['r', 'r'])
            plot_dispersion_wrapper(run_info, mode_type, i_toroidal = i_toroidal, f_lims = f_lims, l_lims = l_lims, ax = ax, show = True, save = True,
                        colors = ['b', 'b'], apply_atten_correction = apply_atten_correction)

        else:

            plot_dispersion_wrapper(run_info, mode_type, i_toroidal = i_toroidal, f_lims = f_lims, l_lims = l_lims, apply_atten_correction = apply_atten_correction)

    return

if __name__ == '__main__':

    main()
