import argparse
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from Ouroboros.common import align_mode_lists, get_Ouroboros_out_dirs, load_eigenfreq, mkdir_if_not_exist, read_input_file
from Ouroboros.misc.compare_eigenfunctions import get_eigfunc_comparison_out_path

def plot_dispersion_wrapper(run_info, mode_type, ax = None, save = True, show = True, i_toroidal = None, f_lims = 'auto', l_lims = 'auto', colors = ['k', 'k'], apply_atten_correction = False):
    
    # Get mode information.
    mode_info = load_eigenfreq(run_info, mode_type, i_toroidal = i_toroidal)
    n = mode_info['n']
    l = mode_info['l']
    f = mode_info['f']

    # Try to also load radial modes.
    if mode_type == 'S':
        
        try:

            mode_info_R = load_eigenfreq(run_info, 'R')
            nlf_radial = (mode_info_R['n'], mode_info_R['l'], mode_info_R['f'])
    
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
        if run_info['code'] == 'mineos':

            dir_out = run_info['dir_output'] 

        elif run_info['code'] == 'ouroboros':

            _, _, _, dir_type = get_Ouroboros_out_dirs(run_info, mode_type)
            dir_out = dir_type

        else:

            raise ValueError

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

def get_f_lims_indices(f, f_lims):

    if (f_lims == 'auto') or (f_lims is None):

        num_modes = len(n)
        i_f = np.array(list(range(num_modes)), dtype = np.int)

    else:

        i_f = np.where((f > f_lims[0]) & (f < f_lims[1]))[0]

    return i_f

def plot_differences(run_info_0, run_info_1, mode_type, diff_type = 'eigenvalues', i_toroidal = None, apply_atten_correction = False, f_lims = None, l_lims = None, save = True, show = True, var_lims = None):
    
    if diff_type == 'eigvals':

        mode_info_0 = load_eigenfreq(run_info_0, mode_type, i_toroidal = i_toroidal)
        n_0 = mode_info_0['n']
        l_0 = mode_info_0['l']
        f_0 = mode_info_0['f']

        mode_info_1 = load_eigenfreq(run_info_1, mode_type, i_toroidal = None)
        n_1 = mode_info_1['n']
        l_1 = mode_info_1['l']
        f_1 = mode_info_1['f']

        n, l, i_align_0, i_align_1 = align_mode_lists(n_0, l_0, n_1, l_1)
        f_0 = f_0[i_align_0]
        f_1 = f_1[i_align_1]

        f_mean = (f_0 + f_1)/2.0
        i_f = get_f_lims_indices(f_mean, f_lims)

        f_diff = (f_0 - f_1)
        abs_f_diff = np.abs(f_diff)
        frac_abs_f_diff = abs_f_diff/f_mean
        min_frac_abs_f_diff = np.min(frac_abs_f_diff[i_f])
        max_frac_abs_f_diff = np.max(frac_abs_f_diff[i_f])
        range_frac_abs_f_diff = max_frac_abs_f_diff - min_frac_abs_f_diff

        var_min = min_frac_abs_f_diff
        var_range = range_frac_abs_f_diff
        var = frac_abs_f_diff

        colors = []
        color_pos = 'b'
        color_neg = 'r'
        for i in range(len(f_1)):

            if f_diff[i] > 0.0:

                colors.append(color_pos)

            else:

                colors.append(color_neg)

    elif diff_type == 'eigvecs':
    
        path_rms = get_eigfunc_comparison_out_path(run_info_0, run_info_1, mode_type) 

        n, l, f_0, f_1, rms = np.loadtxt(path_rms).T
        n = n.astype(np.int)
        l = l.astype(np.int)
        
        f_mean = (f_0 + f_1)/2.0
        i_f = get_f_lims_indices(f_mean, f_lims)

        var_min = np.min(rms[i_f])
        var_max = np.max(rms[i_f])
        var_range = (var_max - var_min)
        var = rms

        colors = 'k'

    else:

        raise ValueError

    if var_range == 0.0:

        var_range = 1.0

    var_lims = [1.0E-7, 1.0E-2]
    if var_lims is not None:

        var_min = var_lims[0]
        var_max = var_lims[1]
        var_range = var_max - var_min

    s_min =  3.0
    s_max = 50.0
    s_range = s_max - s_min
    s_mid = (s_min + s_max)/2.0
    sizes = s_min + s_range*(var - var_min)/var_range

    ax = None
    if ax is None:

        fig = plt.figure(figsize = (9.5, 5.5))
        ax  = plt.gca()

    ax = plot_dispersion(n, l, f_mean, l_lims = l_lims, f_lims = f_lims, ax = ax, sizes = sizes, show = False, colors = colors)

    if diff_type == 'eigvals':

        ax.scatter([], [], c = 'k', s = s_min, label = '{:>.3f} %'.format(var_min*1.0E2))
        ax.scatter([], [], c = 'k', s = s_max, label = '{:>.3f} %'.format(var_max*1.0E2))
        ax.scatter([], [], c = color_pos, s = s_mid, label = 'Positive')
        ax.scatter([], [], c = color_neg, s = s_mid, label = 'Negative')
        plt.legend(title = 'Freq. diff.')

        fig_name = 'dispersion_differences.png'

    elif diff_type == 'eigvecs':

        ax.scatter([], [], c = 'k', s = s_min, label = '{:>8.3e}'.format(var_min))
        ax.scatter([], [], c = 'k', s = s_max, label = '{:>8.3e}'.format(var_max))
        plt.legend(title = 'Difference')

        fig_name = 'eigenfunction_differences.png'

    else:

        raise ValueError

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
    parser.add_argument("path_input", help = "File path (relative or absolute) to input file.")
    parser.add_argument("--toroidal", dest = "layer_number", help = "Plot toroidal modes for the solid shell given by LAYER_NUMBER (0 is outermost solid shell). Default is to plot spheroidal modes. (Note: syntax is different for plotting Mineos toroidal modes.)", type = int)
    parser.add_argument("--f_lims", type = float, nargs = 2, help = "Specify frequency limits (mHz) of plot axes (default: limits are found automatically).")
    parser.add_argument("--l_lims", type = float, nargs = 2, help = "Specify angular order of plot axes (default: limits are found automatically).")
    parser.add_argument("--path_input_comparison", help = "File path to second input file for comparison.")
    parser.add_argument("--plot_diff", choices = ['eigvals', 'eigvecs'], help = 'Plot differences between mode frequencies (option \'eigvals\') or eigenfunctions (option \'eigvecs\').')
    args = parser.parse_args()

    # Rename input arguments.
    path_input = args.path_input
    i_toroidal = args.layer_number
    path_input_comparison = args.path_input_comparison
    diff_type = args.plot_diff
    if diff_type is not None:

        assert path_input_comparison is not None, 'To plot differences, you must specify --path_input_comparison.'

    f_lims = args.f_lims
    l_lims = args.l_lims
    if f_lims is None:
        f_lims = 'auto'
    if l_lims is None:
        l_lims = 'auto'
    
    # Read input file(s).
    run_info = read_input_file(path_input)
    #
    if path_input_comparison is not None:

        run_info_comparison = read_input_file(path_input_comparison)

    ## Read the input file.
    #if use_mineos:

    #    # Read Mineos input file.
    #    run_info = read_Mineos_input_file(path_input)

    #    # Store whether Mineos is being used.
    #    run_info['use_mineos'] = use_mineos

    #elif path_input_mineos is not None:

    #    # Read Mineos input file.
    #    run_info_mineos = read_Mineos_input_file(path_input_mineos)

    #    # Store whether Mineos is being used.
    #    run_info_mineos['use_mineos'] = True

    #    # Read Ouroboros input file.
    #    run_info = read_Ouroboros_input_file(path_input)

    #    # Store whether Mineos is being used.
    #    run_info['use_mineos'] = False

    #else:
    #    
    #    # Read Ouroboros input file.
    #    run_info = read_Ouroboros_input_file(path_input)

    #    # Store whether Mineos is being used.
    #    run_info['use_mineos'] = use_mineos

    # Set mode type string.
    if i_toroidal is not None:

        mode_type = 'T'
    
    elif run_info['mode_types'] == ['R']:
        
        print('Cannot plot dispersion diagram for only radial modes. Try including spheroidal modes in input file.')

    else:

        mode_type = 'S'

    # Plot the dispersion diagram.
    if diff_type is not None:
        
        assert path_input_comparison is not None
        plot_differences(run_info, run_info_comparison, mode_type, diff_type = diff_type, f_lims = f_lims, l_lims = l_lims)

    else:

        if path_input_comparison is not None:

            ax = plot_dispersion_wrapper(run_info_comparison, mode_type, i_toroidal = i_toroidal, f_lims = f_lims, l_lims = l_lims, show = False, save = False,
                        colors = ['r', 'r'])
            plot_dispersion_wrapper(run_info, mode_type, i_toroidal = i_toroidal, f_lims = f_lims, l_lims = l_lims, ax = ax, show = True, save = True,
                        colors = ['b', 'b'])

        else:

            plot_dispersion_wrapper(run_info, mode_type, i_toroidal = i_toroidal, f_lims = f_lims, l_lims = l_lims)

    return

if __name__ == '__main__':

    main()
