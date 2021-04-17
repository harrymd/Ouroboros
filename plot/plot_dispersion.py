import argparse
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
import numpy as np

from Ouroboros.common import align_mode_lists, get_Ouroboros_out_dirs, load_eigenfreq, mkdir_if_not_exist, read_input_file
from Ouroboros.misc.compare_eigenfunctions import get_eigfunc_comparison_out_path

def plot_dispersion_wrapper(run_info, mode_type, ax = None, save = True, show = True, i_toroidal = None, f_lims = 'auto', l_lims = 'auto', colors = ['k', 'k'], apply_atten_correction = False, var = None, path_highlight = None, highlight_box = None):
    
    # Get mode information.
    mode_info = load_eigenfreq(run_info, mode_type, i_toroidal = i_toroidal)
    n = mode_info['n']
    l = mode_info['l']
    f = mode_info['f']

    if var is not None:
    
        if var == 'Q':

            var = mode_info['Q']

            legend_label = 'Q'

        # Get scaling.
        i_f = get_f_lims_indices(f, f_lims)
        var_min = np.min(var[i_f])
        var_max = np.max(var[i_f])
        var_mid = (var_min + var_max)/2.0

        sizes, s_min, s_mid, s_max = get_sizes(var, var_min, var_max)
        size_info = {'sizes' : sizes, 's_min' : s_min, 's_max' : s_max,
                's_mid' : s_mid, 'var_min' : var_min, 'var_mid' : var_mid,
                'var_max' : var_max, 'legend_label' : legend_label}

    else:

        size_info = None

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

    if var is not None:

        add_legend = True

    else:

        add_legend = False
        
    ax = plot_dispersion(n, l, f, ax = ax, show = False, color = colors[0], c_scatter = colors[1], alpha = alpha, add_legend = add_legend, nlf_radial = nlf_radial, f_lims = f_lims, l_lims = l_lims, size_info = size_info, highlight_box = highlight_box)

    if path_highlight is not None:

        n_highlight, l_highlight = np.loadtxt(path_highlight, dtype = np.int).T
        i_highlight = []
        for i in range(len(n_highlight)):

            i_highlight.append(np.where((n_highlight[i] == n) & (l_highlight[i] == l))[0][0])
        
        ax.scatter(l[i_highlight], f[i_highlight], c = 'r', s = 5)

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

def plot_dispersion(n, l, f, ax = None, l_lims = 'auto', f_lims = 'auto', x_label = 'Angular order, $\ell$', y_label = 'Frequency / mHz', title = None, h_lines = None, path_fig = None, show = True, color = 'k', c_scatter = 'k', alpha = 1.0, label = None, add_legend = False, nlf_radial = None, size_info = None, colors = None, legend_label = None, highlight_box = None):

    n_list = sorted(list(set(n)))
    
    if ax is None:
    
        fig = plt.figure()
        ax  = plt.gca()
    
    for ni in n_list:
        
        i = (n == ni)
        
        ax.plot(l[i], f[i], ls = '-', color = color, lw = 1, alpha = alpha)

    if colors is not None:

        c_scatter = colors
        
    if size_info is None:
        sizes = 1
    else:
        sizes = size_info['sizes']
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

    if (size_info is not None) and ('s_min' in size_info.keys()):

        ax.scatter([], [], c = c_scatter, s = size_info['s_min'], label = '{:>8.3e}'.format(size_info['var_min']))
        ax.scatter([], [], c = c_scatter, s = size_info['s_mid'], label = '{:>8.3e}'.format(size_info['var_mid']))
        ax.scatter([], [], c = c_scatter, s = size_info['s_max'], label = '{:>8.3e}'.format(size_info['var_max']))

        legend_label = size_info['legend_label']

    if highlight_box is not None:

        rect = Rectangle((highlight_box[0], highlight_box[1]),
                    highlight_box[2], highlight_box[3],
                    transform = ax.transData,
                    facecolor = 'none',
                    edgecolor = 'r')
        ax.add_artist(rect)

    if add_legend:

        plt.legend(title = legend_label)

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

        num_modes = len(f)
        i_f = np.array(list(range(num_modes)), dtype = np.int)

    else:

        i_f = np.where((f > f_lims[0]) & (f < f_lims[1]))[0]

    return i_f

def get_sizes(var, var_min, var_max, s_min = 3.0, s_max = 50.0):
    
    var_range = var_max - var_min
    s_range = s_max - s_min
    s_mid = (s_min + s_max)/2.0
    sizes = s_min + s_range*(var - var_min)/var_range

    return sizes, s_min, s_mid, s_max

def assign_colors_by_sign(var, color_pos = 'b', color_neg = 'r'):

    colors = []
    for i in range(len(var)):

        if var[i] > 0.0:

            colors.append(color_pos)

        else:

            colors.append(color_neg)

    return colors, color_pos, color_neg

def get_var_aligned(mode_type, run_info_0, run_info_1, var, f_lims):

    mode_info_0 = load_eigenfreq(run_info_0, mode_type, i_toroidal = None)
    n_0 = mode_info_0['n']
    l_0 = mode_info_0['l']
    f_0 = mode_info_0['f']
    var_0 = mode_info_0[var]

    i = np.where((n_0 == 2) & (l_0 == 1))[0]
    n_0 = np.delete(n_0, i)
    l_0 = np.delete(l_0, i)
    f_0 = np.delete(f_0, i)
    var_0 = np.delete(var_0, i)

    mode_info_1 = load_eigenfreq(run_info_1, mode_type, i_toroidal = None)
    n_1 = mode_info_1['n']
    l_1 = mode_info_1['l']
    f_1 = mode_info_1['f']
    var_1 = mode_info_1[var]

    n, l, i_align_0, i_align_1 = align_mode_lists(n_0, l_0, n_1, l_1)
    f_0 = f_0[i_align_0]
    f_1 = f_1[i_align_1]
    var_0 = var_0[i_align_0]
    var_1 = var_1[i_align_1]

    f_mean = (f_0 + f_1)/2.0
    var_mean = (var_0 + var_1)/2.0
    i_f = get_f_lims_indices(f_mean, f_lims)

    var_diff = (var_0 - var_1)
    abs_var_diff = np.abs(var_diff)
    frac_abs_var_diff = abs_var_diff/var_mean
    min_frac_abs_var_diff = np.min(frac_abs_var_diff[i_f])
    max_frac_abs_var_diff = np.max(frac_abs_var_diff[i_f])
    range_frac_abs_var_diff = max_frac_abs_var_diff - min_frac_abs_var_diff

    var_min = min_frac_abs_var_diff
    var_max = max_frac_abs_var_diff

    return n, l, f_mean, var_diff, frac_abs_var_diff, var_min, var_max

def plot_differences(run_info_0, run_info_1, mode_type, diff_type = 'eigenvalues', i_toroidal = None, apply_atten_correction = False, f_lims = None, l_lims = None, save = True, show = True, var_lims = None):
    
    if diff_type == 'eigvals':

        n, l, f_mean, var_diff, var, var_min, var_max  = \
            get_var_aligned(mode_type, run_info_0, run_info_1, 'f', f_lims)

        colors, color_pos, color_neg = assign_colors_by_sign(var_diff)

    elif diff_type == 'eigvecs':
    
        path_rms = get_eigfunc_comparison_out_path(run_info_0, run_info_1, mode_type) 
        if not os.path.exists(path_rms):

            print('Comparison file {:} not found. Try running the misc/compare_eigenfunctions.py command.'.format(path_rms))
            return


        
        n, l, f_0, f_1, rms_diff, rms_A, rms_B = np.loadtxt(path_rms).T
        n = n.astype(np.int)
        l = l.astype(np.int)
        
        # 
        i = np.where((n == 2) & (l == 1))[0]
        n = np.delete(n, i)
        l = np.delete(l, i)
        f_0 = np.delete(f_0, i)
        f_1 = np.delete(f_1, i)
        rms_diff = np.delete(rms_diff, i)
        rms_A = np.delete(rms_A, i)
        rms_B = np.delete(rms_B, i)
        
        f_mean = (f_0 + f_1)/2.0
        i_f = get_f_lims_indices(f_mean, f_lims)

        rms_mean = (rms_A + rms_B)/2.0
        var = rms_diff/rms_mean

        var_min = np.min(var[i_f])
        var_max = np.max(var[i_f])

        colors = 'k'

    elif diff_type == 'Q':

        n, l, f_mean, var_diff, var, var_min, var_max  = \
            get_var_aligned(mode_type, run_info_0, run_info_1, 'Q', f_lims)

        colors, color_pos, color_neg = assign_colors_by_sign(var_diff)

    else:

        raise ValueError
    
    var_range = var_max - var_min
    if var_range == 0.0:

        var_range = 1.0

    #var_lims = [1.0E-7, 1.0E-2]
    #var_lims = [0.0, 0.05]
    #var_lims = [0.001, 0.02]
    #var_lims = [0.0, 0.02]
    if var_lims is not None:

        var_min = var_lims[0]
        var_max = var_lims[1]
        var_range = var_max - var_min

    sizes, s_min, s_mid, s_max = get_sizes(var, var_min, var_max)
    size_info = {'sizes' : sizes}

    ax = None
    if ax is None:

        fig = plt.figure(figsize = (9.5, 5.5))
        ax  = plt.gca()

    ax = plot_dispersion(n, l, f_mean, l_lims = l_lims, f_lims = f_lims, ax = ax, size_info = size_info, show = False, colors = colors)

    if diff_type in ['eigvals', 'Q']:

        #ax.scatter([], [], c = 'k', s = s_min, label = '{:>.3f} %'.format(var_min*1.0E2))
        #ax.scatter([], [], c = 'k', s = s_max, label = '{:>.3f} %'.format(var_max*1.0E2))
        ax.scatter([], [], c = 'k', s = s_min, label = '{:>.1e} %'.format(var_min*1.0E2))
        ax.scatter([], [], c = 'k', s = s_max, label = '{:>.1e} %'.format(var_max*1.0E2))
        ax.scatter([], [], c = color_pos, s = s_mid, label = 'Positive')
        ax.scatter([], [], c = color_neg, s = s_mid, label = 'Negative')

        if diff_type == 'eigvals':

            plt.legend(title = 'Freq. diff.')
            fig_name = 'dispersion_differences.png'

        elif diff_type == 'Q':

            plt.legend(title = 'Q diff.')
            fig_name = 'Q_differences.png'

    elif diff_type == 'eigvecs':

        ax.scatter([], [], c = 'k', s = s_min, label = '{:>.3f}'.format(100.0*var_min))
        ax.scatter([], [], c = 'k', s = s_max, label = '{:>.3f}'.format(100.0*var_max))
        plt.legend(title = 'RMS diff. (%)')

        fig_name = 'eigenfunction_differences.png'

    else:

        raise ValueError

    if save:
        
        if run_info_0['code'] == 'ouroboros':

            _, _, _, dir_type = get_Ouroboros_out_dirs(run_info_0, mode_type)
            dir_out = dir_type

        elif run_info_0['code'] == 'mineos':

            dir_out = run_info_0['dir_run']

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
    parser.add_argument("--path_highlight")
    parser.add_argument("--toroidal", dest = "layer_number", help = "Plot toroidal modes for the solid shell given by LAYER_NUMBER (0 is outermost solid shell). Default is to plot spheroidal modes. (Note: syntax is different for plotting Mineos toroidal modes.)", type = int)
    parser.add_argument("--f_lims", type = float, nargs = 2, help = "Specify frequency limits (mHz) of plot axes (default: limits are found automatically).")
    parser.add_argument("--l_lims", type = float, nargs = 2, help = "Specify angular order of plot axes (default: limits are found automatically).")
    parser.add_argument("--path_input_comparison", help = "File path to second input file for comparison.")
    parser.add_argument("--plot_diff", choices = ['eigvals', 'eigvecs', 'Q'], help = 'Plot differences between mode frequencies (option \'eigvals\') or eigenfunctions (option \'eigvecs\').')
    parser.add_argument("--plot_var", choices = ['Q'], help = 'Plot a variable. Options: Q (attenuation quality factor).')
    parser.add_argument("--var_lims", type = float, nargs = 2)
    parser.add_argument("--highlight_box", type = float, nargs = 4, help = "(x0, y0, width, height) of highlight box.")
    args = parser.parse_args()

    # Rename input arguments.
    path_input = args.path_input    
    path_highlight = args.path_highlight
    i_toroidal = args.layer_number
    path_input_comparison = args.path_input_comparison
    diff_type = args.plot_diff
    plot_var = args.plot_var
    var_lims = args.var_lims
    highlight_box = args.highlight_box
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

    if plot_var == 'Q':

        assert run_info['use_attenuation'], 'Cannot plot Q for runs with no attenuation.'
        assert diff_type is None, 'Cannot combine --plot_var Q and --plot_diff Q. Try using only --plot_diff Q.'

    if plot_var is not None:

        assert path_input_comparison is None, 'Cannot combine --plot_var and --path_input_comparison. Try using --plot_diff.'

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
        plot_differences(run_info, run_info_comparison, mode_type, diff_type = diff_type, f_lims = f_lims, l_lims = l_lims, var_lims = var_lims)

    else:

        if path_input_comparison is not None:

            ax = plot_dispersion_wrapper(run_info_comparison, mode_type, i_toroidal = i_toroidal, f_lims = f_lims, l_lims = l_lims, show = False, save = False,
                        colors = ['r', 'r'], var = plot_var)
            plot_dispersion_wrapper(run_info, mode_type, i_toroidal = i_toroidal, f_lims = f_lims, l_lims = l_lims, ax = ax, show = True, save = True,
                        colors = ['b', 'b'], var = plot_var)

        else:

            plot_dispersion_wrapper(run_info, mode_type, i_toroidal = i_toroidal, f_lims = f_lims, l_lims = l_lims, var = plot_var, path_highlight = path_highlight,
                                        highlight_box = highlight_box)

    return

if __name__ == '__main__':

    main()
