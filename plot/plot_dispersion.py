'''
Plot dispersion diagrams (angular order versus frequency).
'''

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
    '''
    Gathers the necessary data to plot dispersion.
    '''

    if run_info["code"] == "mineos":
        
        if i_toroidal == 0: 

            mode_type = 'I'

        elif i_toroidal == 2:

            mode_type = 'T'

    # Get mode information.
    mode_info = load_eigenfreq(run_info, mode_type, i_toroidal = i_toroidal)
    n = mode_info['n']
    l = mode_info['l']
    f = mode_info['f']
    #if run_info['attenuation'] in ['none', 'linear']:

    #    f = mode_info['f']

    #elif run_info['attenuation'] == 'full':
    #    
    #    f = (mode_info['omega']*1.0E3) / (2.0 * np.pi)
    #    
    #    n = n.flatten()
    #    l = l.flatten()
    #    f = f.flatten()

    # If 'var' is specified, the points will be scaled by a variable.
    if var is not None:
    
        if var == 'Q':

            var = mode_info['Q']

            legend_label = 'Q'

        elif var == 'gamma':

            assert (run_info['attenuation'] == 'full'),\
                    'Cannot plot imaginary part of eigenfunction unless using'\
                    ' full attenuation mode.'

            var = mode_info['gamma']
            
            legend_label = '$\gamma$ (s$^{-1}$)'

        else:

            raise ValueError('Variable {:} not recognised.'.format(var))

        # Get scaling.
        i_f = get_f_lims_indices(f, f_lims)
        var_min = np.min(var[i_f])
        var_max = np.max(var[i_f])
        var_mid = (var_min + var_max)/2.0
        #
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

    # Set transparency (if scaling points).
    alpha = 0.5

    # Create axes (if necessary).
    if ax is None:

        fig = plt.figure(figsize = (9.5, 5.5))
        ax  = plt.gca()

    # Control legend.
    if var is not None:

        add_legend = True

    else:

        add_legend = False
        
    # Plot the dispersion diagram.
    ax = plot_dispersion(n, l, f, ax = ax, show = False, color = colors[0],
            c_scatter = colors[1], alpha = alpha, add_legend = add_legend,
            nlf_radial = nlf_radial, f_lims = f_lims, l_lims = l_lims,
            size_info = size_info, highlight_box = highlight_box)

    # If specified, highlight a subset of the modes.
    if path_highlight is not None:

        n_highlight, l_highlight = np.loadtxt(path_highlight, dtype = np.int).T
        i_highlight = []
        for i in range(len(n_highlight)):

            i_highlight.append(np.where((n_highlight[i] == n) & (l_highlight[i] == l))[0][0])
        
        ax.scatter(l[i_highlight], f[i_highlight], c = 'r', s = 5)

    # If requested, save the plot.
    if save:
        
        fig_name = 'dispersion.png'
        if run_info['code'] == 'mineos':

            dir_out = run_info['dir_output'] 

        elif run_info['code'] in ['ouroboros', 'ouroboros_homogeneous']:

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
    '''
    Plot a dispersion diagram.
    '''

    # Get a unique, sorted list of n-values.
    n_list = sorted(list(set(n)))

    #for i in range(len(n)):

    #    print(n[i], l[i], f[i])
    
    # Create axes (if necessary).
    if ax is None:
    
        fig = plt.figure()
        ax  = plt.gca()

    # Plot each overtone separately as a connected line.
    for ni in n_list:
        
        i = (n == ni)

        ax.plot(l[i], f[i], ls = '-', color = color, lw = 1, alpha = alpha)

    # Match the line and scatter colours.
    if colors is not None:

        c_scatter = colors
        
    # Set sizes of points.
    if size_info is None:

        sizes = 1

    else:

        sizes = size_info['sizes']

    # Draw each mode as a point.
    ax.scatter(l, f, s = sizes, c = c_scatter, alpha = alpha, zorder = 10)

    # Add radial modes if specified.
    if nlf_radial is not None:

        n_radial, l_radial, f_radial = nlf_radial
        ax.scatter(l_radial, f_radial, s = sizes, c = c_scatter, alpha = alpha, zorder = 10)
    
    # Label.
    if label is not None:

        ax.plot([], [], linestyle = '-', marker = '.', color = color, alpha = alpha, label = label)

    font_size_label = 18
    
    # Set x-limits if specified.
    if l_lims != 'auto':

        ax.set_xlim(l_lims)

    # Use integer x ticks.
    ax.xaxis.set_major_locator(MaxNLocator(integer = True))

    # Set f-limits.
    if f_lims is not None:

        if f_lims == 'auto':

            f_min = 0.0
            f_max = np.max(f)
            buff = (f_max - f_min)*0.05
            f_lims = [f_min, f_max + buff]

        ax.set_ylim(f_lims)

    # Set x label.
    if x_label is not None:

        ax.set_xlabel(x_label, fontsize = font_size_label)

    # Set y label.
    if y_label is not None:

        ax.set_ylabel(y_label, fontsize = font_size_label)

    # Add title if specified.
    if title is not None:
        
        ax.set_title(title, fontsize = font_size_label)
    
    # Add horizontal lines if specified.
    if h_lines is not None:

        for h_line in h_lines:

            ax.axhline(h_line, color = 'k', linestyle = ':')

    # Create scatter plot legend (if using scaled points).
    if (size_info is not None) and ('s_min' in size_info.keys()):

        ax.scatter([], [], c = c_scatter, s = size_info['s_min'], label = '{:>8.3e}'.format(size_info['var_min']))
        ax.scatter([], [], c = c_scatter, s = size_info['s_mid'], label = '{:>8.3e}'.format(size_info['var_mid']))
        ax.scatter([], [], c = c_scatter, s = size_info['s_max'], label = '{:>8.3e}'.format(size_info['var_max']))

        legend_label = size_info['legend_label']

    # Draw a box around a subregion (if specified).
    if highlight_box is not None:

        rect = Rectangle((highlight_box[0], highlight_box[1]),
                    highlight_box[2], highlight_box[3],
                    transform = ax.transData,
                    facecolor = 'none',
                    edgecolor = 'r')
        ax.add_artist(rect)

    # Add legend.
    if add_legend:

        plt.legend(title = legend_label)

    # Tidy layout.
    plt.tight_layout()

    # Save (if requested).
    if path_fig is not None:
        
        plt.savefig(path_fig,
            dpi         = 300,
            transparent = True,
            bbox_inches = 'tight')
            
    # Show (if requested).
    if show:
        
        plt.show()
        
    return ax

def get_f_lims_indices(f, f_lims):
    '''
    Find indices of modes within frequency limits.
    '''

    if (f_lims == 'auto') or (f_lims is None):

        num_modes = len(f)
        i_f = np.array(list(range(num_modes)), dtype = np.int)

    else:

        i_f = np.where((f > f_lims[0]) & (f < f_lims[1]))[0]

    return i_f

def get_sizes(var, var_min, var_max, s_min = 3.0, s_max = 50.0):
    '''
    Determine sizes of points, using a linear scale.
    '''
    
    var_range = var_max - var_min
    s_range = s_max - s_min
    s_mid = (s_min + s_max)/2.0
    sizes = s_min + s_range*(var - var_min)/var_range

    return sizes, s_min, s_mid, s_max

def assign_colors_by_sign(var, color_pos = 'b', color_neg = 'r'):
    '''
    Give points different colours depending on whether the variable is
    positive or negative.
    '''

    # Loop over points.
    colors = []
    for i in range(len(var)):

        if var[i] > 0.0:

            colors.append(color_pos)

        else:

            colors.append(color_neg)

    return colors, color_pos, color_neg

def get_var_aligned(mode_type, run_info_0, run_info_1, var, f_lims, i_toroidal = None):
    '''
    Align two mode lists and associated variables.
    '''

    # Load first set of mode information.
    mode_info_0 = load_eigenfreq(run_info_0, mode_type, i_toroidal = i_toroidal)
    n_0 = mode_info_0['n']
    l_0 = mode_info_0['l']
    f_0 = mode_info_0['f']
    var_0 = mode_info_0[var]

    # Remove the 2S1 mode.
    remove_2S1 = False
    if remove_2S1:

        i = np.where((n_0 == 2) & (l_0 == 1))[0]
        n_0 = np.delete(n_0, i)
        l_0 = np.delete(l_0, i)
        f_0 = np.delete(f_0, i)
        var_0 = np.delete(var_0, i)

    # Load second set of mode information.
    mode_info_1 = load_eigenfreq(run_info_1, mode_type, i_toroidal = i_toroidal)
    n_1 = mode_info_1['n']
    l_1 = mode_info_1['l']
    f_1 = mode_info_1['f']
    var_1 = mode_info_1[var]

    # Align mode lists (find common modes and sort in common order).
    n, l, i_align_0, i_align_1 = align_mode_lists(n_0, l_0, n_1, l_1)
    f_0 = f_0[i_align_0]
    f_1 = f_1[i_align_1]
    var_0 = var_0[i_align_0]
    var_1 = var_1[i_align_1]

    # Find mean values of associated variables.
    f_mean = (f_0 + f_1)/2.0
    var_mean = (var_0 + var_1)/2.0
    i_f = get_f_lims_indices(f_mean, f_lims)

    # Find deviations of variable from mean.
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
    '''
    Plot a dispersion diagram where the size of the points indicates the
    difference between two calculations.
    '''

    # Set transparency.
    alpha = 0.5
    
    # Size of points is difference in frequency.
    if diff_type == 'eigvals':

        # Align the mode lists and find differences in frequency.
        n, l, f_mean, var_diff, var, var_min, var_max  = \
            get_var_aligned(mode_type, run_info_0, run_info_1, 'f', f_lims,
                    i_toroidal = i_toroidal)

        # Assign colour values for positive and negative frequency differences.
        colors, color_pos, color_neg = assign_colors_by_sign(var_diff)

    # Size of points is different in eigenvalue.
    elif diff_type == 'eigvecs':
    
        # Locate RMS output file.
        path_rms = get_eigfunc_comparison_out_path(run_info_0, run_info_1, mode_type) 

        if not os.path.exists(path_rms):

            print('Comparison file {:} not found. Try running the misc/compare_eigenfunctions.py command.'.format(path_rms))
            return
        
        # Load eigenfunction difference information.
        n, l, f_0, f_1, rms_diff, rms_A, rms_B = np.loadtxt(path_rms).T
        n = n.astype(np.int)
        l = l.astype(np.int)
        
        remove_2S1 = False
        if remove_2S1:

            # Remove mode 2S1.
            i = np.where((n == 2) & (l == 1))[0]
            n = np.delete(n, i)
            l = np.delete(l, i)
            f_0 = np.delete(f_0, i)
            f_1 = np.delete(f_1, i)
            rms_diff = np.delete(rms_diff, i)
            rms_A = np.delete(rms_A, i)
            rms_B = np.delete(rms_B, i)
        
        # Get mean frequencies.
        f_mean = (f_0 + f_1)/2.0
        i_f = get_f_lims_indices(f_mean, f_lims)

        # Get mean RMS and assign to 'var'.
        rms_mean = (rms_A + rms_B)/2.0
        var = rms_diff/rms_mean

        var_min = np.min(var[i_f])
        var_max = np.max(var[i_f])

        # Constant colour for RMS plot (no negative values).
        colors = 'k'

    # Plot difference in Q.
    elif diff_type == 'Q':

        # Align the mode lists and find differences in Q.
        n, l, f_mean, var_diff, var, var_min, var_max  = \
            get_var_aligned(mode_type, run_info_0, run_info_1, 'Q', f_lims)

        # Assign difference colours for positive and negative Q deviations.
        colors, color_pos, color_neg = assign_colors_by_sign(var_diff)

    else:

        raise ValueError
    
    # Find range of variable and catch case of zero range.
    var_range = var_max - var_min
    if var_range == 0.0:

        var_range = 1.0

    # Set variable limits if not provided.
    if var_lims is not None:

        var_min = var_lims[0]
        var_max = var_lims[1]
        var_range = var_max - var_min

    # Set sizes by scaling the variable.
    sizes, s_min, s_mid, s_max = get_sizes(var, var_min, var_max)
    size_info = {'sizes' : sizes}

    # Create axes if needed.
    ax = None
    if ax is None:

        fig = plt.figure(figsize = (9.5, 5.5))
        ax  = plt.gca()

    # Plot the dispersion diagram with specified sizes and colours.
    ax = plot_dispersion(n, l, f_mean, l_lims = l_lims, f_lims = f_lims,
            ax = ax, size_info = size_info, show = False, colors = colors,
            alpha = alpha)

    # Make colour/size legend.
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

    # Make size legend.
    elif diff_type == 'eigvecs':

        ax.scatter([], [], c = 'k', s = s_min, label = '{:>.3f}'.format(100.0*var_min))
        ax.scatter([], [], c = 'k', s = s_max, label = '{:>.3f}'.format(100.0*var_max))
        plt.legend(title = 'RMS diff. (%)')

        fig_name = 'eigenfunction_differences.png'

    else:

        raise ValueError

    # Save (if requested).
    if save:
        
        if run_info_0['code'] in ['ouroboros', 'ouroboros_homogeneous']:

            _, _, _, dir_type = get_Ouroboros_out_dirs(run_info_0, mode_type)
            dir_out = dir_type

        elif run_info_0['code'] == 'mineos':
            
            #dir_out = run_info_0['dir_run']
            dir_out = run_info_0['dir_output']

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
    parser.add_argument("--plot_var", choices = ['Q', 'gamma'], help = 'Plot a variable. Options: Q (attenuation quality factor [dimensionless]), gamma (imaginary part of eigenvalue).')
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

    # Set frequency limits.
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

    # Check input arguments.
    #if plot_var == 'Q':
    #    
    #    assert run_info['use_attenuation'], 'Cannot plot Q for runs with no attenuation.'
    #    assert diff_type is None, 'Cannot combine --plot_var Q and --plot_diff Q. Try using only --plot_diff Q.'

    if plot_var is not None:

        assert path_input_comparison is None, 'Cannot combine --plot_var and --path_input_comparison. Try using --plot_diff.'

    # Set mode type string.
    if i_toroidal is not None:

        mode_type = 'T'
    
    elif run_info['mode_types'] == ['R']:
        
        print('Cannot plot dispersion diagram for only radial modes. Try including spheroidal modes in input file.')

    else:

        mode_type = 'S'

    # Plot dispersion diagram with scaled points.
    if diff_type is not None:
        
        assert path_input_comparison is not None
        plot_differences(run_info, run_info_comparison, mode_type,
                diff_type = diff_type,
                f_lims = f_lims,
                l_lims = l_lims,
                var_lims = var_lims,
                i_toroidal = i_toroidal)

    # Plot simple dispersion diagram.
    else:

        # Plot dispersion diagram with two runs overlaid.
        if path_input_comparison is not None:

            ax = plot_dispersion_wrapper(run_info_comparison, mode_type, i_toroidal = i_toroidal, f_lims = f_lims, l_lims = l_lims, show = False, save = False,
                        colors = ['r', 'r'], var = plot_var)
            plot_dispersion_wrapper(run_info, mode_type, i_toroidal = i_toroidal, f_lims = f_lims, l_lims = l_lims, ax = ax, show = True, save = True,
                        colors = ['b', 'b'], var = plot_var)

        # Plot one dispersion diagram.
        else:

            plot_dispersion_wrapper(run_info, mode_type, i_toroidal = i_toroidal, f_lims = f_lims, l_lims = l_lims, var = plot_var, path_highlight = path_highlight,
                                        highlight_box = highlight_box)

    return

if __name__ == '__main__':

    main()
