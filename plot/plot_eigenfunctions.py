'''
Plot eigenfunction(s) for a single mode.
'''

import argparse
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from Ouroboros.common import (get_Ouroboros_out_dirs, get_r_fluid_solid_boundary,
                            load_eigenfreq, load_eigenfunc,
                            load_model, mkdir_if_not_exist,
                            read_input_file)
from Ouroboros.misc.compare_eigenfunctions import (check_sign_R,
        check_sign_S, check_sign_P, check_sign_T)

def get_title_str(mode_type, n, l, code, i_toroidal = None):
    '''
    Format the mode's name for a title.
    '''

    # Get title string information.
    if mode_type in ['R', 'S']:

        mode_type_for_title = 'S'

    else:
        
        if code in ['mineos', 'ouroboros_homogeneous']:

            mode_type_for_title = mode_type

        elif code == 'ouroboros':

            if i_toroidal == 0:
                
                mode_type_for_title = 'I'
            
            else:
                
                mode_type_for_title = 'T'

        else:

            raise ValueError

    title = '$_{{{:d}}}${:}$_{{{:d}}}$'.format(n, mode_type_for_title, l)

    return title

def plot_eigenfunc_wrapper(run_info, mode_type, n, l, i_toroidal = None, ax = None, ax_imag = None, save = True, show = True, transparent = True, linestyle = '-', label_suffix = '', plot_gradient = False, plot_potential = False, x_label = 'default', norm_func = 'mineos', units = 'SI', alpha = 1.0, r_lims = None, no_title = False, relaxation = False, duplicate = False): 
    '''
    Wrapper script which gathers the necessary data to plot the eigenfunction.
    '''
    
    # Get model information for axis limits, scaling and horizontal lines.
    if run_info['code'] == 'ouroboros_homogeneous':

        r_srf = run_info['r']*1.0E-3
        i_fluid = []
        r_solid_fluid_boundary = []

    else:

        model = load_model(run_info['path_model'])
        # Convert to km.
        model['r'] = model['r']*1.0E-3
        # r_srf Radius of planet.
        # r_solid_fluid_boundary    List of radii of solid-fluid boundaries.
        r_srf = model['r'][-1]
        i_fluid, r_solid_fluid_boundary, _ = get_r_fluid_solid_boundary(model['r'], model['v_s'])

    # Get frequency information.
    mode_info = load_eigenfreq(run_info, mode_type, i_toroidal = i_toroidal, n_q = n, l_q = l,
                                relaxation = relaxation, duplicate = duplicate)
    f = mode_info['f']

    # Get normalisation arguments.
    f_rad_per_s = f*1.0E-3*2.0*np.pi
    normalisation_args = {'norm_func' : norm_func, 'units' : units}
    normalisation_args['omega'] = f_rad_per_s

    # Get eigenfunction information.
    eigfunc_dict = load_eigenfunc(run_info, mode_type, n, l,
                        i_toroidal = i_toroidal,
                        norm_args = normalisation_args,
                        relaxation = relaxation,
                        duplicate = duplicate)
    eigfunc_dict['r'] = eigfunc_dict['r']*1.0E-3 # Convert to km.
    
    # Get title string.
    title = get_title_str(mode_type, n, l, run_info['code'], i_toroidal = i_toroidal)

    # Check the sign of the plotting variable.
    if plot_potential:
        
        sign = check_sign_P(eigfunc_dict['r'], eigfunc_dict['P'])

    else:

        if mode_type == 'S':

            sign = check_sign_S(eigfunc_dict['r'], eigfunc_dict['U'],
                                eigfunc_dict['V'])

        elif mode_type == 'R':

            sign = check_sign_R(eigfunc_dict['r'], eigfunc_dict['U'])

        elif mode_type in ['I', 'T']:

            sign = check_sign_T(eigfunc_dict['r'], eigfunc_dict['W'])

    # Change sign to always be positive.
    if sign < 0: 

        for val in eigfunc_dict:

            if val != 'r':

                eigfunc_dict[val] = eigfunc_dict[val]*-1.0
    
    # Find maximum value(s) of plotting variable.
    if plot_gradient:

        if plot_potential:

            vals = eigfunc_dict['Pp'] 

        else:

            if mode_type == 'R':

                vals = eigfunc_dict['Up'] 

            elif mode_type == 'S':

                vals = np.concatenate([eigfunc_dict['Up'], eigfunc_dict['Vp']])

            elif mode_type == 'T':

                vals = eigfunc_dict['Wp']

            else:

                raise ValueError
        

    else:

        if plot_potential:

            vals = eigfunc_dict['P']

        else:

            if mode_type == 'R':
                
                vals = eigfunc_dict['U']

            elif mode_type == 'S':

                vals = np.concatenate([eigfunc_dict['U'], eigfunc_dict['V']])

            elif mode_type in ['I', 'T']:

                vals = eigfunc_dict['W']

            else:

                raise ValueError

    # Get maximum value.
    max_ = np.max(np.abs(vals))

    # Mineos saves the toroidal eigenfunction in regions where it has a
    # value of 0. This is clipped.
    if run_info['code'] == 'mineos':
        
        clip_zero = True
        if mode_type in ['T', 'I']:

            if clip_zero:

                i_nonzero = np.nonzero(eigfunc_dict['W'])[0]

            i_0 = i_nonzero[0]
            i_1 = i_nonzero[-1]

            eigfunc_dict['r'] = eigfunc_dict['r'][i_0 : i_1]
            for val in ['W', 'Wp']:

                eigfunc_dict[val] = eigfunc_dict[val][i_0 : i_1]

    # Create axes if not provided.
    if r_lims is None:

        r_range = np.max(eigfunc_dict['r']) - np.min(eigfunc_dict['r'])
        r_frac = r_range/r_srf
        imag_y = 11.0*r_frac

    else:

        imag_y = 7.0

    if run_info['attenuation'] == 'full':

        imag_x = 11.0

    else:

        imag_x = 5.5

    if ax is None:
        
        if run_info['attenuation'] == 'full':

            fig, ax_arr = plt.subplots(1, 2, figsize = (imag_x, imag_y),
                            sharey = True)
            ax = ax_arr[0]
            ax_imag = ax_arr[1]

        else:

            fig = plt.figure(figsize = (imag_x, imag_y))
            ax  = plt.gca()

            ax_imag = None

    else:
    
        imag_x_pre, imag_y_pre = ax.figure.get_size_inches()
        if imag_y_pre < imag_y:

            ax.figure.set_size_inches((imag_x, imag_y))

    # Arguments for all possibilities.
    if run_info['attenuation'] == 'full':
    
        if x_label == 'default':

            x_label = None

    else:

        if x_label == 'default':

            x_label = 'Eigenfunction'

    common_args = {'ax' : ax, 'show' : False,
            'title' : title, 'x_label' : x_label, 'alpha' : alpha,
            'r_lims' : r_lims}

    if no_title:

        common_args['title'] = None

    # Plot.
    if plot_gradient:

        if plot_potential:

            if run_info['attenuation'] == 'full':

                raise NotImplementedError

            plot_P(eigfunc_dict['r'], eigfunc_dict['Pp'],
                    h_lines = r_solid_fluid_boundary,
                    linestyle = linestyle,
                    label_suffix = label_suffix,
                    **common_args)

        else:

            if mode_type == 'R':

                if run_info['attenuation'] == 'full':

                    raise NotImplementedError

                plot_eigenfunc_R_or_T(eigfunc_dict['r'], eigfunc_dict['Up'],
                        h_lines = r_solid_fluid_boundary,
                        linestyle = linestyle,
                        label = 'U{:}'.format(label_suffix),
                        **common_args)
            
            elif mode_type == 'S':

                if run_info['attenuation'] == 'full':

                    raise NotImplementedError

                plot_eigenfunc_S(eigfunc_dict['r'], eigfunc_dict['Up'], eigfunc_dict['Vp'],
                        h_lines = r_solid_fluid_boundary,
                        linestyles = [linestyle, linestyle],
                        label_suffix = label_suffix,
                        **common_args)

            elif mode_type in ['T', 'I']:

                if run_info['attenuation'] == 'full':

                    raise NotImplementedError
                
                plot_eigenfunc_R_or_T(eigfunc_dict['r'], eigfunc_dict['Wp'],
                        h_lines = None,
                        linestyle = linestyle,
                        label = 'W{:}'.format(label_suffix),
                        **common_args)

    elif plot_potential:

        if run_info['attenuation'] == 'full':

            raise NotImplementedError

        plot_P(eigfunc_dict['r'], eigfunc_dict['P'],
                h_lines = r_solid_fluid_boundary,
                linestyle = linestyle,
                label_suffix = label_suffix,
                **common_args)

    else:

        if mode_type == 'R':

            if run_info['attenuation'] == 'full':

                raise NotImplementedError

            plot_eigenfunc_R_or_T(eigfunc_dict['r'], eigfunc_dict['U'],
                    h_lines = r_solid_fluid_boundary,
                    linestyle = linestyle,
                    label = 'U{:}'.format(label_suffix),
                    **common_args)
        
        elif mode_type == 'S':

            if run_info['attenuation'] == 'full':

                common_args['ax'] = ax_imag
                common_args['x_label'] = 'Imaginary'

                plot_eigenfunc_S(eigfunc_dict['r'], eigfunc_dict['U_im'],
                        eigfunc_dict['V_im'],
                        h_lines = None,
                        linestyles = [linestyle, linestyle],
                        label_suffix = label_suffix,
                        y_label = None,
                        **common_args)

                common_args['ax'] = ax
                common_args['x_label'] = 'Real'

            plot_eigenfunc_S(eigfunc_dict['r'], eigfunc_dict['U'], eigfunc_dict['V'],
                    h_lines = r_solid_fluid_boundary,
                    linestyles = [linestyle, linestyle],
                    label_suffix = label_suffix,
                    **common_args)

        elif mode_type in ['T', 'I']:

            if run_info['attenuation'] == 'full':

                common_args['ax'] = ax_imag
                common_args['x_label'] = 'Imaginary'
                #common_args['title'] = None

                plot_eigenfunc_R_or_T(eigfunc_dict['r'], eigfunc_dict['W_im'],
                        h_lines = None,
                        linestyle = linestyle,
                        label = 'W{:}'.format(label_suffix),
                        y_label = None,
                        **common_args)

                common_args['ax'] = ax
                common_args['x_label'] = 'Real'

            plot_eigenfunc_R_or_T(eigfunc_dict['r'], eigfunc_dict['W'],
                    h_lines = None,
                    linestyle = linestyle,
                    label = 'W{:}'.format(label_suffix),
                    **common_args)

    if run_info['attenuation'] == 'full':
        
        font_size_title = 36
        plt.suptitle(title, fontsize = font_size_title)

    # Make the background transparent (if requested).
    if transparent:
        
        fig = plt.gcf()
        set_patch_facecolors(fig, ax) 

    # Set tight layout.
    plt.tight_layout()
    
    # Save (if requested).
    if save:

        method_str = run_info['code']

        if plot_gradient:

            gradient_str = '_gradient'

        else:

            gradient_str = ''

        if plot_potential:

            var_str = 'potential'

        else:

            var_str = 'eigfunc'

        fig_name = '{:}{:}_{:}'.format(var_str, gradient_str, method_str)
        
        if run_info['code'] == 'mineos':
            
            dir_out = run_info['dir_output']
            dir_plot = os.path.join(dir_out, 'plots')

        elif run_info['code'] in ['ouroboros', 'ouroboros_homogeneous']:
            
            _, _, _, dir_out = get_Ouroboros_out_dirs(run_info, mode_type)

        else:

            raise ValueError

        dir_plot = os.path.join(dir_out, 'plots')
        mkdir_if_not_exist(dir_plot)

        if mode_type in ['S', 'R']:

            fig_name = '{:}_{:>05d}_{:}_{:>05d}_{:1d}.png'.format(fig_name, n, mode_type, l, run_info['grav_switch'])

        else:

            if run_info['code'] == 'mineos':

                fig_name = '{:}_{:>05d}_{:}_{:>05d}_{:1d}.png'.format(fig_name, n, mode_type, l, run_info['grav_switch'])

            elif run_info['code'] == 'ouroboros':

                fig_name = '{:}_{:>05d}_{:}{:1d}_{:>05d}_{:1d}.png'.format(fig_name, n, mode_type, i_toroidal, l, run_info['grav_switch'])

            elif run_info['code'] == 'ouroboros_homogeneous':

                fig_name = '{:}_{:>05d}_{:}_{:>05d}_{:>1d}.png'.format(fig_name, n, mode_type, l, run_info['grav_switch'])

            else:

                raise ValueError

        fig_path = os.path.join(dir_plot, fig_name)
        print('Saving figure to {:}'.format(fig_path))
        plt.savefig(fig_path, dpi = 300, bbox_inches = 'tight')

    if show:

        plt.show()

    return ax, ax_imag

def plot_eigenfunc_S(r, U, V, ax = None, h_lines = None, x_label = 'Eigenfunction', y_label = 'Radial coordinate / km', title = None, show = True, add_legend = True, colors = ['r', 'b'], linestyles = ['-', '-'], label_suffix = '', alpha = 1.0, legend_loc = 'best', font_size_label = 12, r_lims = None):
    '''
    Plot spheroidal eigenfunction.
    '''
    
    # Create axes if none provided.
    if ax is None:

        fig = plt.figure()
        ax  = plt.gca()
       
    # Set labels of lines.
    U_label = 'U'
    V_label = 'V'
    U_label = U_label + label_suffix
    V_label = V_label + label_suffix

    # Plot eigenfunctions.
    ax.plot(U, r, label = U_label, color = colors[0], linestyle = linestyles[0], alpha = alpha)
    ax.plot(V, r, label = V_label, color = colors[1], linestyle = linestyles[1], alpha = alpha)

    # Set axis limits.
    max_abs_U_plot = np.max(np.abs(U))
    max_abs_V_plot = np.max(np.abs(V))
    E_max = np.max([max_abs_U_plot, max_abs_V_plot])
    
    # Tidy axes.
    tidy_axes(ax, r, E_max, h_lines = h_lines, add_legend = add_legend, legend_loc = legend_loc, title = title, x_label = x_label, y_label = y_label, r_lims = r_lims)

    if show:

        plt.show()

    return ax, E_max

def plot_P(r, P, ax = None, h_lines = None, x_label = 'Potential', y_label = 'Radial coordinate / km', title = None, show = True, add_legend = True, color = 'r', linestyle = '-', label_suffix = '', alpha = 1.0, legend_loc = 'best', font_size_label = 12):
    '''
    Plot potential eigenfunction.
    '''
    
    # Create axes if none provided.
    if ax is None:

        fig = plt.figure()
        ax  = plt.gca()
    
    # Set label for line.
    label = 'P'
    label = label + label_suffix
    
    # Plot eigenfunction.
    ax.plot(P, r, label = label, color = color, linestyle = linestyle, alpha = alpha)

    # Determine axis limits.
    P_max = np.max(np.abs(P))

    # Tidy axes.
    tidy_axes(ax, r, P_max, h_lines = h_lines, add_legend = add_legend, legend_loc = legend_loc, title = title, x_label = x_label, y_label = y_label)

    # Show (if requested).
    if show:

        plt.show()

    return ax, P_max

def tidy_axes(ax, r, E_max, h_lines = None, add_legend = True, legend_loc = 'best', title = None, x_label = 'Eigenfunction', y_label = 'Radius / km', r_lims = None):
    '''
    Make the axes look neater.
    '''
    
    # Set font sizes.
    font_size_label = 16
    font_size_title = 36 

    # Draw horizontal lines.
    if h_lines is not None:

        for h_line in h_lines:

            ax.axhline(h_line, linestyle = ':', color = 'k')

    # Draw vertical line at x = 0.
    ax.axvline(linestyle = ':', color = 'k')
    
    # Set eigenfunction axis limits.
    buff = 1.05
    ax.set_xlim([-buff*E_max, buff*E_max])

    # Set radius axis limits.
    if r_lims is None:

        ax.set_ylim([np.min(r), np.max(r)])

    else:

        ax.set_ylim(r_lims)
    
    # Add legend.
    if add_legend:

        ax.legend(loc = legend_loc)

    # Add title.
    if title is not None:
        
        ax.set_title(title, fontsize = font_size_title)

    # Label x axis. 
    if x_label is not None:

        ax.set_xlabel(x_label, fontsize = font_size_label)
    
    # Label y axis.
    if y_label is not None:

        ax.set_ylabel(y_label, fontsize = font_size_label)

    return

def set_patch_facecolors(fig, ax):
    '''
    Make transparent plot background.
    '''

    ax.patch.set_facecolor('white')
    ax.patch.set_alpha(1.0)
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(0.0)

    return

def plot_eigenfunc_R_or_T(r, U_or_W, ax = None, show = False, h_lines = None, add_legend = True, legend_loc = 'best', title = None, label = None, x_label = 'Eigenfunction', y_label = 'Radial coordinate / km', linestyle = '-', alpha = 1.0, r_lims = None):
    '''
    Plot eigenfunction for radial or toroidal mode.
    '''

    # Plot the line.
    ax.plot(U_or_W, r, color = 'r', label = label,
            linestyle = linestyle, alpha = alpha)

    # Get eigenfunction axis limits.
    max_abs_U_or_W_plot = np.max(np.abs(U_or_W))
    
    # Tidy up axis.
    tidy_axes(ax, r, max_abs_U_or_W_plot, h_lines = h_lines,
            add_legend = add_legend, legend_loc = legend_loc,
            title = title, x_label = x_label, y_label = y_label,
            r_lims = r_lims)

    return

def get_label_suffixes(path_compare, code, code_compare, plot_gradient):
    '''
    Define suffixes for line labels.
    '''

    if plot_gradient:

        if code == 'mineos':

            label_suffix = '\' (Mineos)'
        
        elif code == 'ouroboros':

            #label_suffix = '\' (Ouroboros)'
            label_suffix = '\' (RadialPNM)'

        else:

            raise ValueError

        if path_compare is not None:

            if code_compare == 'mineos':

                #label_suffix_compare = '\' (Mineos)'
                label_suffix_compare = ''
            
            elif code_compare == 'ouroboros':

                #label_suffix_compare = '\' (Ouroboros)'
                #label_suffix_compare = '\' (RadialPNM)'
                label_suffix_compare = ''

            else:

                raise ValueError

        else:

            label_suffix_compare = None

    else:

        if code == 'mineos':

            label_suffix = ' (Mineos)'
        
        elif code == 'ouroboros':

            #label_suffix = ' (Ouroboros)'
            #label_suffix = ' (RadialPNM)'
            label_suffix = ''

        elif code == 'ouroboros_homogeneous':

            label_suffix = ''

        else:

            raise ValueError

        if path_compare is not None:

            if code_compare == 'mineos':

                #label_suffix_compare = ' (Mineos)'
                label_suffix_compare = ''
            
            elif code_compare == 'ouroboros':

                #label_suffix_compare = ' (Ouroboros)'
                label_suffix_compare = ''

            else:

                raise ValueError

        else:

            label_suffix_compare = None

    return label_suffix, label_suffix_compare

def main():

    # Read input arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_input_file", help = "File path (relative or absolute) to Ouroboros input file.")
    parser.add_argument("mode_type", choices = ['R', 'S', 'T', 'I'], help = 'Mode type (radial, spheroidal or toroidal). Option I is for use with Mineos to plot inner-core toroidal modes. See the --toroidal flag for plotting toroidal modes with Ouroboros.')
    parser.add_argument("n", type = int, help = "Plot mode with radial order n.")
    parser.add_argument("l", type = int, help = "Plot mode with angular order l (must be 0 for radial modes).")
    parser.add_argument("--toroidal", dest = "layer_number", help = "Plot toroidal eigenfunction for the solid shell given by LAYER_NUMBER (0 is outermost solid shell).", type = int)
    parser.add_argument("--gradient", action = "store_true", help = "Include this flag to plot eigenfunction gradients (default: plot eigenfunctions).")
    parser.add_argument("--potential", action = "store_true", help = "Include this flag to plot potential (default: plot eigenfunctions).")
    parser.add_argument("--path_compare", help = "Provide input path to plot a second eigenfunction for comparison.")
    parser.add_argument("--norm_func", choices = ['mineos', 'DT'], default = 'DT', help = "Specify normalisation function. \'mineos\' is the normalisation function used by Mineos and Ouroboros. \'DT\' is the normalisation function used in the Dahlen and Tromp textbook. It does not include the factor of k. See also the --units flag. For more detail, see Ouroboros/doc/Ouroboros_normalisation_notes.pdf.")
    parser.add_argument("--units", choices = ['SI', 'ouroboros', 'mineos'], default = 'mineos', help = 'Specify units used when applying normalisation to eigenfunction. \'SI\' is SI units. \'mineos\' is Mineos units. \'ouroboros\' is Ouroboros units. See also the --norm_func flag. For more detail, see Ouroboros/doc/Ouroboros_normalisation_notes.pdf.')
    parser.add_argument("--r_lims", nargs = 2, type = float, help = 'Specify radius limits of plot (km).')
    parser.add_argument("--relaxation", action = 'store_true', help = 'Plot a relaxation mode (instead of oscillation mode). Note: only available when attenuation == \'full\'.')
    parser.add_argument("--duplicate", action = 'store_true', help = 'Plot a duplicate mode. Note 1: only available when attenuation == \'full\'. Note 2: duplicate modes are not sorted by (n, l) but by a single index n (sorted by real part of frequency). l is ignored.')
    args = parser.parse_args()

    # Rename input arguments.
    path_input = args.path_to_input_file
    mode_type  = args.mode_type
    n           = args.n
    l           = args.l
    i_toroidal = args.layer_number
    plot_gradient = args.gradient
    plot_potential = args.potential
    path_compare = args.path_compare
    norm_func = args.norm_func
    units = args.units
    r_lims = args.r_lims
    relaxation = args.relaxation
    duplicate = args.duplicate

    # Check input arguments.
    if mode_type == 'R':

        assert l == 0, 'Must have l = 0 for radial modes.'

    # Read input file.
    run_info = read_input_file(path_input)
    if path_compare is not None:

        run_info_compare = read_input_file(path_compare)
        code_compare = run_info_compare['code']

    else:

        code_compare = None

    # Get x label.
    if plot_potential:

        if plot_gradient:

            x_label = 'Potential gradient'

        else:

            x_label = 'Potential'

    else:

        if plot_gradient:

            x_label = 'Eigenfunction gradient'

        else:

            x_label = 'Eigenfunction'
    
    # Get label suffixes.
    label_suffix, label_suffix_compare = get_label_suffixes(path_compare,
            run_info['code'], code_compare, plot_gradient)

    # Plot.
    if path_compare is not None:

        if run_info['attenuation'] == 'full' or run_info_compare['attenuation'] == 'full':

            no_title = True
        else:

            no_title = False

        if ((run_info['attenuation'] == 'full') and 
            (run_info_compare['attenuation'] != 'full')):

            # Plot two eigenfunctions overlaid on same plot.
            ax, ax_imag = plot_eigenfunc_wrapper(run_info, mode_type, n, l,
                    i_toroidal = i_toroidal, ax = None, show = False,
                    transparent = False, save = False, linestyle = '-',
                    label_suffix = label_suffix_compare, x_label = None,
                    norm_func = norm_func, units = units,
                    plot_gradient = plot_gradient, plot_potential = plot_potential,
                    alpha = 0.5, no_title = no_title) 

            plot_eigenfunc_wrapper(run_info_compare, mode_type, n, l,
                    i_toroidal = i_toroidal, ax = ax, ax_imag = ax_imag, show = True,
                    label_suffix = label_suffix, plot_gradient = plot_gradient,
                    plot_potential = plot_potential, x_label = x_label,
                    norm_func = norm_func, units = units, linestyle = ':',
                    no_title = no_title)

        else:

            # Plot two eigenfunctions overlaid on same plot.
            ax, ax_imag = plot_eigenfunc_wrapper(run_info_compare, mode_type, n, l,
                    i_toroidal = i_toroidal, ax = None, ax_imag = None, show = False,
                    transparent = False, save = False, linestyle = ':',
                    label_suffix = label_suffix_compare, x_label = None,
                    norm_func = norm_func, units = units,
                    plot_gradient = plot_gradient, plot_potential = plot_potential,
                    no_title = no_title)

            plot_eigenfunc_wrapper(run_info, mode_type, n, l,
                    i_toroidal = i_toroidal, ax = ax, ax_imag = ax_imag, show = True,
                    label_suffix = label_suffix, plot_gradient = plot_gradient,
                    plot_potential = plot_potential, x_label = x_label,
                    norm_func = norm_func, units = units, alpha = 0.5,
                    no_title = no_title) 

    else:

        if run_info['attenuation'] == 'full':

            no_title = True

        else:

            no_title = False

        # Plot a single eigenfunction.
        plot_eigenfunc_wrapper(run_info, mode_type, n, l,
                i_toroidal = i_toroidal, ax = None,
                plot_gradient = plot_gradient,
                plot_potential = plot_potential,
                label_suffix = label_suffix, x_label = x_label,
                norm_func = norm_func, units = units, r_lims = r_lims,
                no_title = no_title,
                relaxation = relaxation,
                duplicate = duplicate) 

    return

if __name__ == '__main__':

    main()
