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
from Ouroboros.misc.compare_eigenfunctions import check_sign_R, check_sign_S, check_sign_P

def get_title_str(mode_type, n, l, code):

    # Get title string information.
    if mode_type in ['R', 'S']:

        mode_type_for_title = 'S'

    else:
        
        if code == 'mineos':

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

def plot_eigenfunc_wrapper(run_info, mode_type, n, l, i_toroidal = None, ax = None, save = True, show = True, transparent = True, linestyle = '-', label_suffix = '', plot_gradient = False, plot_potential = False, x_label = 'Eigenfunction', norm_func = 'mineos', units = 'SI', alpha = 1.0, r_lims = None): 
    
    # Get model information for axis limits, scaling and horizontal lines.
    model = load_model(run_info['path_model'])
    # Convert to km.
    model['r'] = model['r']*1.0E-3
    # r_srf Radius of planet.
    # r_solid_fluid_boundary    List of radii of solid-fluid boundaries.
    r_srf = model['r'][-1]
    i_fluid, r_solid_fluid_boundary, _ = get_r_fluid_solid_boundary(model['r'], model['v_s'])

    # Get frequency information.
    mode_info = load_eigenfreq(run_info, mode_type, i_toroidal = i_toroidal, n_q = n, l_q = l)
    f = mode_info['f']

    # Get normalisation arguments.
    f_rad_per_s = f*1.0E-3*2.0*np.pi
    normalisation_args = {'norm_func' : norm_func, 'units' : units}
    normalisation_args['omega'] = f_rad_per_s

    # Get eigenfunction information.
    eigfunc_dict = load_eigenfunc(run_info, mode_type, n, l, i_toroidal = i_toroidal, norm_args = normalisation_args)
    eigfunc_dict['r'] = eigfunc_dict['r']*1.0E-3 # Convert to km.

    # Get title string.
    title = get_title_str(mode_type, n, l, run_info['code'])

    if plot_potential:
        
        sign = check_sign_P(eigfunc_dict['r'], eigfunc_dict['P'])

    else:

        if mode_type == 'S':

            sign = check_sign_S(eigfunc_dict['r'], eigfunc_dict['U'],
                                eigfunc_dict['V'])

        elif mode_type == 'R':

            sign = check_sign_R(eigfunc_dict['r'], eigfunc_dict['U'])
    
    # Find axis limits.
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

            # Find axis limits.
            if mode_type == 'R':
                
                vals = eigfunc_dict['U']

            elif mode_type == 'S':

                vals = np.concatenate([eigfunc_dict['U'], eigfunc_dict['V']])

            elif mode_type in ['I', 'T']:

                vals = eigfunc_dict['W']

            else:

                raise ValueError

    if sign < 0: 

        for val in eigfunc_dict:

            if val != 'r':

                eigfunc_dict[val] = eigfunc_dict[val]*-1.0

    max_ = np.max(np.abs(vals))

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
        imag_x = 5.5
        imag_y = 11.0*r_frac

    else:

        imag_x = 5.5
        imag_y = 7.0

    if ax is None:

        fig = plt.figure(figsize = (imag_x, imag_y))
        ax  = plt.gca()

    else:
    
        imag_x_pre, imag_y_pre = ax.figure.get_size_inches()
        if imag_y_pre < imag_y:

            ax.figure.set_size_inches((imag_x, imag_y))

    # Arguments for all possibilities.
    common_args = {'ax' : ax, 'show' : False, 'title' : title,
            'x_label' : x_label, 'alpha' : alpha,
            'r_lims' : r_lims}

    if plot_gradient:

        if plot_potential:

            plot_P(eigfunc_dict['r'], eigfunc_dict['Pp'],
                    h_lines = r_solid_fluid_boundary,
                    linestyle = linestyle,
                    label_suffix = label_suffix,
                    **common_args)

        else:

            if mode_type == 'R':

                plot_eigenfunc_R_or_T(eigfunc_dict['r'], eigfunc_dict['Up'],
                        h_lines = r_solid_fluid_boundary,
                        linestyle = linestyle,
                        label = 'U{:}'.format(label_suffix),
                        **common_args)
            
            elif mode_type == 'S':

                plot_eigenfunc_S(eigfunc_dict['r'], eigfunc_dict['Up'], eigfunc_dict['Vp'],
                        h_lines = r_solid_fluid_boundary,
                        linestyles = [linestyle, linestyle],
                        label_suffix = label_suffix,
                        **common_args)

            elif mode_type in ['T', 'I']:
                
                plot_eigenfunc_R_or_T(eigfunc_dict['r'], eigfunc_dict['Wp'],
                        h_lines = None,
                        linestyle = linestyle,
                        label = 'W{:}'.format(label_suffix),
                        **common_args)

    elif plot_potential:

        plot_P(eigfunc_dict['r'], eigfunc_dict['P'],
                h_lines = r_solid_fluid_boundary,
                linestyle = linestyle,
                label_suffix = label_suffix,
                **common_args)

    else:

        if mode_type == 'R':

            plot_eigenfunc_R_or_T(eigfunc_dict['r'], eigfunc_dict['U'],
                    h_lines = r_solid_fluid_boundary,
                    linestyle = linestyle,
                    label = 'U{:}'.format(label_suffix),
                    **common_args)
        
        elif mode_type == 'S':

            plot_eigenfunc_S(eigfunc_dict['r'], eigfunc_dict['U'], eigfunc_dict['V'],
                    h_lines = r_solid_fluid_boundary,
                    linestyles = [linestyle, linestyle],
                    label_suffix = label_suffix,
                    **common_args)

        elif mode_type in ['T', 'I']:
            
            plot_eigenfunc_R_or_T(eigfunc_dict['r'], eigfunc_dict['W'],
                    h_lines = None,
                    linestyle = linestyle,
                    label = 'W{:}'.format(label_suffix),
                    **common_args)

    if transparent:
        
        fig = plt.gcf()
        set_patch_facecolors(fig, ax) 

    plt.tight_layout()
    
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

        elif run_info['code'] == 'ouroboros':
            
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

            else:

                raise ValueError

        fig_path = os.path.join(dir_plot, fig_name)
        print('Saving figure to {:}'.format(fig_path))
        plt.savefig(fig_path, dpi = 300, bbox_inches = 'tight')

    if show:

        plt.show()

    return ax

def plot_eigenfunc_S(r, U, V, ax = None, h_lines = None, x_label = 'Eigenfunction', y_label = 'Radial coordinate / km', title = None, show = True, add_legend = True, colors = ['r', 'b'], linestyles = ['-', '-'], label_suffix = '', alpha = 1.0, legend_loc = 'best', font_size_label = 12, r_lims = None):
    
    if ax is None:

        fig = plt.figure()
        ax  = plt.gca()
       
    U_label = 'U'
    V_label = 'V'
    U_label = U_label + label_suffix
    V_label = V_label + label_suffix

    ax.plot(U, r, label = U_label, color = colors[0], linestyle = linestyles[0], alpha = alpha)
    ax.plot(V, r, label = V_label, color = colors[1], linestyle = linestyles[1], alpha = alpha)

    max_abs_U_plot = np.max(np.abs(U))
    max_abs_V_plot = np.max(np.abs(V))
    E_max = np.max([max_abs_U_plot, max_abs_V_plot])
    
    tidy_axes(ax, r, E_max, h_lines = h_lines, add_legend = add_legend, legend_loc = legend_loc, title = title, x_label = x_label, y_label = y_label, r_lims = r_lims)

    if show:

        plt.show()

    return ax, E_max

def plot_P(r, P, ax = None, h_lines = None, x_label = 'Potential', y_label = 'Radial coordinate / km', title = None, show = True, add_legend = True, color = 'r', linestyle = '-', label_suffix = '', alpha = 1.0, legend_loc = 'best', font_size_label = 12):
    
    if ax is None:

        fig = plt.figure()
        ax  = plt.gca()
    
    label = 'P'
    label = label + label_suffix
    
    ax.plot(P, r, label = label, color = color, linestyle = linestyle, alpha = alpha)

    P_max = np.max(np.abs(P))

    tidy_axes(ax, r, P_max, h_lines = h_lines, add_legend = add_legend, legend_loc = legend_loc, title = title, x_label = x_label, y_label = y_label)

    if show:

        plt.show()

    return ax, P_max

def tidy_axes(ax, r, E_max, h_lines = None, add_legend = True, legend_loc = 'best', title = None, x_label = 'Eigenfunction', y_label = 'Radius / km', r_lims = None):
    
    font_size_label = 16
    font_size_title = 36 

    if h_lines is not None:

        for h_line in h_lines:

            ax.axhline(h_line, linestyle = ':', color = 'k')

    ax.axvline(linestyle = ':', color = 'k')
    
    buff = 1.05
    ax.set_xlim([-buff*E_max, buff*E_max])

    if r_lims is None:

        ax.set_ylim([np.min(r), np.max(r)])

    else:

        ax.set_ylim(r_lims)
    
    if add_legend:

        ax.legend(loc = legend_loc)

    if title is not None:
        
        ax.set_title(title, fontsize = font_size_title)

    if x_label is not None:

        ax.set_xlabel(x_label, fontsize = font_size_label)
    
    if y_label is not None:

        ax.set_ylabel(y_label, fontsize = font_size_label)

    return

def set_patch_facecolors(fig, ax):

    ax.patch.set_facecolor('white')
    ax.patch.set_alpha(1.0)
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(0.0)

    return

def plot_eigenfunc_R_or_T(r, U_or_W, ax = None, show = False, h_lines = None, add_legend = True, legend_loc = 'best', title = None, label = None, x_label = 'Eigenfunction', y_label = 'Radial coordinate / km', linestyle = '-', alpha = 1.0):

    ax.plot(U_or_W, r, color = 'r', label = label, linestyle = linestyle, alpha = alpha)

    #ax.axhline(3480.0, color = 'k', linestyle = ':')
    ax.axvline(0.0, color = 'k', linestyle = ':')

    max_abs_U_or_W_plot = np.max(np.abs(U_or_W))
    
    tidy_axes(ax, r, max_abs_U_or_W_plot, h_lines = h_lines, add_legend = add_legend, legend_loc = legend_loc, title = title, x_label = x_label, y_label = y_label)

    return

def plot_eigenfunc_T(r, W, ax = None, show = False):

    ax.plot(W, r, label = 'W')

    #ax.axhline(3480.0, color = 'k', linestyle = ':')
    ax.axvline(0.0, color = 'k', linestyle = ':')

    tidy_axes(ax, r)

    return

def get_label_suffixes(path_compare, code, code_compare, plot_gradient):

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
    parser.add_argument("mode_type", choices = ['R', 'S', 'T', 'I'], help = 'Mode type (radial, spheroidal or toroidal). Option I is for use with --use_mineos flag to plot inner-core toroidal modes. See the --toroidal flag for plotting toroidal modes with Ouroboros.')
    parser.add_argument("n", type = int, help = "Plot mode with radial order n.")
    parser.add_argument("l", type = int, help = "Plot mode with angular order l (must be 0 for radial modes).")
    parser.add_argument("--toroidal", dest = "layer_number", help = "Plot toroidal eigenfunction for the solid shell given by LAYER_NUMBER (0 is outermost solid shell).", type = int)
    parser.add_argument("--gradient", action = "store_true", help = "Include this flag to plot eigenfunction gradients (default: plot eigenfunctions).")
    parser.add_argument("--potential", action = "store_true", help = "Include this flag to plot potential (default: plot eigenfunctions).")
    parser.add_argument("--use_mineos", action = "store_true", help = "Plot Mineos eigenfunction (default: Ouroboros).")
    parser.add_argument("--path_compare", help = "Provide input path to plot a second eigenfunction for comparison.")
    parser.add_argument("--norm_func", choices = ['mineos', 'DT'], default = 'DT', help = "Specify normalisation function. \'mineos\' is the normalisation function used by Mineos and Ouroboros. \'DT\' is the normalisation function used in the Dahlen and Tromp textbook. It does not include the factor of k. See also the --units flag. For more detail, see Ouroboros/doc/Ouroboros_normalisation_notes.pdf.")
    parser.add_argument("--units", choices = ['SI', 'ouroboros', 'mineos'], default = 'mineos', help = 'Specify units used when applying normalisation to eigenfunction. \'SI\' is SI units. \'mineos\' is Mineos units. \'ouroboros\' is Ouroboros units. See also the --norm_func flag. For more detail, see Ouroboros/doc/Ouroboros_normalisation_notes.pdf.')
    parser.add_argument("--r_lims", nargs = 2, type = float, help = 'Specify radius limits of plot (km).')
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

    # Check input arguments.
    if mode_type == 'R':

        assert l == 0, 'Must have l = 0 for radial modes.'

    ## Check input arguments.
    #if use_mineos:

    #    assert i_toroidal is None, 'Do not use --toroidal flag with Mineos, instead specify mode type T (mantle) or I (inner core).'

    #else:

    #    assert mode_type in ['R', 'S', 'T'], 'Mode type must be R, S or T for Ouroboros modes.'

    #    if mode_type in ['R', 'S']:

    #        assert i_toroidal is None, 'The --toroidal flag should not be used for mode types R or S.'

    #    elif mode_type == 'T':

    #        assert i_toroidal is not None, 'Must use the --toroidal flag for mode type T.'

    #    else:

    #        raise ValueError

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
    
    label_suffix, label_suffix_compare = get_label_suffixes(path_compare,
            run_info['code'], code_compare, plot_gradient)

    # Plot.
    if path_compare is not None:

        #if i_toroidal is not None:

        #    if i_toroidal == 0:

        #        mode_type_mineos = 'I'

        #    elif i_toroidal == 1:

        #        mode_type_mineos = 'T'

        #    else:

        #        raise ValueError('Models with more than two solid regions are not supported by Mineos.')

        #else:

        #    mode_type_mineos = mode_type

        ax = plot_eigenfunc_wrapper(run_info_compare, mode_type, n, l, i_toroidal = None, ax = None, show = False, transparent = False, save = False, linestyle = ':', label_suffix = label_suffix_compare, x_label = None, norm_func = norm_func, units = units, plot_gradient = plot_gradient, plot_potential = plot_potential) 
        plot_eigenfunc_wrapper(run_info, mode_type, n, l, i_toroidal = i_toroidal, ax = ax, show = True, label_suffix = label_suffix, plot_gradient = plot_gradient, plot_potential = plot_potential, x_label = x_label, norm_func = norm_func, units = units, alpha = 0.5) 

    else:

        plot_eigenfunc_wrapper(run_info, mode_type, n, l, i_toroidal = i_toroidal, ax = None, plot_gradient = plot_gradient, plot_potential = plot_potential, label_suffix = label_suffix, x_label = x_label, norm_func = norm_func, units = units, r_lims = r_lims) 

    return

if __name__ == '__main__':

    main()
