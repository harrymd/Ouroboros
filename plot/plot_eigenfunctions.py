import argparse
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from common import get_Ouroboros_out_dirs, get_r_fluid_solid_boundary, load_eigenfreq_Mineos, load_eigenfreq_Ouroboros, load_eigenfunc_Mineos, load_eigenfunc_Ouroboros, load_model, mkdir_if_not_exist, read_Mineos_input_file, read_Ouroboros_input_file
#from stoneley.code.common.Mineos import load_eigenfreq_Mineos, load_eigenfunc_Mineos

def plot_eigenfunc_wrapper(run_info, mode_type, n, l, i_toroidal = None, ax = None, save = True, show = True, transparent = True, linestyle = '-', label_suffix = '', sign = None, plot_gradient = False, x_label = 'Eigenfunction', norm_func = 'mineos', units = 'SI', alpha = 1.0): 

    # Get model information for axis limits, scaling and horizontal lines.
    model = load_model(run_info['path_model'])
    # Convert to km.
    model['r'] = model['r']*1.0E-3
    # r_srf Radius of planet.
    # r_solid_fluid_boundary    List of radii of solid-fluid boundaries.
    r_srf = model['r'][-1]
    i_fluid, r_solid_fluid_boundary, _ = get_r_fluid_solid_boundary(model['r'], model['v_s'])

    # Get frequency information for title.
    if run_info['use_mineos']:

        f = load_eigenfreq_Mineos(run_info, mode_type, n_q = n, l_q = l)

        # Get normalisation arguments.
        f_rad_per_s = f*1.0E-3*2.0*np.pi
        normalisation_args = {'norm_func' : norm_func, 'units' : units}
        if norm_func == 'DT':
            normalisation_args['omega'] = f_rad_per_s

        if mode_type == 'R':

            r, U, Up = load_eigenfunc_Mineos(run_info, mode_type, n, l, **normalisation_args)

        elif mode_type == 'S': 

            r, U, Up, V, Vp, P, Pp = load_eigenfunc_Mineos(run_info, mode_type, n, l, **normalisation_args)

        elif mode_type in ['T', 'I']:

            r, W, Wp = load_eigenfunc_Mineos(run_info, mode_type, n, l, **normalisation_args)

        else:

            raise ValueError

        r = r*1.0E-3 # Convert to km.

    else:

        f = load_eigenfreq_Ouroboros(run_info, mode_type, n_q = n, l_q = l, i_toroidal = i_toroidal)

        # Get normalisation arguments.
        f_rad_per_s = f*1.0E-3*2.0*np.pi
        normalisation_args = {'norm_func' : norm_func, 'units' : units}
        if norm_func == 'DT':
            normalisation_args['omega'] = f_rad_per_s

        if mode_type == 'R':

            r, U = load_eigenfunc_Ouroboros(run_info, mode_type, n, l,
                        **normalisation_args)

            Up = np.zeros(Up) # Gradient not implemented yet.

        elif mode_type == 'S': 

            r, U, V = load_eigenfunc_Ouroboros(run_info, mode_type, n, l,
                            **normalisation_args)

            # Gradient not implemented yet.
            Up = np.zeros(U.shape)
            Vp = np.zeros(V.shape)

        elif mode_type == 'T':

            r, W = load_eigenfunc_Ouroboros(run_info, mode_type, n, l, i_toroidal = i_toroidal,
                        **normalisation_args)

            # Gradient not implemented yet.
            Wp = np.zeros(W.shape)

        else:

            raise ValueError
        
    ## Apply factor of k (used for both Mineos and Ouroboros).
    ## Calculate k value.
    #k = np.sqrt((l*(l + 1.0)))
    #if mode_type == 'S':

    #    V = k*V
    #    Vp = k*Vp

    #elif mode_type in ['T', 'I']:

    #    W = k*W
    #    Wp = k*Wp
    
    # Sign.
    if mode_type == 'R':

        sign_max_eigfunc = np.sign(U[np.argmax(np.abs(U))]) 

    elif mode_type == 'S':
        
        max_abs_U = np.max(np.abs(U))
        max_abs_V = np.max(np.abs(V))
        if max_abs_U > max_abs_V:

            sign_max_eigfunc = np.sign(U[np.argmax(np.abs(U))]) 

        else:

            sign_max_eigfunc = np.sign(V[np.argmax(np.abs(V))]) 

    elif mode_type in ['T', 'I']:

        sign_max_eigfunc = np.sign(W[np.argmax(np.abs(W))]) 

    else:

        raise ValueError

    if sign is not None:

        if sign_max_eigfunc != sign:

            if mode_type == 'R':

                U = U*-1.0
                Up = Up*-1.0

            elif mode_type == 'S':

                U = U*-1.0
                Up = Up*-1.0
                V = V*-1.0
                Vp = Vp*-1.0

            elif mode_type in ['T', 'I']:

                W = W*-1.0
                Wp = Wp*-1.0

            else:

                raise NotImplementedError

    if mode_type in ['R', 'S']:

        mode_type_for_title = 'S'

    else:
        
        if run_info['use_mineos']:

            mode_type_for_title = mode_type

        else:

            if i_toroidal == 0:
                
                mode_type_for_title = 'I'
            
            else:
                
                mode_type_for_title = 'T'

    title = '$_{{{:d}}}${:}$_{{{:d}}}$'.format(n, mode_type_for_title, l)
    
    if plot_gradient:

        # Find axis limits.
        if mode_type == 'R':

            vals = Up

        elif mode_type == 'S':

            vals = np.concatenate([Up, Vp])

        else:

            vals = Wp

    else:

        # Find axis limits.
        if mode_type == 'R':

            vals = U

        elif mode_type == 'S':

            vals = np.concatenate([U, V])

        else:

            vals = W

    max_ = np.max(np.abs(vals))

    if run_info['use_mineos']:
        
        clip_zero = True
        if mode_type in ['T', 'I']:

            if clip_zero:

                i_nonzero = np.nonzero(W)[0]

            i_0 = i_nonzero[0]
            i_1 = i_nonzero[-1]

            r = r[i_0 : i_1]
            W = W[i_0 : i_1]
            Wp = Wp[i_0 : i_1]

    if ax is None:
        
        r_range = np.max(r) - np.min(r)
        r_frac = r_range/r_srf
        #fig = plt.figure(figsize = (5.5, 8.5))
        fig = plt.figure(figsize = (5.5, 11.0*r_frac))
        ax  = plt.gca()

    common_args = {'ax' : ax, 'show' : False, 'title' : title,
            'x_label' : x_label, 'alpha' : alpha}

    if plot_gradient:

        if not run_info['use_mineos']:

            raise NotImplementedError('Gradient not implemented yet for Ouroboros.')

        if mode_type == 'R':

            plot_eigenfunc_R_or_T(r, Up, h_lines = r_solid_fluid_boundary, linestyle = linestyle, label = 'U{:}'.format(label_suffix), **common_args)
        
        elif mode_type == 'S':

            plot_eigenfunc_S(r, Up, Vp,
                    h_lines = r_solid_fluid_boundary, linestyles = [linestyle, linestyle], label_suffix = label_suffix, **common_args)

        elif mode_type in ['T', 'I']:
            
            plot_eigenfunc_R_or_T(r, Wp, h_lines = None, linestyle = linestyle, label = 'W{:}'.format(label_suffix), **common_args)

    else:

        if mode_type == 'R':

            plot_eigenfunc_R_or_T(r, U, h_lines = r_solid_fluid_boundary, linestyle = linestyle, label = 'U{:}'.format(label_suffix), **common_args)
        
        elif mode_type == 'S':

            plot_eigenfunc_S(r, U, V,
                    h_lines = r_solid_fluid_boundary, linestyles = [linestyle, linestyle], label_suffix = label_suffix, **common_args)

        elif mode_type in ['T', 'I']:
            
            plot_eigenfunc_R_or_T(r, W, h_lines = None, linestyle = linestyle, label = 'W{:}'.format(label_suffix), **common_args)

    if transparent:
        
        fig = plt.gcf()
        set_patch_facecolors(fig, ax) 

    plt.tight_layout()
    
    if save:
        
        if run_info['use_mineos']:
            
            if plot_gradient:

                fig_name = 'eigfunc_gradient_Mineos'

            else:

                fig_name = 'eigfunc_Mineos'

            dir_out = run_info['dir_output']
            dir_plot = os.path.join(dir_out, 'plots')

        else:
            
            if plot_gradient:

                fig_name = 'eigfunc_gradient_Ouroboros'

            else:

                fig_name = 'eigfunc_Ouroboros'
            _, _, _, dir_out = get_Ouroboros_out_dirs(run_info, mode_type)
            dir_plot = os.path.join(dir_out, 'plots')

        mkdir_if_not_exist(dir_plot)

        if mode_type in ['S', 'R']:

            fig_name = '{:}_{:>05d}_{:}_{:>05d}_{:1d}.png'.format(fig_name, n, mode_type, l, run_info['grav_switch'])

        else:

            if run_info['use_mineos']:

                fig_name = '{:}_{:>05d}_{:}_{:>05d}_{:1d}.png'.format(fig_name, n, mode_type, l, run_info['grav_switch'])

            else:

                fig_name = '{:}_{:>05d}_{:}{:1d}_{:>05d}_{:1d}.png'.format(fig_name, n, mode_type, i_toroidal, l, run_info['grav_switch'])

        fig_path = os.path.join(dir_plot, fig_name)
        print('Saving figure to {:}'.format(fig_path))
        plt.savefig(fig_path, dpi = 300, bbox_inches = 'tight')

    if show:

        plt.show()

    return ax, sign_max_eigfunc

def plot_eigenfunc_S(r, U, V, k = None, ax = None, h_lines = None, x_label = 'Eigenfunction', y_label = 'Radial coordinate / km', title = None, show = True, add_legend = True, colors = ['r', 'b'], linestyles = ['-', '-'], label_suffix = '', alpha = 1.0, legend_loc = 'best', font_size_label = 12):
    
    if ax is None:

        fig = plt.figure()
        ax  = plt.gca()
       
    if k is None:

        U_label = 'U'
        V_label = 'V'
        k = 1.0

    else:

        U_label = 'U'
        V_label = 'kV'

    U_label = U_label + label_suffix
    V_label = V_label + label_suffix

    ax.plot(U, r, label = U_label, color = colors[0], linestyle = linestyles[0], alpha = alpha)
    ax.plot(k*V, r, label = V_label, color = colors[1], linestyle = linestyles[1], alpha = alpha)

    max_abs_U_plot = np.max(np.abs(U))
    max_abs_V_plot = np.max(np.abs(k*V))
    E_max = np.max([max_abs_U_plot, max_abs_V_plot])
    
    tidy_axes(ax, r, E_max, h_lines = h_lines, add_legend = add_legend, legend_loc = legend_loc, title = title, x_label = x_label, y_label = y_label)

    if show:

        plt.show()

    return ax, E_max

def tidy_axes(ax, r, E_max, h_lines = None, add_legend = True, legend_loc = 'best', title = None, x_label = 'Eigenfunction', y_label = 'Radius / km'):
    
    font_size_label = 16
    font_size_title = 36 

    if h_lines is not None:

        for h_line in h_lines:

            ax.axhline(h_line, linestyle = ':', color = 'k')

    ax.axvline(linestyle = ':', color = 'k')
    
    buff = 1.05
    ax.set_xlim([-buff*E_max, buff*E_max])
    ax.set_ylim([np.min(r), np.max(r)])
    
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

def main():

    # Read input arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_input_file", help = "File path (relative or absolute) to Ouroboros input file.")
    parser.add_argument("mode_type", choices = ['R', 'S', 'T', 'I'], help = 'Mode type (radial, spheroidal or toroidal). Option I is for use with --use_mineos flag to plot inner-core toroidal modes. See the --toroidal flag for plotting toroidal modes with Ouroboros.')
    parser.add_argument("n", type = int, help = "Plot mode with radial order n.")
    parser.add_argument("l", type = int, help = "Plot mode with angular order l (must be 0 for radial modes).")
    parser.add_argument("--toroidal", dest = "layer_number", help = "Plot toroidal eigenfunction for the solid shell given by LAYER_NUMBER (0 is outermost solid shell).", type = int)
    parser.add_argument("--gradient", action = "store_true", help = "Include this flag to plot eigenfunction gradients (default: plot eigenfunctions).")
    parser.add_argument("--use_mineos", action = "store_true", help = "Plot Mineos eigenfunction (default: Ouroboros).")
    parser.add_argument("--path_input_mineos_compare", help = "Provide Mineos input path to plot both Ouroboros and Mineos eigenfunction (default: Ouroboros only).")
    parser.add_argument("--norm_func", choices = ['mineos', 'DT'], default = 'DT', help = "Specify normalisation function. \'mineos\' is the normalisation function used by Mineos and Ouroboros. \'DT\' is the normalisation function used in the Dahlen and Tromp textbook. It does not include the factor of k. See also the --units flag. For more detail, see Ouroboros/doc/Ouroboros_normalisation_notes.pdf.")
    parser.add_argument("--units", choices = ['SI', 'ouroboros', 'mineos'], default = 'mineos', help = 'Specify units used when applying normalisation to eigenfunction. \'SI\' is SI units. \'mineos\' is Mineos units. \'ouroboros\' is Ouroboros units. See also the --norm_func flag. For more detail, see Ouroboros/doc/Ouroboros_normalisation_notes.pdf.')
    args = parser.parse_args()

    # Rename input arguments.
    path_input = args.path_to_input_file
    mode_type  = args.mode_type
    n           = args.n
    l           = args.l
    i_toroidal = args.layer_number
    plot_gradient = args.gradient
    use_mineos = args.use_mineos
    path_input_mineos_compare = args.path_input_mineos_compare
    norm_func = args.norm_func
    units = args.units
    assert not (use_mineos and (not(path_input_mineos_compare is None))), 'Only one of --use_mineos and --path_input_mineos_compare may be used.'

    if mode_type == 'R':

        assert l == 0, 'Must have l = 0 for radial modes.'

    # Check input arguments.
    if use_mineos:

        assert i_toroidal is None, 'Do not use --toroidal flag with Mineos, instead specify mode type T (mantle) or I (inner core).'

    else:

        assert mode_type in ['R', 'S', 'T'], 'Mode type must be R, S or T for Ouroboros modes.'

        if mode_type in ['R', 'S']:

            assert i_toroidal is None, 'The --toroidal flag should not be used for mode types R or S.'

        elif mode_type == 'T':

            assert i_toroidal is not None, 'Must use the --toroidal flag for mode type T.'

        else:

            raise ValueError

    # Read the input file and command-line arguments.
    if use_mineos:

        run_info = read_Mineos_input_file(path_input)
        run_info['use_mineos'] = True

    else:

        run_info = read_Ouroboros_input_file(path_input)
        run_info['use_mineos'] = False

        if path_input_mineos_compare is not None:
            
            run_info_mineos = read_Mineos_input_file(path_input_mineos_compare)
            run_info_mineos['use_mineos'] = True

    #Ouroboros_info, mode_type, n, l, i_toroidal = prep_Ouroboros_info()
    #run_info, mode_type, n, l, i_toroidal = prep_run_info(args)

    if plot_gradient:

        x_label = 'Eigenfunction gradient'

    else:

        x_label = 'Eigenfunction'

    # Plot.
    if path_input_mineos_compare is not None:

        if i_toroidal is not None:

            if i_toroidal == 0:

                mode_type_mineos = 'I'

            elif i_toroidal == 1:

                mode_type_mineos = 'T'

            else:

                raise ValueError('Models with more than two solid regions are not supported by Mineos.')

        else:

            mode_type_mineos = mode_type

        if plot_gradient:

            label_suffix_Mineos = '\' (Mineos)'
            label_suffix_Ouroboros = '\' (Ouroboros)'

        else:

            label_suffix_Mineos = ' (Mineos)'
            label_suffix_Ouroboros = ' (Ouroboros)'

        ax, sign = plot_eigenfunc_wrapper(run_info_mineos, mode_type_mineos, n, l, i_toroidal = None, ax = None, show = False, transparent = False, save = False, linestyle = ':', label_suffix = label_suffix_Mineos, x_label = None, norm_func = norm_func, units = units) 
        plot_eigenfunc_wrapper(run_info, mode_type, n, l, i_toroidal = i_toroidal, ax = ax, show = True, label_suffix = label_suffix_Ouroboros, sign = sign, plot_gradient = plot_gradient, x_label = x_label, norm_func = norm_func, units = units, alpha = 0.5) 

    else:

        if plot_gradient:

            label_suffix = '\''

        else:

            label_suffix = ''

        plot_eigenfunc_wrapper(run_info, mode_type, n, l, i_toroidal = i_toroidal, ax = None, plot_gradient = plot_gradient, label_suffix = label_suffix, x_label = x_label, norm_func = norm_func, units = units) 

    return

if __name__ == '__main__':

    main()
