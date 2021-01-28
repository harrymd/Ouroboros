import argparse
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from common import get_Ouroboros_out_dirs, get_r_fluid_solid_boundary, load_eigenfreq_Ouroboros, load_eigenfunc_Ouroboros, load_model, mkdir_if_not_exist, read_Mineos_input_file, read_Ouroboros_input_file
from stoneley.code.common.Mineos import load_eigenfreq_Mineos, load_eigenfunc_Mineos

def plot_eigenfunc_wrapper(run_info, mode_type, n, l, i_toroidal = None, ax = None, save = True, show = True, transparent = True): 

    # Calculate k value.
    k = np.sqrt((l*(l + 1.0)))

    # Get model information for axis limits, scaling and horizontal lines.
    print(run_info['path_model'])
    model = load_model(run_info['path_model'])
    # Convert to km.
    model['r'] = model['r']*1.0E-3
    # r_srf Radius of planet.
    # r_solid_fluid_boundary    List of radii of solid-fluid boundaries.
    r_srf = model['r'][-1]
    i_fluid, r_solid_fluid_boundary, _ = get_r_fluid_solid_boundary(model['r'], model['v_s'])

    # Get frequency information for title.
    if run_info['use_mineos']:

        if i_toroidal is not None:

            raise NotImplementedError('Plotting toroidal mode eigenfunctions not yet implemented for Mineos.')

        f = load_eigenfreq_Mineos(run_info, mode_type, n_q = n, l_q = l)

        if mode_type == 'R':

            raise NotImplementedError

        elif mode_type == 'S': 

            r, U, _, V, _, _, _ = load_eigenfunc_Mineos(run_info, mode_type, n, l)

        elif mode_type == 'T':

            raise NotImplementedError('Plotting toroidal mode eigenfunctions not yet implemented for Mineos.')
            r, W = load_eigenfunc_Mineos(run_info, mode_type, n, l, i_toroidal = i_toroidal)

        r = r*1.0E-3 # Convert to km.

    else:

        f = load_eigenfreq_Ouroboros(run_info, mode_type, n_q = n, l_q = l, i_toroidal = i_toroidal)

        if mode_type == 'R':

            r, U = load_eigenfunc_Ouroboros(run_info, mode_type, n, l)
            U[0] = 0.0 # Value of U at planet core appears to be buggy for R modes.

        elif mode_type == 'S': 

            r, U, V = load_eigenfunc_Ouroboros(run_info, mode_type, n, l)

        elif mode_type == 'T':

            r, W = load_eigenfunc_Ouroboros(run_info, mode_type, n, l, i_toroidal = i_toroidal)

        # Convert to Mineos normalisation.
        ratio = 1.0E-3*(r_srf**2.0)
        if mode_type == 'R':
            
            U = U*ratio

        elif mode_type == 'S':

            U = U*ratio
            V = k*V*ratio

        elif mode_type == 'T':

            W = k*W*ratio

    #print(r.shape)
    #print(U.shape)
    #print(r[0:10])
    #print(U[0:10])
    #import sys
    #sys.exit()

    #title = '$_{{{:d}}}${:}$_{{{:d}}}$, {:.4f} mHz'.format(n, mode_type, l, f_Ouroboros)
    if mode_type in ['R', 'S']:

        mode_type_for_title = 'S'

    else:

        if i_toroidal == 0:
            
            mode_type_for_title = 'I'
        
        else:
            
            mode_type_for_title = 'T'

    title = '$_{{{:d}}}${:}$_{{{:d}}}$'.format(n, mode_type_for_title, l)

    # Find axis limits.
    if mode_type == 'R':

        vals = U

    elif mode_type == 'S':

        vals = np.concatenate([U, V])

    else:

        vals = W

    max_ = np.max(np.abs(vals))

    if ax is None:
        
        r_range = np.max(r) - np.min(r)
        r_frac = r_range/r_srf
        #fig = plt.figure(figsize = (5.5, 8.5))
        fig = plt.figure(figsize = (5.5, 11.0*r_frac))
        ax  = plt.gca()

    common_args = {'ax' : ax, 'show' : False, 'title' : title}

    if mode_type == 'R':

        plot_eigenfunc_R_or_T(r, U, h_lines = r_solid_fluid_boundary, label = 'U', **common_args)
    
    elif mode_type == 'S':

        plot_eigenfunc_S(r, U, V,
                h_lines = r_solid_fluid_boundary, **common_args)

    elif mode_type == 'T':
        
        plot_eigenfunc_R_or_T(r, W, h_lines = None, label = 'W', **common_args)

    #ax.set_title(title, fontsize = 20) 
    #ax.set_xlim([-1.1*max_, 1.1*max_])
    
    if transparent:

        set_patch_facecolors(fig, ax) 

    plt.tight_layout()
    
    if save:
        
        if run_info['use_mineos']:

            fig_name = 'eigfunc_Mineos'
            dir_out = run_info['dir_output']
            dir_plot = os.path.join(dir_out, 'plots')

        else:

            fig_name = 'eigfunc_Ouroboros'
            _, _, _, dir_out = get_Ouroboros_out_dirs(run_info, mode_type)
            dir_plot = os.path.join(dir_out, 'plots')

        mkdir_if_not_exist(dir_plot)

        if mode_type in ['S', 'R']:

            fig_name = '{:}_{:>05d}_{:}_{:>05d}_{:1d}.png'.format(fig_name, n, mode_type, l, run_info['g_switch'])

        else:

            fig_name = '{:}_{:>05d}_{:}{:1d}_{:>05d}_{:1d}.png'.format(fig_name, n, mode_type, i_toroidal, l, run_info['g_switch'])

        fig_path = os.path.join(dir_plot, fig_name)
        print('Saving figure to {:}'.format(fig_path))
        plt.savefig(fig_path, dpi = 300, bbox_inches = 'tight')

    if show:

        plt.show()

    return ax

def plot_eigenfunc_S(r, U, V, k = None, ax = None, h_lines = None, x_label = 'Eigenfunction', y_label = 'Radial coordinate / km', title = None, show = True, add_legend = True, colors = ['r', 'b'], linestyles = ['-', '-'], label_append = '', alpha = 1.0, legend_loc = 'best', font_size_label = 12):
    
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

    U_label = U_label + label_append
    V_label = V_label + label_append

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

def plot_eigenfunc_R_or_T(r, U_or_W, ax = None, show = False, h_lines = None, add_legend = True, legend_loc = 'best', title = None, label = None, x_label = 'Eigenfunction', y_label = 'Radial coordinate / km'):

    ax.plot(U_or_W, r, label = label)

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
    parser.add_argument("mode_type", choices = ['R', 'S', 'T'], help = 'Mode type (radial, spheroidal or toroidal).')
    parser.add_argument("n", type = int, help = "Plot mode with radial order n.")
    parser.add_argument("l", type = int, help = "Plot mode with angular order l (must be 0 for radial modes).")
    parser.add_argument("--toroidal", dest = "layer_number", help = "Plot toroidal modes for the solid shell given by LAYER_NUMBER (0 is outermost solid shell).", type = int)
    parser.add_argument("--mineos", action = "store_true", help = "Plot Mineos modes (default: Ouroboros).")
    args = parser.parse_args()

    # Rename input arguments.
    path_input = args.path_to_input_file
    mode_type  = args.mode_type
    n           = args.n
    l           = args.l
    i_toroidal = args.layer_number
    use_mineos = args.mineos

    if mode_type == 'R':

        assert l == 0, 'Must have l = 0 for radial modes.'

    # Check input arguments.
    if mode_type in ['R', 'S']:

        assert i_toroidal is None, 'The --toroidal flag should not be used for mode types R or S.'

    else:

        assert i_toroidal is not None, 'Must use the --toroidal flag for mode type T.'

    # Read the input file and command-line arguments.
    run_info = read_Ouroboros_input_file(path_input)
    run_info['use_mineos'] = use_mineos
    #Ouroboros_info, mode_type, n, l, i_toroidal = prep_Ouroboros_info()
    #run_info, mode_type, n, l, i_toroidal = prep_run_info(args)

    # Plot.
    plot_eigenfunc_wrapper(run_info, mode_type, n, l, i_toroidal = i_toroidal, ax = None) 

    return

if __name__ == '__main__':

    main()
