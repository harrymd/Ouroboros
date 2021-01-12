import argparse
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from common import load_model, mkdir_if_not_exist, read_Mineos_input_file, read_RadialPNM_input_file
from plot.plot_common import get_r_fluid_solid_boundary, prep_run_info
from post.read_output import get_dir_eigval_RadialPNM, load_eigenfreq_RadialPNM, load_eigenfunc_RadialPNM
from stoneley.code.common.Mineos import load_eigenfreq_Mineos, load_eigenfunc_Mineos

def plot_RadialPNM_old(RadialPNM_info, mode_type, n, l, i_toroidal = None, ax = None, save = True, show = True, transparent = True): 

    k = np.sqrt((l*(l + 1.0)))

    # Get model information for axis limits, scaling and horizontal lines.
    model_data, shape, radius, rho, vp, vs, mu, ka = load_model(RadialPNM_info['path_model'])
    # Convert to km.
    radius = radius*1.0E3
    # r_srf Radius of planet.
    # r_solid_fluid_boundary    List of radii of solid-fluid boundaries.
    r_srf = radius[-1]
    i_fluid, r_solid_fluid_boundary = get_r_fluid_solid_boundary(radius, vs)

    # Get frequency information for title.
    f_RadialPNM = load_eigenfreq_RadialPNM(RadialPNM_info, mode_type, n_q = n, l_q = l, i_toroidal = i_toroidal)
    if mode_type == 'S': 

        r_RadialPNM, U_RadialPNM, V_RadialPNM = load_eigenfunc_RadialPNM(RadialPNM_info, mode_type, n, l)

    elif mode_type == 'T':

        r_RadialPNM, W_RadialPNM = load_eigenfunc_RadialPNM(RadialPNM_info, mode_type, n, l, i_toroidal = i_toroidal)

    # Convert to Mineos normalisation.
    ratio = 1.0E-3*(r_srf**2.0)
    if mode_type == 'S':

        U_RadialPNM = U_RadialPNM*ratio
        V_RadialPNM = k*V_RadialPNM*ratio

    elif mode_type == 'T':

        W_RadialPNM = k*W_RadialPNM*ratio

    #title = '$_{{{:d}}}${:}$_{{{:d}}}$, {:.4f} mHz'.format(n, mode_type, l, f_RadialPNM)
    title = '$_{{{:d}}}${:}$_{{{:d}}}$'.format(n, mode_type, l)

    # Find axis limits.
    if mode_type == 'S':

        vals = np.concatenate([U_RadialPNM, V_RadialPNM])

    else:

        vals = W_RadialPNM

    max_ = np.max(np.abs(vals))


    
    if ax is None:
        
        r_range = np.max(r_RadialPNM) - np.min(r_RadialPNM)
        r_frac = r_range/r_srf
        #fig = plt.figure(figsize = (5.5, 8.5))
        fig = plt.figure(figsize = (5.5, 11.0*r_frac))
        ax  = plt.gca()

    common_args = {'ax' : ax, 'show' : False, 'title' : title}
    
    if mode_type == 'S':

        plot_eigenfunc_S(r_RadialPNM, U_RadialPNM, V_RadialPNM,
                h_lines = r_solid_fluid_boundary, **common_args)

    elif mode_type == 'T':
        
        plot_eigenfunc_T(r_RadialPNM, W_RadialPNM, ax = ax, show = False)

    #ax.set_title(title, fontsize = 20) 
    #ax.set_xlim([-1.1*max_, 1.1*max_])
    
    if transparent:

        set_patch_facecolors(fig, ax) 

    plt.tight_layout()
    
    if save:
        
        dir_out = get_dir_eigval_RadialPNM(RadialPNM_info, mode_type)
        dir_plot_RadialPNM = os.path.join(dir_out, 'plots')
        if mode_type in ['S', 'R']:

            fig_name = 'eigfunc_RadialPNM_{:>05d}_{:}_{:>05d}_{:1d}.png'.format(n, mode_type, l, RadialPNM_info['g_switch'])

        else:

            fig_name = 'eigfunc_RadialPNM_{:>05d}_{:}{:1d}_{:>05d}_{:1d}.png'.format(n, mode_type, i_toroidal, l, RadialPNM_info['g_switch'])

        fig_path = os.path.join(dir_plot_RadialPNM, fig_name)
        print('Saving figure to {:}'.format(fig_path))
        plt.savefig(fig_path, dpi = 300, bbox_inches = 'tight')

    if show:

        plt.show()

    return ax

def plot_eigenfunc_wrapper(run_info, mode_type, n, l, i_toroidal = None, ax = None, save = True, show = True, transparent = True): 

    # Calculate k value.
    k = np.sqrt((l*(l + 1.0)))

    # Get model information for axis limits, scaling and horizontal lines.
    print(run_info['path_model'])
    model_data, shape, radius, rho, vp, vs, mu, ka = load_model(run_info['path_model'])
    # Convert to km.
    radius = radius*1.0E3
    # r_srf Radius of planet.
    # r_solid_fluid_boundary    List of radii of solid-fluid boundaries.
    r_srf = radius[-1]
    i_fluid, r_solid_fluid_boundary, _ = get_r_fluid_solid_boundary(radius, vs)

    # Get frequency information for title.
    if run_info['use_mineos']:

        if i_toroidal is not None:

            raise NotImplementedError('Plotting toroidal mode eigenfunctions not yet implemented for Mineos.')

        f = load_eigenfreq_Mineos(run_info, mode_type, n_q = n, l_q = l)

        if mode_type == 'S': 

            r, U, _, V, _, _, _ = load_eigenfunc_Mineos(run_info, mode_type, n, l)

        elif mode_type == 'T':

            raise NotImplementedError('Plotting toroidal mode eigenfunctions not yet implemented for Mineos.')
            r, W = load_eigenfunc_Mineos(run_info, mode_type, n, l, i_toroidal = i_toroidal)

        r = r*1.0E-3 # Convert to km.

    else:

        f = load_eigenfreq_RadialPNM(run_info, mode_type, n_q = n, l_q = l, i_toroidal = i_toroidal)

        if mode_type == 'S': 

            r, U, V = load_eigenfunc_RadialPNM(run_info, mode_type, n, l)

        elif mode_type == 'T':

            r, W = load_eigenfunc_RadialPNM(run_info, mode_type, n, l, i_toroidal = i_toroidal)

        # Convert to Mineos normalisation.
        ratio = 1.0E-3*(r_srf**2.0)
        if mode_type == 'S':

            U = U*ratio
            V = k*V*ratio

        elif mode_type == 'T':

            W = k*W*ratio

    #title = '$_{{{:d}}}${:}$_{{{:d}}}$, {:.4f} mHz'.format(n, mode_type, l, f_RadialPNM)
    title = '$_{{{:d}}}${:}$_{{{:d}}}$'.format(n, mode_type, l)

    # Find axis limits.
    if mode_type == 'S':

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
    
    if mode_type == 'S':

        plot_eigenfunc_S(r, U, V,
                h_lines = r_solid_fluid_boundary, **common_args)

    elif mode_type == 'T':
        
        plot_eigenfunc_T(r, W, ax = ax, show = False)

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

            fig_name = 'eigfunc_RadialPNM'
            dir_out = get_dir_eigval_RadialPNM(run_info, mode_type)
            dir_plot = os.path.join(dir_out, 'plots')

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

def plot_eigenfunc_T(r, W, ax = None, show = False):

    ax.plot(W, r, label = 'W')

    #ax.axhline(3480.0, color = 'k', linestyle = ':')
    ax.axvline(0.0, color = 'k', linestyle = ':')

    tidy_axes(ax, r)

    return

def main():

    # Read input arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type = int, help = "Plot mode with radial order n.")
    parser.add_argument("l", type = int, help = "Plot mode with angular order l.")
    parser.add_argument("--toroidal", dest = "layer_number", help = "Plot toroidal modes for the solid shell given by LAYER_NUMBER (0 is outermost solid shell).", type = int)
    parser.add_argument("--mineos", action = "store_true", help = "Plot Mineos modes (default: RadialPNM).")
    args = parser.parse_args()

    # Read the input file and command-line arguments.
    #RadialPNM_info, mode_type, n, l, i_toroidal = prep_RadialPNM_info()
    run_info, mode_type, n, l, i_toroidal = prep_run_info(args)

    # Plot.
    plot_eigenfunc_wrapper(run_info, mode_type, n, l, i_toroidal = i_toroidal, ax = None) 

    return

if __name__ == '__main__':

    main()
