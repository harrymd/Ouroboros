import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from Ouroboros.common import (  get_Ouroboros_out_dirs, load_eigenfreq,
                                load_eigenfreq_Ouroboros_anelastic,
                                mkdir_if_not_exist, read_input_file)
from Ouroboros.plot.plot_dispersion import plot_dispersion_wrapper

def get_var_lim(mode_info, dataset_key_list, var, append_vals = None):

    mode_info_merged = []
    for dataset_key in dataset_key_list:
        
        mode_info_merged.extend(mode_info[dataset_key][var])

    if (append_vals is not None) and len(append_vals) > 0:
        
        for append_val_i in append_vals:
            
            append_val_i = np.atleast_1d(append_val_i)
            mode_info_merged.extend(list(append_val_i))
    
    x_min = np.min(mode_info_merged)
    x_max = np.max(mode_info_merged)
    
    #x_min = np.min([np.min(mode_info[dataset_key][var]) for dataset_key in dataset_key_list])
    #x_max = np.max([np.max(mode_info[dataset_key][var]) for dataset_key in dataset_key_list])
    x_range = (x_max - x_min)
    if (x_range == 0.0):

        x_range = np.abs(x_min)
    
    buff = (0.1 * x_range)
    if x_min > 0.0:

        x_min = 0.0

    elif x_max < 0.0:

        x_max = 0.0
    
    x_min = (x_min - buff)
    x_max = (x_max + buff)

    x_lims = np.array([x_min, x_max])

    return x_lims

def load_roots_poles_Ouroboros_anelastic(run_info, mode_type, i_toroidal = None):

    if mode_type == 'T':

        assert i_toroidal is not None, 'For toroidal modes, the optional argument \'i toroidal\' must specify the layer number.'
        
        name_poles = 'poles_{:>03d}.txt'.format(i_toroidal)
        name_roots = 'roots_{:>03d}.txt'.format(i_toroidal)

    else:

        name_poles = 'poles.txt'
        name_roots = 'roots.txt'

    # Get output paths.
    dir_model, dir_run, dir_g, dir_type = get_Ouroboros_out_dirs(run_info, mode_type)

    # Get path to files.
    path_poles = os.path.join(dir_type, name_poles)
    path_roots = os.path.join(dir_type, name_roots)

    # Load.
    poles = np.loadtxt(path_poles, dtype = np.complex_)
    roots = np.loadtxt(path_roots, dtype = np.complex_)

    return poles, roots

def plot_YP1982_data(ax, gamma_scale):

    path_in = "/Users/hrmd_work/Documents/research/refs/yuen_1982/yuen_1982_fig_03a.txt"
    data = np.loadtxt(path_in)
    s_r_per_day, s_i_per_min = data.T

    s_r_mHz = (1000.0 / (2.0 * np.pi)) * s_r_per_day / (60.0 * 60.0 * 24.0)
    s_i_mHz = (1000.0 / (2.0 * np.pi)) * s_i_per_min / 60.0

    s = s_r_mHz + (1.0j * s_i_mHz)
    nu = -1.0j * s

    om = np.real(nu)
    ga = np.imag(nu)
    
    print("YP1982")
    print(om)
    print(ga)


    ax.scatter(om, ga * gamma_scale, label = 'Yuen and Peltier (1982)')

    return

def plot_modes_complex_plane(run_info, mode_type, i_toroidal = None, save = True, include_duplicates = True, dataset_types = ['oscil', 'relax'], label_modes = False, equal_aspect = True, show_roots = False, show_poles = False):

    # Load mode data.
    mode_info = load_eigenfreq_Ouroboros_anelastic(run_info, mode_type,
                    i_toroidal = i_toroidal)

    # Load roots and/or poles, if requested.
    if (show_roots or show_poles):

        poles, roots = load_roots_poles_Ouroboros_anelastic(run_info,
                            mode_type, i_toroidal = i_toroidal)
    
        ## Convert from Laplace variable to angular frequency.
        if show_poles:
            
            #poles = -1.0j * poles

            print('roots:')
            print(roots)

        if show_roots:

            #roots = -1.0j * roots

            print('poles:')
            print(poles)

    # Create canvas.
    fig = plt.figure(figsize = (8.5, 8.5), constrained_layout = True)
    ax  = plt.gca()

    color_dict = {  'oscil' : 'blue', 'oscil_duplicate' : 'purple',
                    'relax' : 'red',  'relax_duplicate' : 'orange'}

    label_dict = {  'oscil' : 'Oscillation',
                    'relax' : 'Relaxation',
                    'oscil_duplicate' : 'Oscillation (duplicate)',
                    'relax_duplicate' : 'Relaxation (duplicate)'}

    scatter_kwargs = {'alpha' : 0.5}
    
    if include_duplicates:

        dataset_key_list = []
        for dataset_key in dataset_types:

            dataset_key_list.append(dataset_key)
            dataset_key_list.append('{:}_duplicate'.format(dataset_key))

    else:
        
        dataset_key_list = dataset_types
    
    # Find axis limits.
    append_vals_real = []
    append_vals_imag = []
    if show_roots:

        append_vals_real.append(np.real(roots))
        append_vals_imag.append(np.imag(roots))

    if show_poles:

        append_vals_real.append(np.real(poles))
        append_vals_imag.append(np.imag(poles))
     
    f_lim       = get_var_lim(mode_info, dataset_key_list, 'f',
                        append_vals = append_vals_real)
    gamma_lim   = get_var_lim(mode_info, dataset_key_list, 'gamma',
                        append_vals = append_vals_imag)
    
    if equal_aspect:

        gamma_scale = 1.0
        #y_label = 'Decay rate, $\gamma$ (2$\pi$ $\\times$ 10$^{3}$ s$^{-1}$)'
        y_label = 'Decay rate, $\gamma$ (2$\pi$ $\\times$ 10$^{-3}$ s$^{-1}$)'

    else:

        gamma_scale = 1.0E3
        #y_label = 'Decay rate, $\gamma$ (2$\pi$ $\\times$ 10$^{6}$ s$^{-1}$)'
        y_label = 'Decay rate, $\gamma$ (2$\pi$ $\\times$ 10$^{-6}$ s$^{-1}$)'


    if equal_aspect:

        if np.abs(f_lim[1]) > np.abs(gamma_lim[1] * gamma_scale):

            gamma_lim = f_lim / gamma_scale

        else:
            
            f_lim = (gamma_lim * gamma_scale)
    
    # Plot modes.
    for dataset_key in dataset_key_list:

        # Unpack.
        l       = mode_info[dataset_key]['l']
        f       = mode_info[dataset_key]['f']
        gamma   = mode_info[dataset_key]['gamma']
    
        print('\n')
        print(dataset_key)
        print(f, gamma)

        ax.scatter(f, gamma * gamma_scale,
                c = color_dict[dataset_key],
                label = label_dict[dataset_key],
                **scatter_kwargs)

        if label_modes:

            num_modes = len(l)
            for i in range(num_modes):

                if 'n' in mode_info[dataset_key].keys():

                    label = '{:>d},{:>d}'.format(
                            mode_info[dataset_key]['n'][i], l[i])

                else:

                    label = '{:>d}'.format(i)

                ax.annotate(label, (f[i], gamma[i] * gamma_scale),
                        xytext = (5, 5), xycoords = 'data',
                        textcoords = 'offset points')

    # Plot roots and/or poles.
    if show_roots:
        
        ax.scatter(np.real(roots), np.imag(roots) * gamma_scale, marker = 'x', 
                    label = 'Roots', **scatter_kwargs)

    if show_poles:

        ax.scatter(np.real(poles), np.imag(poles) * gamma_scale, marker = 'x', 
                    label = 'Poles', **scatter_kwargs)
    
    show_YP1982 = True
    if show_YP1982:

        plot_YP1982_data(ax, gamma_scale)

    # Set axis limits.
    ax.set_xlim(f_lim)
    ax.set_ylim(gamma_lim * gamma_scale)
    
    # Label axes.
    font_size_label = 16
    #ax.set_xlabel('Angular frequency, $\omega$ (rad s$^{-1}$)',
    #                fontsize = font_size_label)
    #ax.set_ylabel('Decay rate, $\gamma$ (s$^{-1}$)',
    #                fontsize = font_size_label)
    ax.set_xlabel('Frequency, $f$ (mHz)', fontsize = font_size_label)
    ax.set_ylabel(y_label, fontsize = font_size_label)
    ax.legend(loc= 'best')

    # Set the x-spine.
    ax.spines['left'].set_position('zero')
    
    # Turn off the right spine/ticks.
    ax.spines['right'].set_color('none')
    ax.yaxis.tick_left()
    
    # Set the y-spine.
    ax.spines['bottom'].set_position('zero')
    
    # Turn off the top spine/ticks.
    ax.spines['top'].set_color('none')
    ax.xaxis.tick_bottom()
    
    if equal_aspect:

        ax.set_aspect(1.0 / gamma_scale)

    ## Tidy axes.
    #line_kwargs = {'color' : 'black', 'alpha' : 0.5}
    #ax.axhline(**line_kwargs)




    # Save file.
    if save:

        fig_name = 'modes_complex_plane.png'
        _, _, _, dir_type = get_Ouroboros_out_dirs(run_info, mode_type)
        dir_out = dir_type

        dir_plot = os.path.join(dir_out, 'plots')
        mkdir_if_not_exist(dir_plot)
        fig_path = os.path.join(dir_plot, fig_name)
        print('Saving figure to {:}'.format(fig_path))
        plt.savefig(fig_path, dpi = 300, bbox_inches = 'tight')

    plt.show()

    return

def plot_f_versus_Q(run_info, mode_type, i_toroidal = None):


    # Load mode data.
    mode_info = load_eigenfreq_Ouroboros_anelastic(run_info, mode_type,
                    i_toroidal = i_toroidal)
    mode_info = mode_info['oscil']

    font_size_label = 13

    fig = plt.figure(figsize = (8.5, 8.5), constrained_layout = True)
    ax  = plt.gca()

    ax.scatter(mode_info['f'], mode_info['Q'])

    ax.set_xlabel('Frequency (mHz)', fontsize = font_size_label)
    ax.set_ylabel('$Q$ (s$^{-1}$)', fontsize = font_size_label)

    plt.show()

    return

def main():

    # Read input arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("path_input", help = "File path (relative or absolute) to input file.")
    parser.add_argument("--i_toroidal", dest = "layer_number", help = "Plot toroidal modes for the solid shell given by LAYER_NUMBER (0 is outermost solid shell). Default is to plot spheroidal modes.", type = int)
    parser.add_argument("--relax_or_oscil", choices = ['both', 'relax', 'oscil'], default = 'both', help = "Choose between plotting oscillation modes, relaxation modes, or both.")
    parser.add_argument("--fQ", action = 'store_true', help = "Include this flag to plot frequency (f) versus quality factor (Q).")
    parser.add_argument("--show_duplicates", action = 'store_true', help = 'Include this flag to show modes that were identified as duplicates.')
    parser.add_argument("--label_modes", action = 'store_true', help = 'Include this flag to label modes.')
    parser.add_argument("--equal_aspect", action = 'store_true', help = 'Include this flag to plot equal aspect ratios for real and imaginary axes.')
    parser.add_argument("--poles", action = 'store_true', help = 'Include this flag to plot the poles of the anelastic model.')
    parser.add_argument("--roots", action = 'store_true', help = 'Include this flag to plot the roots of the anelastic model.')
    #
    args = parser.parse_args()

    # Rename input arguments.
    path_input      = args.path_input    
    i_toroidal      = args.layer_number
    relax_or_oscil  = args.relax_or_oscil
    plot_f_Q        = args.fQ
    show_duplicates = args.show_duplicates
    label_modes     = args.label_modes
    equal_aspect    = args.equal_aspect
    roots           = args.roots
    poles           = args.poles

    if relax_or_oscil == 'both':

        dataset_types = ['relax', 'oscil']

    else:

        dataset_types = [relax_or_oscil]

    # Read input file(s).
    run_info = read_input_file(path_input)

    # Set mode type string.
    if i_toroidal is not None:

        mode_type = 'T'
    
    elif run_info['mode_types'] == ['R']:
        
        raise NotImplementedError

    else:

        mode_type = 'S'

    # Plot.
    if plot_f_Q:

        plot_f_versus_Q(run_info, mode_type, i_toroidal = i_toroidal) 
    
    else:

        plot_modes_complex_plane(run_info, mode_type, i_toroidal = i_toroidal,
            include_duplicates  = show_duplicates,
            dataset_types       = dataset_types,
            label_modes         = label_modes,
            equal_aspect        = equal_aspect,
            show_poles          = poles,
            show_roots          = roots)

    return

if __name__ == '__main__':

    main()
