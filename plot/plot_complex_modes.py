import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from Ouroboros.common import (  get_Ouroboros_out_dirs, load_eigenfreq,
                                load_eigenfreq_Ouroboros_anelastic,
                                mkdir_if_not_exist, read_input_file)
from Ouroboros.plot.plot_dispersion import plot_dispersion_wrapper

def get_var_lim(mode_info, dataset_key_list, var):
    
    x_min = np.min([np.min(mode_info[dataset_key][var]) for dataset_key in dataset_key_list])
    x_max = np.max([np.max(mode_info[dataset_key][var]) for dataset_key in dataset_key_list])
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

def plot_modes_complex_plane(run_info, mode_type, i_toroidal = None, save = True, include_duplicates = True, dataset_types = ['oscil', 'relax'], label_modes = False):

    # Load mode data.
    #mode_info = load_eigenfreq(run_info, mode_type, i_toroidal = i_toroidal)
    mode_info = load_eigenfreq_Ouroboros_anelastic(run_info, mode_type,
                    i_toroidal = i_toroidal)

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

    f_lim       = get_var_lim(mode_info, dataset_key_list, 'f')
    gamma_lim   = get_var_lim(mode_info, dataset_key_list, 'gamma')

    gamma_scale = 1.0E3

    # Plot modes.
    for dataset_key in dataset_key_list:

        # Unpack.
        l       = mode_info[dataset_key]['l']
        f       = mode_info[dataset_key]['f']
        gamma   = mode_info[dataset_key]['gamma']

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
    ax.set_ylabel('Decay rate, $\gamma$ (2$\pi$ $\\times$ 10$^{6}$ s$^{-1}$)',
                    fontsize = font_size_label)
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

def main():

    # Read input arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("path_input", help = "File path (relative or absolute) to input file.")
    parser.add_argument("--i_toroidal", dest = "layer_number", help = "Plot toroidal modes for the solid shell given by LAYER_NUMBER (0 is outermost solid shell). Default is to plot spheroidal modes.", type = int)
    parser.add_argument("--relax_or_oscil", choices = ['both', 'relax', 'oscil'], default = 'both', help = "Choose between plotting oscillation modes, relaxation modes, or both.")
    parser.add_argument("--show_duplicates", action = 'store_true', help = 'Include this flag to show modes that were identified as duplicates.')
    parser.add_argument("--label_modes", action = 'store_true', help = 'Include this flag to label modes.')
    #
    args = parser.parse_args()

    # Rename input arguments.
    path_input      = args.path_input    
    i_toroidal      = args.layer_number
    relax_or_oscil  = args.relax_or_oscil
    show_duplicates = args.show_duplicates
    label_modes     = args.label_modes

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
    plot_modes_complex_plane(run_info, mode_type, i_toroidal = i_toroidal,
            include_duplicates  = show_duplicates,
            dataset_types       = dataset_types,
            label_modes         = label_modes)

    return

if __name__ == '__main__':

    main()
