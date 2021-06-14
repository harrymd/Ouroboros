import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from Ouroboros.common import (  get_Ouroboros_out_dirs, load_eigenfreq,
                                mkdir_if_not_exist, read_input_file)
from Ouroboros.plot.plot_dispersion import plot_dispersion_wrapper

def plot_modes_complex_plane(run_info, mode_type, i_toroidal = None, save = True):

    # Load mode data.
    mode_info = load_eigenfreq(run_info, mode_type, i_toroidal = i_toroidal)
    
    # Unpack.
    l       = mode_info['l']
    omega   = mode_info['omega']
    gamma   = mode_info['gamma']

    # Create canvas.
    fig = plt.figure(figsize = (8.5, 8.5), constrained_layout = True)
    ax  = plt.gca()

    # Plot modes.
    ax.scatter(omega, gamma)
    
    # Label axes.
    font_size_label = 13
    ax.set_xlabel('Angular frequency, $\omega$ (rad s$^{-1}$)',
                    fontsize = font_size_label)
    ax.set_ylabel('Decay rate, $\gamma$ (s$^{-1}$)',
                    fontsize = font_size_label)

    # Tidy axes.
    line_kwargs = {'color' : 'black', 'alpha' : 0.5}
    ax.axhline(**line_kwargs)

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

def plot_modes_dispersion(run_info, mode_type, i_toroidal = None, path_compare = None, save = True):

    # Load mode data.
    mode_info = load_eigenfreq(run_info, mode_type, i_toroidal = i_toroidal)
    
    # Unpack.
    l       = mode_info['l']
    omega   = mode_info['omega']
    gamma   = mode_info['gamma']

    # Convert to mHz.
    f_mHz = (omega / (2.0 * np.pi)) * 1.0E3

    # Create canvas.
    fig = plt.figure(figsize = (8.5, 8.5), constrained_layout = True)
    ax  = plt.gca()

    # Plot modes.
    ax.scatter(l, f_mHz)
    
    # Label axes.
    font_size_label = 13
    ax.set_xlabel('Angular order, $\ell$',
                    fontsize = font_size_label)
    ax.set_ylabel('Frequency (mHz)',
                    fontsize = font_size_label)

    # Tidy axes.
    line_kwargs = {'color' : 'black', 'alpha' : 0.5}
    ax.axhline(**line_kwargs)

    #
    if path_compare is not None:

        run_info_compare = read_input_file(path_compare)
        plot_dispersion_wrapper(run_info_compare, mode_type,
                i_toroidal = i_toroidal, ax = ax,
                save = False, show = False)

    # Save file.
    if save:

        fig_name = 'modes_complex_dispersion.png'
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
    parser.add_argument("--dispersion", action = 'store_true', help = "Include this flag to plot a dispersion diagram (default plots a complex plane diagram).")
    parser.add_argument("--path_compare", help = "Path to Ouroboros input file for comparison.")
    #
    args = parser.parse_args()

    # Rename input arguments.
    path_input = args.path_input    
    i_toroidal = args.layer_number
    dispersion = args.dispersion
    path_compare = args.path_compare

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
    if dispersion:

        plot_modes_dispersion(run_info, mode_type, i_toroidal = i_toroidal,
                path_compare = path_compare)

    else:

        plot_modes_complex_plane(run_info, mode_type, i_toroidal = i_toroidal)

    return

if __name__ == '__main__':

    main()
