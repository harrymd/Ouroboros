import argparse

import matplotlib.pyplot as plt
import numpy as np

from Ouroboros.common import (load_eigenfunc, read_input_file)

def plot_eigenfunctions_anelastic_toroidal(run_info, n, l, i_toroidal):

    eigfunc_info = load_eigenfunc(run_info, 'T', n, l, i_toroidal = i_toroidal)

    fig, ax_arr = plt.subplots(1, 2, figsize = (6.0, 8.0), sharey = True,
                        constrained_layout = True)

    font_size_label = 13

    # Real part.
    ax = ax_arr[0]

    ax.plot(eigfunc_info['W_real'], eigfunc_info['r'])

    ax.set_xlabel('Real part', fontsize = font_size_label)
    ax.set_ylabel('Radius (km)', fontsize = font_size_label)

    # Imaginary part.
    ax = ax_arr[1]

    ax.plot(eigfunc_info['W_imag'], eigfunc_info['r'])

    ax.set_xlabel('Imaginary part', fontsize = font_size_label)

    # Tidy axes.
    line_kwargs = {'color' : 'black', 'alpha' : 0.5}
    for ax in ax_arr:

        ax.axvline(**line_kwargs)

    ax.set_ylim(eigfunc_info['r'][0], eigfunc_info['r'][-1])

    plt.show()

    return

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_input_file", help = "File path (relative or absolute) to Ouroboros input file.")
    parser.add_argument("mode_type", choices = ['R', 'S', 'T'], help = 'Mode type (radial, spheroidal or toroidal).') 
    parser.add_argument("n", type = int, help = "Plot mode with radial order n.")
    parser.add_argument("l", type = int, help = "Plot mode with angular order l (must be 0 for radial modes).")
    parser.add_argument("--i_toroidal", dest = "layer_number", help = "Plot toroidal eigenfunction for the solid shell given by LAYER_NUMBER (0 is outermost solid shell).", type = int)
    #
    args = parser.parse_args()

    # Rename input arguments.
    path_input = args.path_to_input_file
    mode_type  = args.mode_type
    n           = args.n
    l           = args.l
    i_toroidal = args.layer_number

    # Read input file.
    run_info = read_input_file(path_input)

    # Plot.
    if mode_type == 'T':

        plot_eigenfunctions_anelastic_toroidal(run_info, n, l, i_toroidal)

    else:

        raise NotImplementedError

    return

if __name__ == '__main__':

    main()
