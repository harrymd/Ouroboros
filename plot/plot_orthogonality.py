'''
Plot scalar products of different mode pairs to demonstrate orthogonality.
'''

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import numpy as np

from Ouroboros.common import get_Mineos_out_dirs, read_input_file

def main():

    # Read input arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_input_file", help = "File path (relative or absolute) to Ouroboros input file.")
    parser.add_argument("n_rows", type = int, help = "Number of rows in diagram.")
    parser.add_argument("n_cols", type = int, help = "Number of columns in diagram.")
    args = parser.parse_args()

    # Rename input arguments.
    path_input  = args.path_to_input_file
    n_rows = args.n_rows
    n_cols = args.n_cols

    # Read the input file and command-line arguments.
    run_info = read_input_file(path_input)
    _, run_info['dir_run'] = get_Mineos_out_dirs(run_info)

    # Load the orthonormality data.
    name_out = 'test_orthonormality.txt'
    print(run_info)
    #path_out = os.path.join(run_info['dir_run'], name_out)
    path_out = os.path.join(run_info['dir_output'], name_out)
    print('Reading {:}'.format(path_out))
    data = np.loadtxt(path_out)
    #
    n_A = data[:, 0].astype(np.int)
    l_A = data[:, 1].astype(np.int)
    n_B = data[:, 2].astype(np.int)
    l_B = data[:, 3].astype(np.int)
    product = data[:, 4]
    
    # Set maximum n-value.
    n_max = 3

    # Set maximum l-value that will fit on plot.
    l_max = (n_rows*n_cols)

    # Find mode pairs with the same l-value which satisfy the n- and l-limits.
    l_cond = (l_A == l_B)
    n_cond = ((n_A <= n_max) & (n_B <= n_max))
    l_cond2 = (l_A <= l_max)
    #
    cond = (l_cond) & (n_cond) & (l_cond2)
    i_cond = np.where(cond)[0]

    # Extract these mode pairs.
    n_A = n_A[i_cond]
    l = l_A[i_cond]
    n_B = n_B[i_cond]
    product = product[i_cond]

    # Fill out a grid of values.
    #l_max = np.max(l)
    grid = np.zeros((l_max - 1, n_max + 1, n_max + 1))
    grid = grid + np.nan
    for l_i in range(2, l_max + 1):

        i = np.where(l == l_i)[0]
        
        for n_j in range(n_max + 1):
            
            j = np.where(n_A[i] == n_j)[0]

            for n_k in range(n_j, n_max + 1):

                k = np.where(n_B[i[j]] == n_k)[0]

                grid[l_i - 2, n_j, n_k] = np.abs(product[i[j[k]]])
    
    # Create the axes.
    fig, ax_arr = plt.subplots(n_rows, n_cols, figsize = (11.0, 8.5),
                    constrained_layout = True, sharex = True, sharey = True)

    # Create a colour scale.
    #norm = LogNorm(vmin = 0.001, vmax = 1.0)
    norm = LogNorm(vmin = 1.0E-6, vmax = 1.0)
    #cmap = 'magma'
    cmap = matplotlib.cm.get_cmap('magma')
    cmap.set_bad(color = 'grey')
    
    # Loop over l-values.
    for l_i in range(2, l_max + 1):
        
        # Find location of axis in axis array.
        i = (l_i - 2)
        k = (i % n_cols)
        j = (i // n_cols)
        #
        ax = ax_arr[j, k]

        # Plot the grid of scalar products for this l-value.
        handle = ax.imshow(grid[l_i - 2, :, :].T,
                norm = norm,
                cmap = cmap)

        # Label the axis.
        ax.text(0.9, 0.9, '$\ell$ = {:d}'.format(l_i),
                transform = ax.transAxes,
                ha = 'right', va = 'top')

        # Equal aspect ratio.
        ax.set_aspect(1.0)

    # Set axis ticks.
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Create the colour bar.
    fig.canvas.draw()
    ax_pos = ax.get_position()
    #cb_ax = fig.add_axes([ax_pos.x0 + 0.15, ax_pos.y0, 0.03, ax_pos.y1 - ax_pos.y0])
    cb_ax = fig.add_axes([ax_pos.x0 + 0.25, ax_pos.y0, 0.03, ax_pos.y1 - ax_pos.y0])
    plt.colorbar(handle, cax = cb_ax, label = '| Scalar product |')

    # Delete empty axes.
    for l_i in range(l_max + 1, (n_rows * n_cols) + 2):

        i = (l_i - 2)
        k = (i % n_cols)
        j = (i // n_cols)

        ax = ax_arr[j, k]

        ax.remove()

    # Save.
    name_out = 'test_orthogonality.png'
    #path_out = os.path.join(run_info['dir_run'], name_out)
    path_out = os.path.join(run_info['dir_output'], name_out)
    print('Saving to {:}'.format(path_out))
    plt.savefig(path_out, dpi = 300)

    plt.show()

    return

if __name__ == '__main__':

    main()
