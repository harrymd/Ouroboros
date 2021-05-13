'''
Plot group and phase speeds calculated by Mineos.
'''

import argparse
import os

import matplotlib.pyplot as plt
from matplotlib import colors as mpl_colors
import numpy as np

from Ouroboros.common import load_eigenfreq, read_input_file

def plot_speed(mode_info, speed_type, n_max = 4, dir_out = None):
    '''
    Plot phase or group speed. Note twin x axes.
    '''

    # Check inputs.
    assert speed_type in ['group', 'phase']

    # Unpack.
    n = mode_info['n']
    l = mode_info['l']
    f_mHz = mode_info['f']
    f_Hz = 1.0E-3*f_mHz
    T = 1.0/f_Hz # Period in s.
    if speed_type == 'group':

        v = mode_info['u']

    elif speed_type == 'phase':

        v = mode_info['c']

    # Get unique sorted list of n values.
    n_list = np.sort(np.unique(n))
    n_list = n_list[n_list <= n_max]
    num_n_values = len(n_list)

    # Create colour scale.
    c_map = plt.get_cmap('rainbow_r')
    c_norm = mpl_colors.Normalize(vmin = 0.0, vmax = (num_n_values - 1.0))

    # Make axes.
    fig = plt.figure(figsize = (8.0, 6.0), constrained_layout = True)
    ax  = plt.gca()

    def forward(T):
        
        # s to mHz.
        return 1.0E3/T

    def reverse(f_mHz):

        # mHz to s.
        return 1.0e3/f_mHz

    # Define a closure function to register as a callback
    def convert_ax_T_to_f(ax):
        """
        Update second axis according with first axis.
        See https://matplotlib.org/stable/gallery/subplots_axes_and_figures/fahrenheit_celsius_scales.html
        """

        x1, x2 = ax.get_xlim()
        ax_1.set_xlim(1.0E3/x2, 1.0E3/x1)
        ax_1.figure.canvas.draw()

    ax_1 = ax.twiny()
    ax_1.set_xscale('function', functions=(forward, reverse))

    # automatically update ylim of ax2 when ylim of ax1 changes.
    ax.callbacks.connect("xlim_changed", convert_ax_T_to_f)

    # Plot each branch.
    for i in range(num_n_values):

        # Find the modes from this branch.
        j = np.where(n == n_list[i])[0]

        # Plot.
        #ax.plot(f_mHz[j], v[j])
        #ax.scatter(f_mHz[j], v[j])

        color = c_map(c_norm(i))

        ax.plot(T[j], v[j], color = color)
        ax.scatter(T[j], v[j], color = color)

        # Dummy plot for legend.
        ax.plot([], [],
            linestyle = '-',
            marker = '.',
            label = '{:>d}'.format(n_list[i]),
            color = color)
    
    # Set y-axis limits and label.
    font_size_label = 12
    if speed_type == 'group':

        #y_lims = [0.0, 21.0]
        #y_lims = [2.5, 17.5]
        y_label = 'Group speed (km s$^{-1}$)'

    elif speed_type == 'phase':

        y_label = 'Phase speed (km s$^{-1}$)'
        #y_lims = [2.5, 26.0]
        #y_lims = [7.5, 10.0]

    #ax.set_ylim(y_lims)
    ax.set_ylabel(y_label, fontsize = font_size_label)
    
    # Add legend.
    ax.legend(title = '$n$')

    #x_lims = [0.0, 400.0]
    #x_lims = [100.0, 3500.0]
    #x_lims = [100.0, 500.0]
    x_label = 'Period (s)'
    x_label_alt = 'Frequency (mHz)'
    #ax.set_xscale('log')

    #ax.set_xlim(x_lims)
    ax.set_xlabel(x_label, fontsize = font_size_label)
    ax_1.set_xlabel(x_label_alt, fontsize = font_size_label)

    if dir_out is not None:
        
        if speed_type == 'group':

            name_out = 'group_speed.png'

        elif speed_type == 'phase':

            name_out = 'phase_speed.png'

        path_out = os.path.join(dir_out, name_out)
        print('Writing to {:}'.format(path_out))
        plt.savefig(path_out, dpi = 300)

    plt.show()

    return

def main():

    # Read input arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("path_input", help = "File path (relative or absolute) to input file.")
    parser.add_argument("type", choices = ['phase', 'group'], help = 'Choose whether to plot phase or group speed.')
    args = parser.parse_args()

    # Rename input arguments.
    path_input = args.path_input    
    speed_type = args.type

    # Read input file.
    run_info = read_input_file(path_input)

    # Load modes.
    mode_type = 'S'
    mode_info = load_eigenfreq(run_info, mode_type)

    # Plot.
    dir_out = os.path.join(run_info['dir_output'], 'plots')
    plot_speed(mode_info, speed_type, dir_out = dir_out)

    return

if __name__ == '__main__':

    main()
