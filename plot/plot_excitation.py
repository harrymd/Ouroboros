'''
Plot the mode excitation coefficients used in mode summation.
'''

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas

from Ouroboros.common import mkdir_if_not_exist

def plot_coeffs_one_station(dir_output, station, path_save_fig, comps, f_lims, l_lims, A_lims, fig_size, show = True):
    '''
    Plot the excitation coefficients for a single station.
    '''

    # Count how many components are being plotted.
    n_comps = len(comps)

    # Load mean excitation.
    if station == 'mean':

        # Load mode data.
        path_modes = os.path.join(dir_output, 'modes_mean.pkl')
        print('Loading {:}'.format(path_modes))
        modes = pandas.read_pickle(path_modes)

        # Load coeff data.
        path_coeffs = os.path.join(dir_output, 'coeffs_mean.pkl')
        print('Loading {:}'.format(path_coeffs))
        coeffs = pandas.read_pickle(path_coeffs)

        station_loc_info = None

    # Load excitation for a specific station.
    else:

        # Load mode data.
        path_modes = os.path.join(dir_output, 'modes.pkl')
        print('Loading {:}'.format(path_modes))
        modes = pandas.read_pickle(path_modes)

        # Load station data.
        path_stations = os.path.join(dir_output, 'stations.pkl')
        print('Loading {:}'.format(path_stations))
        stations = pandas.read_pickle(path_stations)
        #
        station_info = stations.loc[station]
        station_loc_info = {'Theta' : station_info['Theta'],
                            'Phi'   : station_info['Phi']}

        # Load coeff data.
        path_coeffs = os.path.join(dir_output, 'coeffs.pkl')
        print('Loading {:}'.format(path_coeffs))
        coeffs = pandas.read_pickle(path_coeffs)
        #
        coeffs = coeffs.loc[station]

    # Get list of mode types.
    mode_type_list = np.unique(modes['type'])

    # Set plotting parameters.
    line_kwargs = {'linewidth' : 1, 'alpha' : 0.3}
    scatter_kwargs = {'alpha' : 0.5}
    color_dict = {'R' : 'red', 'S' : 'blue', 'T' : 'green'}

    # Get list of component names.
    comp_list = ['A_{:}'.format(x) for x in comps]

    # Get list of absolute values of components.
    abs_list = [np.abs(coeffs[comp]) for comp in comp_list]
    max_abs_list = [np.max(abs_) for abs_ in abs_list]

    # Determine limits of excitation coefficients.
    if A_lims is None:

        max_abs_list = np.zeros(n_comps)

        # Loop over coefficients and look for maximum value.
        for i in range(n_comps):

            for mode_type in mode_type_list:

                j = np.where(modes['type'] == mode_type)[0]

                n = np.array(modes['n'][j], dtype = np.int)
                l = np.array(modes['l'][j], dtype = np.int)
                f = np.array(modes['f'][j])
                A = np.array(coeffs[comp_list[i]][j])
                abs_A = np.abs(A)
                    
                if (f_lims is None) and (l_lims is None):

                    plot_cond = np.ones(f.shape, dtype = np.bool)

                elif (f_lims is None) and (l_lims is not None):

                    plot_cond = (l < l_lims[1]) & (l > l_lims[0])

                elif (f_lims is not None) and (l_lims is None):

                    plot_cond = (f < f_lims[1]) & (f > f_lims[0])
                
                elif (f_lims is not None) and (l_lims is not None):

                    plot_cond = (f < f_lims[1]) & (f > f_lims[0]) & (l < l_lims[1]) & (l > l_lims[0])

                jj = np.where(plot_cond)[0]

                max_abs_list[i] = np.max(abs_A[jj])

        max_abs = np.max(max_abs_list)
        abs_A_thresh = 0.001*max_abs

    else:
        
        # Use pre-defined limits.
        abs_A_thresh = A_lims[0]
        max_abs = A_lims[1]

    # Set size range of points in scatter plot.
    s_min   =  10.0
    s_max   = 300.0
    s_range = s_max - s_min

    # Determine size of figure if not specified.
    if fig_size is None:

        fig_size_x = n_comps*5.0
        fig_size_y = 7.0
        fig_size = (fig_size_x, fig_size_y)

    # Create axes.
    fig, ax_arr = plt.subplots(1, n_comps, figsize = fig_size, sharex = True,
                    sharey = True, constrained_layout = True)

    # Ensure axis list is indexable.
    if n_comps == 1:

        ax_arr = [ax_arr]

    # Loop over components.
    for i in range(n_comps):
        
        ax = ax_arr[i]

        # Plot the coefficients for the specified mode type.
        for mode_type in mode_type_list:

            #mode_type = mode_type_dict[mode_type_int]
            color = color_dict[mode_type]

            j = np.where(modes['type'] == mode_type)[0]

            n = np.array(modes['n'][j], dtype = np.int)
            l = np.array(modes['l'][j], dtype = np.int)
            f = np.array(modes['f'][j])
            A = np.array(coeffs[comp_list[i]][j])
            abs_A = np.abs(A)
                
            # Set the size of the points.
            s = s_min + s_range*abs_A/max_abs
            
            # Only plot points above a certain amplitude and within the
            # frequency and l-limits.
            amp_cond = (abs_A > abs_A_thresh)
            plot_cond = amp_cond
            if f_lims is not None:

                f_cond = (f < f_lims[1]) & (f > f_lims[0])
                plot_cond = (plot_cond & f_cond)

            if l_lims is not None:

                l_cond = (l < l_lims[1]) & (l > l_lims[0])
                plot_cond = (plot_cond & l_cond)

            # Find the indices of the points satisfying the limits (and not).
            jj = np.where(plot_cond)[0]
            jjj = np.where(~plot_cond)[0]

            # Scatter both sets of points.
            ax.scatter(l[jj], f[jj], color = color, s = s[jj], **scatter_kwargs)
            ax.scatter(l[jjj], f[jjj], color = color, s = s[jjj], facecolor = 'white', **scatter_kwargs)

            # Get a list of n-values.
            n_list = np.sort(np.unique(n))

            # Join overtones by a continuous line.
            for n_k in n_list:

                k = np.where(n == n_k)[0]
                l_k = l[k]
                f_k = f[k]

                ax.plot(l_k, f_k, color = color, **line_kwargs)

        # Make a legend showing the size of the points.
        ax.scatter([], [], color = 'k', s = s_min,                  **scatter_kwargs, facecolor = 'white', label = 'Less than {:.1e}'.format(abs_A_thresh))
        ax.scatter([], [], color = 'k', s = s_min,                  **scatter_kwargs, facecolor = 'k', label = '{:.1e}'.format(abs_A_thresh))
        ax.scatter([], [], color = 'k', s = s_min + 0.1*s_max,      **scatter_kwargs, facecolor = 'k', label = '{:.1e}'.format(0.1*max_abs))
        ax.scatter([], [], color = 'k', s = s_max,                  **scatter_kwargs, facecolor = 'k', label = '{:.1e}'.format(max_abs))
        ax.legend(title = 'Excitation magnitude', loc = 'best')#'center right')
    
    # Set label font sizes.
    font_size_label = 12
    font_size_title = 12
    
    # Label the axes.
    ax_labels_dict = {'r' : '$r$', 'Theta' : '$\Theta$', 'Phi' : '$\Phi$'}
    for i in range(n_comps):

        ax = ax_arr[i]
        ax_label = ax_labels_dict[comps[i]]

        ax.set_xlabel('Angular order, $\ell$', fontsize = font_size_label)
        ax.set_ylabel('Frequency (mHz)', fontsize = font_size_label)
        ax.text(0.9, 0.05, ax_label, transform = ax.transAxes, fontsize = font_size_title,
                    ha = 'right', va = 'bottom')
    
    # Set axis limits.
    if l_lims is not None:

        ax_arr[0].set_xlim(*l_lims)

    if f_lims is not None:

        ax_arr[0].set_ylim(*f_lims)
    
    # Label with station location information.
    if station_loc_info is not None: 

        #title = 'Excitation coefficients for station {:>5}'.format(station)
        title = 'Epicentral distance: {:>7.2f} $\degree$, azimuth: {:>7.2f} $\degree$'.format(
                    np.rad2deg(station_loc_info['Theta']), np.rad2deg(station_loc_info['Phi']))

        h_title = plt.suptitle(title, fontsize = font_size_title, family = 'monospace')
        #plt.setp(h_title.texts, family = 'Consolas')

    # Save.
    if path_save_fig is not None:

        print("Saving to {:}".format(path_save_fig))
        plt.savefig(path_save_fig, dpi = 300)
    
    # Show.
    if show:

        plt.show()

    # Close.
    plt.close()

    return

def plot_coeffs_all_stations(dir_output, path_save_fig, comps, f_lims, l_lims, A_lims, fig_size):
    '''
    Loop over all stations and plot the coefficients for each one.
    To make into mp4 animation

    ffmpeg -r 24 -vcodec libx264 -crf 25 -pix_fmt yuv420p test.mp4 -pattern_type glob -r 5 -i '../../output/mineos/prem_noocean_at_03.000_mHz/00020_00070_2/summation_Ouroboros/great_circle_000_station_list/point_00000/coeff_plots/coeffs_*.png'
    '''

    # Load mode data.
    path_modes = os.path.join(dir_output, 'modes.pkl')
    print('Loading {:}'.format(path_modes))
    modes = pandas.read_pickle(path_modes)

    # Load station data.
    path_stations = os.path.join(dir_output, 'stations.pkl')
    print('Loading {:}'.format(path_stations))
    stations = pandas.read_pickle(path_stations)

    # Create output directory.
    dir_coeff_plots = os.path.join(dir_output, 'coeff_plots')
    mkdir_if_not_exist(dir_coeff_plots)

    # Loop over stations.
    for i in range(0, 201):
        
        station = '{:>05d}'.format(i)

        path_save_fig = os.path.join(dir_coeff_plots, 'coeffs_{:>05d}.png'.format(i))

        plot_coeffs_one_station(dir_output, station, path_save_fig, comps, f_lims, l_lims, A_lims, fig_size, show = False)

    return

def main():

    # Parse input arguments.
    parser = argparse.ArgumentParser()
    #
    parser.add_argument("dir_output", help = "Path to directory containing coefficients and mode information")
    parser.add_argument("station", help = 'Station code.')
    parser.add_argument("--path_save_fig", help = 'Name of output figure.')
    parser.add_argument("--comps", choices = ['r', 'Theta', 'Phi'], default = ['r', 'Theta', 'Phi'], nargs = "+")
    parser.add_argument("--f_lims", nargs = 2, type = float)
    parser.add_argument("--l_lims", nargs = 2, type = float)
    parser.add_argument("--A_lims", nargs = 2, type = float)
    parser.add_argument("--fig_size", nargs = 2, type = float)
    #
    args = parser.parse_args()
    #path_coeffs = args.path_coeffs
    dir_output = args.dir_output
    station = args.station
    path_save_fig = args.path_save_fig
    comps = args.comps
    f_lims = args.f_lims
    l_lims = args.l_lims
    A_lims = args.A_lims
    fig_size = args.fig_size

    # Plot coeffficients for one or all stations.
    if station == 'all':

        plot_coeffs_all_stations(dir_output, path_save_fig, comps, f_lims, l_lims, A_lims, fig_size)

    else:

        plot_coeffs_one_station(dir_output, station, path_save_fig, comps, f_lims, l_lims, A_lims, fig_size)

    return

if __name__ == '__main__':

    main()
