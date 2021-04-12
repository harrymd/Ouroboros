import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas

from Ouroboros.common import mkdir_if_not_exist

def plot_coeffs_one_station(dir_output, station, path_save_fig, comps, f_lims, l_lims, A_lims, fig_size, show = True):

    n_comps = len(comps)

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

    mode_type_list = np.unique(modes['type'])

    line_kwargs = {'linewidth' : 1, 'alpha' : 0.3}
    scatter_kwargs = {'alpha' : 0.5}
    color_dict = {'R' : 'red', 'S' : 'blue', 'T' : 'green'}
    #mode_type_dict = {0 : 'R', 1 : 'S', 2 : 'T'}
    
    #comp_list = ['A_r', 'A_Theta', 'A_Phi']
    comp_list = ['A_{:}'.format(x) for x in comps]
    abs_list = [np.abs(coeffs[comp]) for comp in comp_list]
    max_abs_list = [np.max(abs_) for abs_ in abs_list]

    if A_lims is None:

        #max_abs = np.max(max_abs_list)
        #abs_A_thresh = 0.001*max_abs

        max_abs_list = np.zeros(n_comps)

        for i in range(n_comps):

            for mode_type in mode_type_list:

                j = np.where(modes['type'] == mode_type)[0]

                n = np.array(modes['n'][j], dtype = np.int)
                l = np.array(modes['l'][j], dtype = np.int)
                f = np.array(modes['f'][j])
                A = np.array(coeffs[comp_list[i]][j])
                abs_A = np.abs(A)

                plot_cond = (f < f_lims[1]) & (f > f_lims[0]) & (l < l_lims[1]) & (l > l_lims[0])
            
                jj = np.where(plot_cond)[0]

                max_abs_list[i] = np.max(abs_A[jj])

        max_abs = np.max(max_abs_list)
        abs_A_thresh = 0.001*max_abs

    else:
        
        abs_A_thresh = A_lims[0]
        max_abs = A_lims[1]

    s_min   =  10.0
    s_max   = 300.0
    s_range = s_max - s_min

    #fig = plt.figure()
    #ax  = plt.gca()
    #for comp in comp_list:

    #    ax.hist(np.abs(coeffs[comp]), alpha = 0.5, label = comp)

    #ax.legend()

    #
    #plt.show()

    #import sys
    #sys.exit()

    #fig = plt.figure(figsize = (8.5, 11.0))
    #ax  = plt.gca()

    if fig_size is None:

        fig_size_x = n_comps*5.0
        fig_size_y = 7.0
        fig_size = (fig_size_x, fig_size_y)

    fig, ax_arr = plt.subplots(1, n_comps, figsize = fig_size, sharex = True,
                    sharey = True, constrained_layout = True)

    if n_comps == 1:

        ax_arr = [ax_arr]

    #color_pos = 'r'
    #color_neg = 'b'

    for i in range(n_comps):
        
        ax = ax_arr[i]

        for mode_type in mode_type_list:

            #mode_type = mode_type_dict[mode_type_int]
            color = color_dict[mode_type]

            j = np.where(modes['type'] == mode_type)[0]

            n = np.array(modes['n'][j], dtype = np.int)
            l = np.array(modes['l'][j], dtype = np.int)
            f = np.array(modes['f'][j])
            A = np.array(coeffs[comp_list[i]][j])
            abs_A = np.abs(A)
                
            #color = []
            #n_modes = len(A)
            #for k in range(n_modes):

            #    if A[i] > 0.0:

            #        color.append(color_pos)

            #    else:

            #        color.append(color_neg)
            #color = np.array(color)

            s = s_min + s_range*abs_A/max_abs
            
            amp_cond = (abs_A > abs_A_thresh)
            plot_cond = amp_cond
            if f_lims is not None:

                f_cond = (f < f_lims[1]) & (f > f_lims[0])
                plot_cond = (plot_cond & f_cond)

            if l_lims is not None:

                l_cond = (l < l_lims[1]) & (l > l_lims[0])
                plot_cond = (plot_cond & l_cond)

            jj = np.where(plot_cond)[0]
            jjj = np.where(~plot_cond)[0]

            ax.scatter(l[jj], f[jj], color = color, s = s[jj], **scatter_kwargs)
            ax.scatter(l[jjj], f[jjj], color = color, s = s[jjj], facecolor = 'white', **scatter_kwargs)

            for pp in jj:

                #print('{:>4d} {:>4d} {:>.4e}'.format(n[pp], l[pp], abs_A[pp]))
                print('{:>4d} {:>4d} {:>12.1f}'.format(n[pp], l[pp], 1.0E10*abs_A[pp]))

            n_list = np.sort(np.unique(n))

            for n_k in n_list:

                k = np.where(n == n_k)[0]
                l_k = l[k]
                f_k = f[k]

                ax.plot(l_k, f_k, color = color, **line_kwargs)

        ax.scatter([], [], color = 'k', s = s_min,                  **scatter_kwargs, facecolor = 'white', label = 'Less than {:.1e}'.format(abs_A_thresh))
        ax.scatter([], [], color = 'k', s = s_min,                  **scatter_kwargs, facecolor = 'k', label = '{:.1e}'.format(abs_A_thresh))
        ax.scatter([], [], color = 'k', s = s_min + 0.1*s_max,      **scatter_kwargs, facecolor = 'k', label = '{:.1e}'.format(0.1*max_abs))
        ax.scatter([], [], color = 'k', s = s_max,                  **scatter_kwargs, facecolor = 'k', label = '{:.1e}'.format(max_abs))
        ax.legend(title = 'Excitation magnitude', loc = 'center right')
    
    font_size_label = 12
    #font_size_title = 24
    font_size_title = 14
    
    #ax_labels = ['$r$ component', '$\Theta$ component', '$\Phi$ component']
    #ax_labels = ['$r$', '$\Theta$', '$\Phi$']
    ax_labels_dict = {'r' : '$r$', 'Theta' : '$\Theta$', 'Phi' : '$\Phi$'}
    for i in range(n_comps):

        ax = ax_arr[i]
        #ax_label = ax_labels[i]
        ax_label = ax_labels_dict[comps[i]]

        ax.set_xlabel('Angular order, $\ell$', fontsize = font_size_label)
        ax.set_ylabel('Frequency (mHz)', fontsize = font_size_label)
        ax.text(0.9, 0.05, ax_label, transform = ax.transAxes, fontsize = font_size_title,
                    ha = 'right', va = 'bottom')
    
    if l_lims is not None:

        ax_arr[0].set_xlim(*l_lims)

    if f_lims is not None:

        ax_arr[0].set_ylim(*f_lims)
    
    hlines = [4.69, 4.90, 5.90, 6.11]
    for hline in hlines:
        
        ax.axhline(hline, linestyle = '-', color = 'k', alpha = 0.5, lw = 3)
    
    #show_title = False
    #if show_title:
    if station_loc_info is not None: 

        #title = 'Excitation coefficients for station {:>5}'.format(station)
        title = 'Epicentral distance: {:>7.2f} $\degree$, azimuth: {:>7.2f} $\degree$'.format(
                    np.rad2deg(station_loc_info['Theta']), np.rad2deg(station_loc_info['Phi']))

        h_title = plt.suptitle(title, fontsize = font_size_title, family = 'monospace')
        #plt.setp(h_title.texts, family = 'Consolas')

    if path_save_fig is not None:

        print("Saving to {:}".format(path_save_fig))
        plt.savefig(path_save_fig, dpi = 300)
    
    if show:

        plt.show()

    plt.close()

    return

def plot_coeffs_all_stations(dir_output, path_save_fig, comps, f_lims, l_lims, A_lims, fig_size):
    '''
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

    ## Load coeff data.
    #path_coeffs = os.path.join(dir_output, 'coeffs.pkl')
    #print('Loading {:}'.format(path_coeffs))
    #coeffs = pandas.read_pickle(path_coeffs)

    dir_coeff_plots = os.path.join(dir_output, 'coeff_plots')
    mkdir_if_not_exist(dir_coeff_plots)

    # Loop over stations.
    #for i in range(0, 201, 20):
    for i in range(0, 201):
    #for i in range(2):
        
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

    # Plot coeffficients for one station.
    if station == 'all':

        plot_coeffs_all_stations(dir_output, path_save_fig, comps, f_lims, l_lims, A_lims, fig_size)

    else:

        plot_coeffs_one_station(dir_output, station, path_save_fig, comps, f_lims, l_lims, A_lims, fig_size)

    return

if __name__ == '__main__':

    main()
