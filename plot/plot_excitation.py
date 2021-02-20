import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas

def main():

    # Parse input arguments.
    parser = argparse.ArgumentParser()
    #
    parser.add_argument("dir_output", help = "Path to directory containing coefficients and mode information")
    parser.add_argument("station", help = 'Station code.')
    #
    args = parser.parse_args()
    #path_coeffs = args.path_coeffs
    dir_output = args.dir_output
    station = args.station

    # Load mode data.
    path_modes = os.path.join(dir_output, 'modes.pkl')
    print('Loading {:}'.format(path_modes))
    modes = pandas.read_pickle(path_modes)

    # Load coeff data.
    path_coeffs = os.path.join(dir_output, 'coeffs.pkl')
    print('Loading {:}'.format(path_coeffs))
    coeffs = pandas.read_pickle(path_coeffs)
    #
    coeffs = coeffs.loc[station]
    mode_type_list = np.unique(modes['type'])

    line_kwargs = {'linewidth' : 1, 'alpha' : 0.3}
    scatter_kwargs = {}
    color_dict = {'R' : 'red', 'S' : 'blue', 'T' : 'green'}
    #mode_type_dict = {0 : 'R', 1 : 'S', 2 : 'T'}
    
    comp_list = ['A_r', 'A_Theta', 'A_Phi']
    abs_list = [np.abs(coeffs[comp]) for comp in comp_list]
    max_abs_list = [np.max(abs_) for abs_ in abs_list]
    max_abs = np.max(max_abs_list)
    abs_A_thresh = 0.001*max_abs

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
    fig, ax_arr = plt.subplots(1, 3, figsize = (14.0, 7.0), sharex = True,
                    sharey = True, constrained_layout = True)

    for i in range(3):

        ax = ax_arr[i]

        for mode_type in mode_type_list:

            #mode_type = mode_type_dict[mode_type_int]
            color = color_dict[mode_type]

            j = np.where(modes['type'] == mode_type)[0]

            n = modes['n'][j]
            l = modes['l'][j]
            f = modes['f'][j]
            A = coeffs[comp_list[i]][j]
            abs_A = np.abs(A)

            s = s_min + s_range*abs_A/max_abs
            
            jj_cond = (abs_A > abs_A_thresh)
            jj = np.where(jj_cond)[0]
            jjj = np.where(~jj_cond)[0]

            ax.scatter(l[jj], f[jj], color = color, s = s[jj], **scatter_kwargs)
            ax.scatter(l[jjj], f[jjj], color = color, s = s[jjj], facecolor = 'white', **scatter_kwargs)

            n_list = np.sort(np.unique(n))

            for n_k in n_list:

                k = np.where(n == n_k)[0]
                l_k = l[k]
                f_k = f[k]

                ax.plot(l_k, f_k, color = color, **line_kwargs)
    
    font_size_label = 12
    font_size_title = 24
    
    #ax_labels = ['$r$ component', '$\Theta$ component', '$\Phi$ component']
    ax_labels = ['$r$', '$\Theta$', '$\Phi$']
    for i in range(3):

        ax = ax_arr[i]
        ax_label = ax_labels[i]

        ax.set_xlabel('Angular order, $\ell$', fontsize = font_size_label)
        ax.set_ylabel('Frequency (mHz)', fontsize = font_size_label)
        ax.text(0.9, 0.05, ax_label, transform = ax.transAxes, fontsize = font_size_title,
                    ha = 'right', va = 'bottom')
    
    plt.suptitle('Excitation coefficients for station {:>5}'.format(station),
            fontsize = font_size_title)
    plt.show()

    #print(coeffs['l'])

    # Create figure.
    #fig = plt.figure()
    #ax  = plt.gca()



    return

if __name__ == '__main__':

    main()
