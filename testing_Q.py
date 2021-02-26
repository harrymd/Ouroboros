import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from Ouroboros.common import get_Ouroboros_out_dirs, load_eigenfreq_Ouroboros, load_eigenfreq_Mineos, read_Mineos_input_file, read_Ouroboros_input_file
from Ouroboros.plot.plot_dispersion import align_mode_lists

def main():

    # Parse input arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("path_input_Ouroboros", help = "File path (relative or absolute) to Ouroboros input file.")
    parser.add_argument("path_input_Mineos", help = "File path (relative or absolute) to Mineos input file.")
    input_args = parser.parse_args()
    path_input_Ouroboros = input_args.path_input_Ouroboros
    path_input_Mineos = input_args.path_input_Mineos

    # Define constants.
    mode_type = 'S'

    # Read the Ouroboros modes.
    run_info_O = read_Ouroboros_input_file(path_input_Ouroboros)
    mode_info_O = load_eigenfreq_Ouroboros(run_info_O, mode_type)
    n_O = mode_info_O['n']
    l_O = mode_info_O['l']
    f_O = mode_info_O['f']
    Q_O = mode_info_O['Q']
    #_, _, _, dir_output = get_Ouroboros_out_dirs(run_info_O, mode_type)
    #path_Ouroboros_modes = os.path.join(dir_output, 'Q_values.txt')
    #n_O, l_O, f_O, Q_O = np.loadtxt(path_Ouroboros_modes).T
    #n_O = n_O.astype(np.int)
    #l_O = l_O.astype(np.int)
    num_modes_O = len(n_O)
    
    show_no_ka = False
    if show_no_ka:

        path_Ouroboros_modes_no_ka = os.path.join(dir_output, 'Q_values_no_ka.txt')
        _, _, _, Q_O_no_ka = np.loadtxt(path_Ouroboros_modes_no_ka).T

        Q_ratio = Q_O/Q_O_no_ka

        print(Q_ratio)

        fig = plt.figure()
        ax  = plt.gca()
        
        ax.scatter(Q_O, Q_O_no_ka)
        ax.set_aspect(1.0)
        ax.plot([0.0, 0.007], [0.0, 0.007])

        plt.show()

    # Prepare output array.
    run_info_M = read_Mineos_input_file(path_input_Mineos)
    n_M, l_M, f_M, Q_M = load_eigenfreq_Mineos(run_info_M, mode_type, return_Q = True)
    
    n, l, Q_O, Q_M, _ = align_mode_lists(n_O, l_O, Q_O, n_M, l_M, Q_M)

    #Q_M = np.zeros(num_modes_O)

    ## Load Mineos modes.
    #for i in range(num_modes_O):
    #    
    #    try:

    #        _, Q_M[i] = load_eigenfreq_Mineos(run_info_M, mode_type, n_q = n_O[i],
    #                    l_q = l_O[i], return_Q = True)

    #    except IndexError:

    #        Q_M[i] = np.nan

    #i_good = np.where(~np.isnan(Q_M))[0]
    #n = n_O[i_good]
    #l = l_O[i_good]
    #f = f_O[i_good]
    #Q_O = Q_O[i_good]
    #Q_M = Q_M[i_good]

    ratio = Q_M/Q_O
    median_ratio = np.median(ratio)
    print(median_ratio)
    
    cond_bad = (ratio < 0.9*median_ratio) | (ratio > 1.1*median_ratio)
    i_bad = np.where(cond_bad)[0]
    i_good = np.where(~cond_bad)[0]

    for i in i_bad:

        print(n[i], l[i], f_O[i])

    show_hist = False
    if show_hist:

        fig = plt.figure()
        ax  = plt.gca()

        ax.hist(ratio)

        plt.show()
    
    show_scatter = True
    if show_scatter:

        fig = plt.figure(figsize = (6.5, 6.5), constrained_layout = True) 
        ax  = plt.gca()

        ax.scatter(Q_M[i_good], Q_O[i_good], s = 5, alpha = 0.5, c = 'b')
        ax.scatter(Q_M[i_bad], Q_O[i_bad], s = 5, alpha = 0.5, c = 'r')
        ax.set_aspect(1.0)

        ax.plot([0.0, 1000.0], [0.0, 1000.0], alpha = 0.5, color = 'k')

        ax.set_xlim([0.0, 1000.0])
        ax.set_ylim([0.0, 1000.0])
        
        font_size_label = 13
        ax.set_xlabel('Q (Mineos)', fontsize = font_size_label)
        ax.set_ylabel('Q (New code)', fontsize = font_size_label)

        plt.savefig('testing_Q.png', dpi = 300)

        plt.show()

    return

if __name__ == '__main__':

    main()
