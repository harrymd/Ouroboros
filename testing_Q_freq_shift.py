import argparse

import matplotlib.pyplot as plt
import numpy as np

from Ouroboros.common import load_eigenfreq_Ouroboros, load_eigenfreq_Mineos, read_Ouroboros_input_file, read_Mineos_input_file
from Ouroboros.plot.plot_dispersion import align_mode_lists

def main():

    # Parse input arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("path_input_Ouroboros_no_Q", help = "File path (relative or absolute) to Ouroboros input file.")
    parser.add_argument("path_input_Ouroboros_with_Q", help = "File path (relative or absolute) to Ouroboros input file.")
    parser.add_argument("path_input_Mineos_no_Q", help = "File path (relative or absolute) to Mineos input file.")
    parser.add_argument("path_input_Mineos_with_Q", help = "File path (relative or absolute) to Mineos input file.")
    input_args = parser.parse_args()
    path_input_Ouroboros_no_Q = input_args.path_input_Ouroboros_no_Q
    path_input_Ouroboros_with_Q = input_args.path_input_Ouroboros_with_Q
    path_input_Mineos_no_Q = input_args.path_input_Mineos_no_Q
    path_input_Mineos_with_Q = input_args.path_input_Mineos_with_Q
    
    mode_type = 'S'

    run_info_Ouroboros_no_Q = read_Ouroboros_input_file(path_input_Ouroboros_no_Q)
    run_info_Ouroboros_with_Q = read_Ouroboros_input_file(path_input_Ouroboros_with_Q)

    run_info_Mineos_no_Q = read_Mineos_input_file(path_input_Mineos_no_Q)
    run_info_Mineos_with_Q = read_Mineos_input_file(path_input_Mineos_with_Q)

    mode_info_Ouroboros_no_Q = load_eigenfreq_Ouroboros(run_info_Ouroboros_no_Q, mode_type)
    mode_info_Ouroboros_with_Q = load_eigenfreq_Ouroboros(run_info_Ouroboros_with_Q, mode_type)

    n_Mineos_no_Q, l_Mineos_no_Q, f_Mineos_no_Q = load_eigenfreq_Mineos(run_info_Mineos_no_Q, mode_type)
    n_Mineos_with_Q, l_Mineos_with_Q, f_Mineos_with_Q = load_eigenfreq_Mineos(run_info_Mineos_with_Q, mode_type)

    n_Ouroboros_no_Q = mode_info_Ouroboros_no_Q['n']
    l_Ouroboros_no_Q = mode_info_Ouroboros_no_Q['l']
    f_Ouroboros_no_Q = mode_info_Ouroboros_no_Q['f']
    
    n_Ouroboros_with_Q = mode_info_Ouroboros_with_Q['n']
    l_Ouroboros_with_Q = mode_info_Ouroboros_with_Q['l']
    f_Ouroboros_with_Q = mode_info_Ouroboros_with_Q['f']

    f_max = 5.0
    i_f_good = np.where(f_Ouroboros_no_Q < f_max)[0]
    n_Ouroboros_no_Q = n_Ouroboros_no_Q[i_f_good]
    l_Ouroboros_no_Q = l_Ouroboros_no_Q[i_f_good]
    f_Ouroboros_no_Q = f_Ouroboros_no_Q[i_f_good]

    n, l, f_Ouroboros_no_Q, f_Ouroboros_with_Q, i_good = align_mode_lists(
            n_Ouroboros_no_Q, l_Ouroboros_no_Q, f_Ouroboros_no_Q,
            n_Ouroboros_with_Q, l_Ouroboros_with_Q, f_Ouroboros_with_Q)

    n, l, f_Ouroboros_no_Q, f_Mineos_no_Q, i_good = align_mode_lists(
            n_Ouroboros_no_Q, l_Ouroboros_no_Q, f_Ouroboros_no_Q,
            n_Mineos_no_Q, l_Mineos_no_Q, f_Mineos_no_Q)
    f_Ouroboros_with_Q = f_Ouroboros_with_Q[i_good]

    n, l, f_Ouroboros_no_Q, f_Mineos_with_Q, i_good = align_mode_lists(
            n_Ouroboros_no_Q, l_Ouroboros_no_Q, f_Ouroboros_no_Q,
            n_Mineos_with_Q, l_Mineos_with_Q, f_Mineos_with_Q)
    f_Ouroboros_with_Q = f_Ouroboros_with_Q[i_good]
    f_Mineos_no_Q = f_Mineos_no_Q[i_good]

    f_diff_Ouroboros = f_Ouroboros_with_Q - f_Ouroboros_no_Q
    f_diff_Mineos = f_Mineos_with_Q - f_Mineos_no_Q
    
    plot_scatter = False
    if plot_scatter:

        fig = plt.figure()
        ax  = plt.gca()

        #ax.scatter(f_Ouroboros_no_Q, f_Ouroboros_with_Q, alpha = 0.5, s = 1, c = 'k')
        #ax.scatter(f_Ouroboros_no_Q, f_Mineos_no_Q, alpha = 0.5, s = 1, c = 'k')
        #ax.scatter(f_Ouroboros_with_Q, f_Mineos_with_Q, alpha = 0.5, s = 1, c = 'k')
        ax.scatter(f_diff_Ouroboros, f_diff_Mineos, alpha = 0.5, s = 1, c = 'k')
        ax.plot([0.0, 5.0], [0.0, 5.0])

        ax.set_xlabel('No Q')
        ax.set_ylabel('With Q')

        ax.set_xlim([0.0, 5.0])
        ax.set_ylim([0.0, 5.0])

        ax.set_aspect(1.0)

        plt.show()
    
    plot_hist = False 
    if plot_hist:
        
        #ratio = f_Ouroboros_no_Q/f_Ouroboros_with_Q
        #ratio = f_Ouroboros_no_Q/f_Mineos_no_Q
        ratio = f_Ouroboros_with_Q/f_Mineos_with_Q

        fig = plt.figure()
        ax  = plt.gca()

        ax.hist(ratio, 100)

        plt.show()
    
    plot_scatter_diff = False
    if plot_scatter_diff:

        fig = plt.figure()
        ax  = plt.gca()

        ax.scatter(f_diff_Ouroboros, f_diff_Mineos, alpha = 0.5, s = 1, c = 'k')
        #ax.plot([0.0, 5.0], [0.0, 5.0])

        ax.set_xlabel('Ouroboros')
        ax.set_ylabel('Mineos')

        ax.set_aspect(1.0)

    plot_hist_diff = False
    if plot_hist_diff:

        fig = plt.figure()
        ax  = plt.gca()

        ratio = f_diff_Ouroboros/f_diff_Mineos

        ax.hist(ratio, 100)

    plt.show()

    return

if __name__ == '__main__':

    main()
