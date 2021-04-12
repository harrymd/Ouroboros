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
    parser.add_argument("var", choices = ['Q', 'f'])
    parser.add_argument("--show_hist", action = 'store_true', help = 'Plot histogram.')
    input_args = parser.parse_args()
    path_input_Ouroboros = input_args.path_input_Ouroboros
    path_input_Mineos = input_args.path_input_Mineos
    var = input_args.var
    show_hist = input_args.show_hist

    # Read the Ouroboros modes.
    run_info_O = read_Ouroboros_input_file(path_input_Ouroboros)
    var_M_dict = dict()
    var_O_dict = dict()
    for mode_type in run_info_O['mode_types']:

        mode_info_O = load_eigenfreq_Ouroboros(run_info_O, mode_type)
        n_O = mode_info_O['n']
        l_O = mode_info_O['l']
        f_O = mode_info_O['f']
        var_O = mode_info_O[var]
        num_modes_O = len(n_O)
        
        # Read mineos modes.
        run_info_M = read_Mineos_input_file(path_input_Mineos)
        mode_info_M = load_eigenfreq_Mineos(run_info_M, mode_type) 
        n_M = mode_info_M['n']
        l_M = mode_info_M['l']
        f_M = mode_info_M['f']
        var_M = mode_info_M[var]
        
        # Prepare output array.
        n, l, i_align_O, i_align_M = align_mode_lists(n_O, l_O, n_M, l_M)
        var_O = var_O[i_align_O]
        var_M = var_M[i_align_M]

        # Calculate ratio.
        ratio = var_M/var_O
        median_ratio = np.median(ratio)
        
        ## Identify bad cases (for highlighting in plot).
        #cond_bad = (ratio < 0.9*median_ratio) | (ratio > 1.1*median_ratio)
        #i_bad = np.where(cond_bad)[0]
        #i_good = np.where(~cond_bad)[0]

        ## Print bad cases.
        #for i in i_bad:

        #    print(n[i], l[i], f_O[i], var_O[i], var_M[i])
        for i in range(len(var_O)):

            print(mode_type, n[i], l[i], var_O[i]/var_M[i])

        # Store.
        var_M_dict[mode_type] = var_M
        var_O_dict[mode_type] = var_O

    ## Plot histogram.
    #if show_hist:

    #    fig = plt.figure(constrained_layout = True)
    #    ax  = plt.gca()

    #    ax.hist(ratio)

    #    ax.set_xlabel(var)
    #    ax.set_ylabel('Number of occurrences')

    var_M_merged = np.array([y for x in var_M_dict for y in var_M_dict[x]])
    var_O_merged = np.array([y for x in var_O_dict for y in var_O_dict[x]])
    #var_O_merged = np.array([var_O_dict[x] for x in var_O_dict])
    
    show_scatter = True
    if show_scatter:

        fig = plt.figure(figsize = (6.5, 6.5), constrained_layout = True) 
        ax  = plt.gca()
        
        var_max = 0.0
        for mode_type in run_info_O['mode_types']:

            #ax.scatter(var_M[i_good], var_O[i_good], s = 5, alpha = 0.5, c = 'b')
            #ax.scatter(var_M[i_bad], var_O[i_bad], s = 5, alpha = 0.5, c = 'r')
            ax.scatter(var_M_dict[mode_type], var_O_dict[mode_type], s = 5, alpha = 0.5, c = 'k')



            ax.set_aspect(1.0)

            var_max = np.max([np.max(var_M_dict[mode_type]), np.max(var_O_dict[mode_type]), var_max])
            
        #a, _, _, _ = np.linalg.lstsq(var_M_merged[:, np.newaxis], var_O_merged)
        y_max = 1.1*var_max
        ax.plot([0.0, y_max], [0.0, y_max], alpha = 0.5, color = 'k', label  = 'y = x')
        #ax.plot([0.0, y_max], [0.0, y_max*a[0]], alpha = 0.5, color = 'g', label = 'y = {:>5.3f} x'.format(a[0]))

        for mode_type in run_info_O['mode_types']:

            a, _, _, _ = np.linalg.lstsq(var_M_dict[mode_type][:, np.newaxis], var_O_dict[mode_type])
            ax.plot([0.0, y_max], [0.0, y_max*a[0]], alpha = 0.5, color = 'g', label = 'y = {:>5.3f} x'.format(a[0]))
            
        ax.set_xlim([0.0, y_max])
        ax.set_ylim([0.0, y_max])
        
        font_size_label = 13

        ax.set_xlabel('{:} (Mineos)'.format(var), fontsize = font_size_label)
        ax.set_ylabel('{:} (New code)'.format(var), fontsize = font_size_label)

        ax.legend()
        
        path_out = 'scatter_plot_{:}.png'.format(var)
        print('Saving to {:}'.format(path_out))
        plt.savefig(path_out, dpi = 300)

    if show_scatter or show_hist:

        plt.show()

    return

if __name__ == '__main__':

    main()
