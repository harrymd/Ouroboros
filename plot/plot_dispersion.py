import argparse
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from common import mkdir_if_not_exist, read_Mineos_input_file, read_RadialPNM_input_file 
from stoneley.code.common.Mineos import load_eigenfreq_Mineos
from post.read_output import get_dir_eigval_RadialPNM, load_eigenfreq_RadialPNM
#from stoneley.code.mineos.process_mineos import read_minos_bran_output

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--toroidal", dest = "layer_number", help = "Plot toroidal modes for the solid shell given by LAYER_NUMBER (0 is outermost solid shell).", type = int)
    parser.add_argument("--mineos", action = "store_true", help = "Plot Mineos modes (default: RadialPNM).")
    args = parser.parse_args()

    # Rename input arguments.
    i_toroidal = args.layer_number
    use_mineos = args.mineos

    # Read the input file.
    if use_mineos:

        path_model, dir_output, g_switch, mode_type, l_min, l_max, n_min, n_max = \
        read_Mineos_input_file()

    else:

        path_model, dir_output, g_switch, mode_type, l_min, l_max, n_min, n_max, n_layers = \
        read_RadialPNM_input_file()

    if mode_type == 'R':

        print('Cannot plot dispersion (l versus f) for radial modes (which all have l = 0). However, try changing mode_type to \'S\', radial modes will be included in the plot.')
        raise ValueError

    #if mode_type == 'T':

    #    assert len(sys.argv) == 2, 'Usage for plotting toroidal modes: python3 plot/plot_dispersion.py 0, where 0 is the layer number.'

    #    i_toroidal = int(sys.argv[1])

    #else:

    #    assert len(sys.argv) == 1, 'Usage for plotting spheroidal modes: python3 plot/plot_dispersion.py'
    #    i_toroidal = None

    # Get the name of the model from the file.
    name_model = os.path.splitext(os.path.basename(path_model))[0]

    # Store n- and l- limits as arrays.
    n_lims      = [n_min, n_max]
    l_lims      = [l_min, l_max]

    run_info = {        'dir_output': dir_output, 
                        'path_model' : path_model,
                        'model'     : name_model,
                        'n_lims'    : n_lims,
                        'l_lims'    : l_lims,
                        'g_switch'  : g_switch,
                        'use_mineos': use_mineos}

    if not use_mineos:

        run_info['n_layers'] = n_layers

    plot_dispersion_wrapper(run_info, mode_type, i_toroidal = i_toroidal)

    return

def plot_dispersion_wrapper(run_info, mode_type, ax = None, save = True, show = True, i_toroidal = None):
    
    # Get frequency information.
    if run_info['use_mineos']:

        if i_toroidal is not None:

            raise NotImplementedError('Dispersion plotting for Mineos toroidal modes not implemented yet.')
        
        #name_run = '{:>05d}_{:>05d}_{:>1d}'.format(run_info['l_lims'][1], run_info['n_lims'][1], run_info['g_switch'])
        #dir_run = os.path.join(run_info['dir_output'], run_info['model'], name_run)
        #path_minos_bran = os.path.join(dir_run, 'minos_bran_out_{:}.txt'.format(mode_type))
        #mode_data = read_minos_bran_output(path_minos_bran)
        #print(mode_data)
        #n = mode_data['n']
        #l = mode_data['l']
        #f = mode_data['w'] # Freq in mHz.

        n, l, f = load_eigenfreq_Mineos(run_info, mode_type)

    else:

        n, l, f = load_eigenfreq_RadialPNM(run_info, mode_type, i_toroidal = i_toroidal)
    
    # Try to also load radial modes.
    if mode_type == 'S':
    
        try:

            if run_info['use_mineos']:

                #path_minos_bran_R = os.path.join(dir_run, 'minos_bran_out_{:}.txt'.format('R'))
                #mode_data_R = read_minos_bran_output(path_minos_bran_R)
                #n_R = mode_data_R['n']
                #l_R = mode_data_R['l']
                #f_R = mode_data_R['w'] # Freq in mHz.
                #nlf_radial = (n_R, l_R, f_R)
                nlf_R = load_eigenfreq_Mineos(run_info, 'R')

            else:

                nlf_radial = load_eigenfreq_RadialPNM(run_info, 'R')
    
        except OSError:

            print('No radial mode data found; plot will not include radial modes.')
            nlf_radial = None

    else:

        nlf_radial = None

    alpha = 0.5

    if ax is None:

        fig = plt.figure(figsize = (9.5, 5.5))
        ax  = plt.gca()
     
    ax = plot_dispersion(n, l, f, ax = ax, show = False, color = 'b', c_scatter = 'k', alpha = alpha, add_legend = False, nlf_radial = nlf_radial)

    #ax.set_ylim([0.0, 14.0])
    #ax.set_xlim([0.0, 60.0])

    if save:
        
        fig_name = 'dispersion.png'
        if run_info['use_mineos']:

            dir_out = run_info['dir_output'] 

        else:

            dir_out = get_dir_eigval_RadialPNM(run_info, mode_type)

        dir_plot = os.path.join(dir_out, 'plots')
        mkdir_if_not_exist(dir_plot)
        fig_path = os.path.join(dir_plot, fig_name)
        print('Saving figure to {:}'.format(fig_path))
        plt.savefig(fig_path, dpi = 300, bbox_inches = 'tight')

    if show:
    
        plt.show()

def plot_dispersion(n, l, f, ax = None, x_lim = None, y_lim = 'auto', x_label = 'Angular order, $\ell$', y_label = 'Frequency / mHz', title = None, h_lines = None, path_fig = None, show = True, color = 'k', c_scatter = 'k', alpha = 1.0, label = None, add_legend = False, nlf_radial = None):

    n_list = sorted(list(set(n)))
    
    if ax is None:
    
        fig = plt.figure()
        ax  = plt.gca()
    
    for ni in n_list:
        
        i = (n == ni)
        
        ax.plot(l[i], f[i], ls = '-', color = color, lw = 1, alpha = alpha)
        
    point_size = 1
    ax.scatter(l, f, s = point_size, c = c_scatter, alpha = alpha)

    if nlf_radial is not None:

        n_radial, l_radial, f_radial = nlf_radial
        ax.scatter(l_radial, f_radial, s = point_size, c = c_scatter, alpha = alpha)
    
    if label is not None:

        ax.plot([], [], linestyle = '-', marker = '.', color = color, alpha = alpha, label = label)

    font_size_label = 18
    
    if x_lim is not None:

        ax.set_xlim(x_lim)

    ax.xaxis.set_major_locator(MaxNLocator(integer = True))

    if y_lim is not None:

        if y_lim == 'auto':

            f_min = 0.0
            f_max = np.max(f)
            buff = (f_max - f_min)*0.05
            y_lim = [f_min, f_max + buff]

        ax.set_ylim(y_lim)

    if x_label is not None:

        ax.set_xlabel(x_label, fontsize = font_size_label)

    if y_label is not None:

        ax.set_ylabel(y_label, fontsize = font_size_label)

    if title is not None:
        
        ax.set_title(title, fontsize = font_size_label)
    
    if h_lines is not None:

        for h_line in h_lines:

            ax.axhline(h_line, color = 'k', linestyle = ':')

    if add_legend:

        plt.legend()

    plt.tight_layout()

    if path_fig is not None:
        
        plt.savefig(path_fig,
            dpi         = 300,
            transparent = True,
            bbox_inches = 'tight')
            
    if show:
        
        plt.show()
        
    return ax

if __name__ == '__main__':

    main()
