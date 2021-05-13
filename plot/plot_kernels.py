'''
Plot sensitivity kernels.
'''

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from Ouroboros.plot.plot_kernels_brute import get_kernel_brute
from Ouroboros.common import (  get_Ouroboros_out_dirs, mkdir_if_not_exist,
                                read_input_file)
from Ouroboros.common import (  load_eigenfreq, load_kernel, load_model,
                                get_r_fluid_solid_boundary)

def main():

    # Parse the command-line arguments.
    parser = argparse.ArgumentParser()
    #
    parser.add_argument('path_input', help = 'File path (relative or absolute) to Ouroboros input file.')
    parser.add_argument('mode_type', choices = ['R', 'S', 'T'], help = 'Mode type (radial, spheroidal, or toroidal).')
    parser.add_argument('n', type = int, help = 'Radial order.')
    parser.add_argument('l', type = int, help = 'Angular order.')
    parser.add_argument("--i_toroidal", dest = "layer_number", help = "Plot toroidal modes for the solid shell given by LAYER_NUMBER (0 is outermost solid shell). Default is to plot spheroidal modes.", type = int)
    #parser.add_argument('param', choices = ['ka', 'mu', 'rho'], help = 'Plot sensitivity to bulk modulus (ka), shear modulus (mu), density (rho)')
    #parser.add_argument("--toroidal", dest = "layer_number", help = "Plot toroidal modes for the solid shell given by LAYER_NUMBER (0 is outermost solid shell). Default is to plot spheroidal modes.", type = int)
    parser.add_argument('--include_brute_force', action = 'store_true', help = 'Also include brute-force kernels in the plots.')
    parser.add_argument('--units', choices = ['SI', 'standard'], default = 'standard', help = 'Choose between SI units or \'standard\' units (km for distance, mHz for freq., GPa for elastic moduli')
    args = parser.parse_args()

    # Rename input arguments.
    path_input = args.path_input
    mode_type = args.mode_type
    n = args.n
    l = args.l
    i_toroidal = args.layer_number
    #param = args.param
    include_brute_force = args.include_brute_force
    units = args.units
    #i_toroidal = args.layer_number

    if (i_toroidal is not None) or (mode_type == 'T'):

        assert mode_type == 'T', 'Must specify --i_toroidal for toroidal modes.'
        assert i_toroidal is not None, 'Must specify --i_toroidal for '\
                                            'toroidal modes.'

    # Read the input file.
    run_info = read_input_file(path_input)

    # Get model information for axis limits, scaling and horizontal lines.
    model = load_model(run_info['path_model'])
    # Convert to km.
    model['r'] = model['r']*1.0E-3 # Convert to km.
    # r_srf Radius of planet.
    # r_solid_fluid_boundary    List of radii of solid-fluid boundaries.
    r_srf = model['r'][-1]
    i_fluid, r_solid_fluid_boundary, _ = get_r_fluid_solid_boundary(
            model['r'], model['v_s'])
    h_lines = r_solid_fluid_boundary
    
    # Load kernel.
    # Units are mHz per GPa per km.
    r, K_ka, K_mu, K_rho = load_kernel(run_info, mode_type, n, l, units = units,
                                        i_toroidal = i_toroidal)
    r = r*1.0E-3 # Convert to km.

    # Multiply the kernel by a fixed constant to give a value close to 1.
    #param_scale_exponent_dict = {'ka' : 6, 'mu' : 6, 'rho' : 3}
    #param_scale_exponent_dict = {'ka' : 7, 'mu' : 7, 'rho' : 4}
    if units == 'standard':

        param_scale_exponent_dict = {'ka' : -7, 'mu' : -6, 'rho' : 0}

        # Get the label string for this parameter.
        param_unit_dict = {'freq' : 'mHz', 'dist' : 'km', 'ka' : 'GPa', 'mu' : 'GPa', 'rho' : '(g cm$^{-3}$)'}

    elif units == 'SI':

        param_scale_exponent_dict = {'ka' : -22, 'mu' : -21, 'rho' : 0}

        # Get the label string for this parameter.
        param_unit_dict = {'freq' : 'Hz', 'dist' : 'm', 'ka' : 'Pa', 'mu' : 'Pa', 'rho' : 'kg m$^{-3}$'}

    else:

        raise ValueError

    #
    param_symbol_dict = {'ka' : 'kappa', 'mu' : 'mu', 'rho' : 'rho'}

    font_size_label = 12

    #array_list = np.array([K_ka, K_mu, K_rho])
    array_list = np.array([K_ka, K_mu])
    #param_list = ['ka', 'mu', 'rho']
    param_list = ['ka', 'mu']
    #for i in range(3):
    n_params = len(param_list)
    fig, ax_arr = plt.subplots(1, n_params, figsize = (11.0, 4.25*n_params),
            sharey = True, constrained_layout = True)
    for i in range(n_params):
        
        param = param_list[i]

        param_exponent = param_scale_exponent_dict[param]
        param_scale = 10.0**param_exponent
        #
        param_unit = param_unit_dict[param]
        freq_unit = param_unit_dict['freq']
        dist_unit = param_unit_dict['dist']
        #
        param_symbol = param_symbol_dict[param]


        K_plt = array_list[i]
        K_plt = K_plt/param_scale
        #
        K_label = '$K_{{\{:}}}$ (10$^{{{:d}}}$ {:} {:}$^{{-1}}$ {:}$^{{-1}})$'\
                .format(param_symbol, param_exponent, freq_unit, param_unit,
                        dist_unit)

        ax = ax_arr[i]
        
        ax.plot(K_plt, r, label = 'Analytical')

        if include_brute_force:
            
            r_bf, K_bf = get_kernel_brute(path_input, mode_type, n, l, param,
                    units = units)
            K_bf = K_bf/param_scale

            ax.plot(K_bf, r_bf, label = 'Brute force')

            ax.legend()

        ax.set_xlabel(K_label, fontsize = font_size_label)
        ax.axvline(linestyle = ':', color = 'k')

        if h_lines is not None:

            for h_line in h_lines:
                
                ax.axhline(h_line, linestyle = ':', color = 'k')

    ax = ax_arr[0]
    ax.set_ylabel('Radius (km)', fontsize = font_size_label)
    ax.set_ylim([r[0], r[-1]])

    save = True
    if save:
        
        fig_name = 'kernel'
        _, _, _, dir_out = get_Ouroboros_out_dirs(run_info, mode_type)
        dir_plot = os.path.join(dir_out, 'plots')

        mkdir_if_not_exist(dir_plot)

        if mode_type in ['S', 'R']:

            fig_name = '{:}_{:>05d}_{:}_{:>05d}_{:1d}.png'.format(fig_name, n, mode_type, l, run_info['grav_switch'])

        else:

            fig_name = '{:}_{:>05d}_{:}{:1d}_{:>05d}.png'.format(fig_name, n, mode_type, i_toroidal, l)

        fig_path = os.path.join(dir_plot, fig_name)
        print('Saving figure to {:}'.format(fig_path))
        plt.savefig(fig_path, dpi = 300, bbox_inches = 'tight')

    plt.show()

    return

if __name__ == '__main__':

    main()
