import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from Ouroboros.plot.plot_kernels_brute import get_kernel_brute
from Ouroboros.common import get_Ouroboros_out_dirs, mkdir_if_not_exist, read_input_file

def main():

    # Parse the command-line arguments.
    parser = argparse.ArgumentParser()
    #
    parser.add_argument('path_input', help = 'File path (relative or absolute) to Ouroboros input file.')
    parser.add_argument('mode_type', choices = ['R', 'S', 'T'], help = 'Mode type (radial, spheroidal, or toroidal).')
    parser.add_argument('n', type = int, help = 'Radial order.')
    parser.add_argument('l', type = int, help = 'Angular order.')
    #parser.add_argument('param', choices = ['ka', 'mu', 'rho'], help = 'Plot sensitivity to bulk modulus (ka), shear modulus (mu), density (rho)')
    #parser.add_argument("--toroidal", dest = "layer_number", help = "Plot toroidal modes for the solid shell given by LAYER_NUMBER (0 is outermost solid shell). Default is to plot spheroidal modes.", type = int)
    parser.add_argument('--include_brute_force', action = 'store_true', help = 'Also include brute-force kernels in the plots.')
    args = parser.parse_args()

    # Rename input arguments.
    path_input = args.path_input
    mode_type = args.mode_type
    n = args.n
    l = args.l
    #param = args.param
    include_brute_force = args.include_brute_force
    #i_toroidal = args.layer_number

    # Read the input file.
    run_info = read_input_file(path_input)

    # Load the kernels for this mode.
    _, _, _, dir_out = get_Ouroboros_out_dirs(run_info, mode_type)
    dir_kernels = os.path.join(dir_out, 'kernels')
    #
    #out_arr = np.array([r, g, P, K_ka, K_mu, K_rho, K_alpha, K_beta, K_rhop])
    name_kernel_file = 'kernels_{:>05d}_{:>05d}.npy'.format(n, l)
    path_kernel = os.path.join(dir_kernels, name_kernel_file)
    print('Loading {:}'.format(path_kernel))
    kernel_arr = np.load(path_kernel)

    # Unpack the array.
    r, K_ka, K_mu = kernel_arr
    # Convert from m to km.
    r = r*1.0E-3

    # Multiply the kernel by a fixed constant to give a value close to 1.
    #param_scale_exponent_dict = {'ka' : 6, 'mu' : 6, 'rho' : 3}
    #param_scale_exponent_dict = {'ka' : 7, 'mu' : 7, 'rho' : 4}
    param_scale_exponent_dict = {'ka' : 0, 'mu' : 0, 'rho' : 0}

    # Get the label string for this parameter.
    param_unit_dict = {'ka' : 'GPa', 'mu' : 'GPa', 'rho' : '(g cm$^{-3}$)'}

    #
    param_symbol_dict = {'ka' : 'kappa', 'mu' : 'mu', 'rho' : 'rho'}

    font_size_label = 12

    fig, ax_arr = plt.subplots(1, 3, figsize = (11.0, 8.5), sharey = True)
    #array_list = np.array([K_ka, K_mu, K_rho])
    array_list = np.array([K_ka, K_mu])
    #param_list = ['ka', 'mu', 'rho']
    param_list = ['ka', 'mu']
    #for i in range(3):
    for i in range(2):
        
        param = param_list[i]

        param_exponent = param_scale_exponent_dict[param]
        param_scale = 10.0**param_exponent
        #
        param_unit = param_unit_dict[param]
        #
        param_symbol = param_symbol_dict[param]

        K_plt = array_list[i]
        K_plt = K_plt*param_scale
        #if param == 'rho':
        #    K_plt = K_plt*1.0E-12
        #scale = (4.0/np.pi)**2.0
        #scale = ((1.0E3*np.pi)**2.0)
        #scale = 1.04E7
        #K_plt = K_plt/scale
        #K_plt = K_plt/scale
        #K_plt = K_plt/(((np.pi)**2.0)/6.0)
        #
        K_label = '$K_{{\{:}}}$ (10$^{{{:d}}}$ mHz {:}$^{{-1}}$ km$^{{-1}})$'.format(param_symbol, param_exponent, param_unit)

        ax = ax_arr[i]
        
        ax.plot(K_plt, r, label = 'Analytical')

        if include_brute_force:
            
            r_bf, K_bf = get_kernel_brute(path_input, mode_type, n, l, param)
            K_bf = K_bf*param_scale

            ax.plot(K_bf, r_bf, label = 'Brute force')

            ax.legend()

        ax.set_xlabel(K_label, fontsize = font_size_label)
        ax.axvline(linestyle = ':', color = 'k')

        #print(param, np.max(np.abs(K_plt))/np.nanmax(np.abs(K_bf)))

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

    plt.tight_layout()
    plt.show()

    return

if __name__ == '__main__':

    main()
