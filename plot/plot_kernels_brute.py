import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from common import read_Ouroboros_input_file, get_Ouroboros_out_dirs

def get_kernel_brute(path_input, mode_type, n, l, param, units = 'standard'):

    # Found output files.
    run_info = read_Ouroboros_input_file(path_input)
    #run_info['dir_output'] = os.path.join(run_info['dir_output'], 'cluster')
    _, _, _, dir_type = get_Ouroboros_out_dirs(run_info, mode_type)
    dir_kernels_brute = os.path.join(dir_type, 'kernels_brute')
    #
    path_VX = os.path.join(dir_kernels_brute, 'VX.npy')
    path_param = os.path.join(dir_kernels_brute, 'd_{:}.npy'.format(param))
    path_omega_ptb = os.path.join(dir_kernels_brute, 'omega_ptb_{:}_{:>05d}_{:>05d}.txt'.format(param, n, l)) 
    
    # Load output files.
    # vertices  Edges of computational grid.
    # d_param   Change of parameter applied to calculate perturbation.
    #           Units of GPa for ka and mu, and g/cm3 for rho.
    # omega_ref Reference frequency (mHz).
    # omega_ptb Frequency when perturbation is applied to each layer (mHz).
    vertices = np.load(path_VX)
    vertices = vertices*1.0E3 # Convert from Mm to km.
    #
    d_param = np.load(path_param) # Units of GPa for ka and mu.
    # 
    with open(path_omega_ptb, 'r') as in_id:

        omega_ref = float(in_id.readline().split()[-1])

    omega_ptb = np.loadtxt(path_omega_ptb)

    # Calculate sensitivity.
    # d_omega   Change in frequency when perturbation is applied to each layer
    #           (mHz).
    # d_r       Thickness of each layer (km).
    d_omega = omega_ptb - omega_ref
    d_r = np.diff(vertices)

    #fig = plt.figure()
    #ax = plt.gca()
    #ax.plot(d_r)

    #plt.show()

    # Calculate kernel.
    # Units of mHz per GPa per km for kappa and mu, mHz per 
    K = (d_omega/d_param)/d_r

    # Convert to requested units.
    # By default the units are mHz per GPa per km and do not need to be
    # adjusted.
    if units == 'standard':

        pass

    # Otherwise, convert to SI units.
    elif units == 'SI':

        # Convert from (mHz 1/GPa 1/km) to (Hz 1/Pa 1/m).
        Hz_to_mHz   = 1.0E3
        Pa_to_GPa   = 1.0E-9
        m_to_km     = 1.0E-3

        K = K*Pa_to_GPa*m_to_km/Hz_to_mHz

    else:

        raise ValueError

    # Thresholding for numerical noise on thin layers.
    d_r_thresh = 1.0
    K[d_r < d_r_thresh] = np.nan

    # Mask NaN values in fluid regions.
    if param == 'mu':

        K[d_param == 0.0] = 0.0

    # Make arrays for plotting, correctly indicating step-wise perturbation.
    r_plt = np.repeat(vertices, 2)
    r_plt = r_plt[1:-1]
    K_plt = np.repeat(K, 2)

    return r_plt, K_plt

def main():

    # Parse the command-line arguments.
    parser = argparse.ArgumentParser()
    #
    parser.add_argument('path_to_input_file', help = 'File path (relative or absolute) to Ouroboros input file.')
    parser.add_argument('mode_type', choices = ['R', 'S', 'T'], help = 'Mode type (radial, spheroidal, or toroidal).')
    parser.add_argument('n', type = int, help = 'Radial order.')
    parser.add_argument('l', type = int, help = 'Angular order.')
    parser.add_argument('param', choices = ['ka', 'mu', 'rho'], help = 'Plot sensitivity to bulk modulus (ka), shear modulus (mu), density (rho)')
    #parser.add_argument("--toroidal", dest = "layer_number", help = "Plot toroidal modes for the solid shell given by LAYER_NUMBER (0 is outermost solid shell). Default is to plot spheroidal modes.", type = int)
    args = parser.parse_args()

    # Rename input arguments.
    path_input = args.path_to_input_file
    mode_type = args.mode_type
    n = args.n
    l = args.l
    param = args.param
    #i_toroidal = args.layer_number

    # Check input arguments.
    if mode_type == 'R':

        assert l == 0, 'Angular order must be 0 for radial modes ({:>5d} was given)'.format(l)

    # Get the kernel from the output files.
    r_plt, K_plt = get_kernel_brute(path_input, mode_type, n, l, param)

    # Multiply the kernel by a fixed constant to give a value close to 1.
    param_scale_exponent_dict = {'ka' : 6, 'mu' : 6, 'rho' : 3}
    param_exponent = param_scale_exponent_dict[param]
    param_scale = 10.0**param_exponent
    #
    K_plt = K_plt*param_scale

    # Get the label string for this parameter.
    param_unit_dict = {'ka' : 'GPa', 'mu' : 'GPa', 'rho' : '(g cm$^{-3}$)'}
    param_unit = param_unit_dict[param]
    #
    param_symbol_dict = {'ka' : 'kappa', 'mu' : 'mu', 'rho' : 'rho'}
    param_symbol = param_symbol_dict[param]
    #
    K_label = '$K_{{\{:}}}$ (10$^{{{:d}}}$ mHz {:}$^{{-1}}$ km$^{{-1}})$'.format(param_symbol, param_exponent, param_unit)

    # Make the plot.
    fig = plt.figure()
    ax  = plt.gca()

    ax.plot(K_plt, r_plt)

    ax.set_ylim([np.min(r_plt), np.max(r_plt)])
    ax.axvline(linestyle = ':', color = 'k')
    
    font_size_label = 12
    ax.set_xlabel(K_label, fontsize = font_size_label)
    ax.set_ylabel('Radius (km)', fontsize = font_size_label)

    plt.tight_layout()

    plt.show()

    return

if __name__ == '__main__':

    main()
