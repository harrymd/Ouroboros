import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from Ouroboros.plot.plot_kernels_brute import get_kernel_brute
from Ouroboros.common import get_Ouroboros_out_dirs, load_eigenfreq, load_kernel, read_input_file

def main():

    # Parse the command-line arguments.
    parser = argparse.ArgumentParser()
    #
    parser.add_argument('path_input', help = 'File path (relative or absolute) to Ouroboros input file.')
    #
    args = parser.parse_args()
    #
    path_input = args.path_input

    # Read the input file.
    mode_type = 'S'
    run_info = read_input_file(path_input)

    _, _, _, dir_out = get_Ouroboros_out_dirs(run_info, mode_type)
    dir_kernels = os.path.join(dir_out, 'kernels')

    mode_info = load_eigenfreq(run_info, mode_type)
    n = mode_info['n']
    l = mode_info['l']
    f_mHz = mode_info['f']
    f_Hz = f_mHz*1.0E-3
    f_rad_per_s = f_Hz*2.0*np.pi

    num_modes = len(n)
    K_ka_max = np.zeros(num_modes)
    K_mu_max = np.zeros(num_modes)
    K_bf_ka_max = np.zeros(num_modes)
    K_bf_mu_max = np.zeros(num_modes)

    for i in range(num_modes):

        r, K_ka, K_mu = load_kernel(run_info, mode_type, n[i], l[i])

        #name_kernel_file = 'kernels_{:>05d}_{:>05d}.npy'.format(n[i], l[i])
        #path_kernel = os.path.join(dir_kernels, name_kernel_file)
        #kernel_arr = np.load(path_kernel)

        ## Unpack the array.
        #r, K_ka, K_mu = kernel_arr

        ## Convert from (Hz 1/Pa 1/m) to (mHz 1/GPa 1/km).
        #Hz_to_mHz   = 1.0E3
        #Pa_to_GPa   = 1.0E-9
        #m_to_km     = 1.0E-3
        ##
        #scale = Hz_to_mHz/(Pa_to_GPa*m_to_km)
        #K_ka = K_ka*scale
        #K_mu = K_mu*scale

        ##scale = f_mHz[i]
        ##scale = f_rad_per_s[i]**2.0
        #scale = (f_mHz[i]**2.0)
        #K_ka = K_ka*scale
        #K_mu = K_mu*scale
        
        _, K_bf_ka = get_kernel_brute(path_input, mode_type, n[i], l[i], 'ka')
        _, K_bf_mu = get_kernel_brute(path_input, mode_type, n[i], l[i], 'mu')

        K_ka_max[i] = np.nanmax(np.abs(K_ka))
        K_mu_max[i] = np.nanmax(np.abs(K_mu))

        K_bf_ka_max[i] = np.nanmax(np.abs(K_bf_ka))
        K_bf_mu_max[i] = np.nanmax(np.abs(K_bf_mu))

    ratio_ka = K_bf_ka_max/K_ka_max
    ratio_mu = K_bf_mu_max/K_mu_max

    plot_hist = True
    if plot_hist:

        fig, ax_arr = plt.subplots(2, 1)
        
        bins = np.linspace(0.0, 2.0, num = 100)
        #bins = 100
        ax = ax_arr[0]
        ax.hist(ratio_mu, bins)
        ax.set_title('mu')
    
        ax = ax_arr[1]
        ax.hist(ratio_ka, bins)
        ax.set_title('ka')
    
        plt.show()

    #fig, ax_arr = plt.subplots(2, 1)
    #
    #ax = ax_arr[0]

    #ax.scatter(K_bf_ka_max, K_ka_max)

    #ax = ax_arr[1]

    #ax.scatter(K_bf_mu_max, K_mu_max)

    #plt.show()
    
    plot_scatter = False
    if plot_scatter:

        fig, ax_arr = plt.subplots(2, 1, sharex = True, figsize = (11.0, 8.5))

        ax = ax_arr[0]
        ax.scatter(f_mHz, ratio_ka)

        ax = ax_arr[1]
        ax.scatter(f_mHz, ratio_mu)
        
        plt.show()

    return

if __name__ == '__main__':

    main()
