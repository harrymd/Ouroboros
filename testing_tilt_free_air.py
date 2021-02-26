import argparse

import matplotlib.pyplot as plt
import numpy as np

from Ouroboros.common import load_eigenfunc_Mineos, read_Mineos_input_file
from Ouroboros.summation.run_summation import seismometer_response_correction

def main():

    # Parse input arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("path_mode_input", help = "File path (relative or absolute) to Mineos mode input file.")
    input_args = parser.parse_args()
    path_mode_input = input_args.path_mode_input

    # Read Mineos input file.
    run_info = read_Mineos_input_file(path_mode_input)

    # Load the reference data.
    # (Dahlen and Tromp, 1998, table 10.1.)
    path_test_data = '/Users/hrmd_work/Documents/research/refs/dahlen_1998/dahlen_1998_tab_10.1.csv'
    test_data = np.loadtxt(path_test_data, delimiter = ',', skiprows = 1)

    # Unpack reference data.
    n = test_data[:, 0].astype(np.int)
    l = test_data[:, 1].astype(np.int)
    f_mHz = test_data[:, 2]
    U_frac      = test_data[:, 3]
    U_free_frac = test_data[:, 4] 
    U_potl_frac = test_data[:, 5] 
    V_frac      = test_data[:, 6]
    V_tilt_frac = test_data[:, 7]
    V_potl_frac = test_data[:, 8]

    # Convert to radians per second.
    f_rad_per_s = 1.0E-3*2.0*np.pi*f_mHz

    # Define constants (SI units).
    # r_planet  Planetary radius.
    # g         Surface accelerate due to gravity.

    r_planet = 6371.0E3
    g        = 9.81 # Not quite true for PREM with no ocean, but close enough.

    # Prepare output arrays.
    num_modes = len(n)
    U_frac_compare      = np.zeros(num_modes) 
    U_free_frac_compare = np.zeros(num_modes)  
    U_potl_frac_compare = np.zeros(num_modes)  
    V_frac_compare      = np.zeros(num_modes) 
    V_tilt_frac_compare = np.zeros(num_modes) 
    V_potl_frac_compare = np.zeros(num_modes) 

    # Define other parameters.
    mode_type = 'S'
    norm_args = {'norm_func' : 'DT', 'units' : 'SI'}

    # Calculate the corrections.
    for i in range(num_modes):

        norm_args['omega'] = f_rad_per_s[i]

        try:

            # Load eigenfunction.
            r, U, Up, V, Vp, P, Pp = \
                load_eigenfunc_Mineos(run_info, mode_type, n[i], l[i],
                    **norm_args)

            # Get eigenfunction at surface.
            U = U[0]
            V = V[0]
            P = P[0]

            # Calculate response.
            U_free, U_potl, V_tilt, V_potl = seismometer_response_correction(
                    l[i], f_rad_per_s[i], r_planet, g, U, P)

            # Take ratios.
            U_star = U + U_free + U_potl
            V_star = V + V_tilt + V_potl 
            #
            U_frac_compare[i]       = U/U_star
            U_free_frac_compare[i]  = U_free/U_star
            U_potl_frac_compare[i]  = U_potl/U_star
            #
            V_frac_compare[i]       = V/V_star
            V_tilt_frac_compare[i]  = V_tilt/V_star
            V_potl_frac_compare[i]  = V_potl/V_star

        except OSError:

            U_frac_compare[i]       = np.nan 
            U_free_frac_compare[i]  = np.nan 
            U_potl_frac_compare[i]  = np.nan 
            #
            V_frac_compare[i]       = np.nan 
            V_tilt_frac_compare[i]  = np.nan 
            V_potl_frac_compare[i]  = np.nan 

    frac_list = [U_frac, U_free_frac, U_potl_frac, V_frac, V_tilt_frac, V_potl_frac]
    frac_compare_list = [U_frac_compare, U_free_frac_compare, U_potl_frac_compare,
                        V_frac_compare, V_tilt_frac_compare, V_potl_frac_compare]
    label_list = ['$U/U_{\\ast}$', '$U_{free}/U_{\\ast}$', '$U_{pot}/U_{\\ast}$',
            '$V/V_{\\ast}$', '$V_{tilt}/V_{\\ast}$', '$V_{pot}/V_{\\ast}$']

    scatter_kwargs = {'s': 10, 'c' : 'k', 'alpha' : 0.5}
    fig, ax_arr = plt.subplots(3, 2, figsize = (5.5, 8.5), constrained_layout = True)
        
    ax_list = [ ax_arr[0][0], ax_arr[1][0], ax_arr[2][0],
                ax_arr[0][1], ax_arr[1][1], ax_arr[2][1]]

    for i in range(6):

        ax = ax_list[i]
                    
        ax.scatter(frac_list[i], frac_compare_list[i], **scatter_kwargs)

        ax.text(0.05, 0.95, label_list[i], ha = 'left', va = 'top',
                transform = ax.transAxes)
    
    for ax_list in ax_arr:

        for ax in ax_list:

            ax.set_aspect(1.0)    
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            ax.plot([-3.0, 3.0], [-3.0, 3.0], zorder = 0)

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

    ax = ax_arr[-1][0]
    ax.set_xlabel('Textbook')
    #ax.set_ylabel('Ouroboros')
    ax.set_ylabel('New code')

    plt.savefig('corrections.png', dpi = 300)

    plt.show()

     

    return

if __name__ == '__main__':

    main()
