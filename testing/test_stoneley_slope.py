from almanac.seismology import solve_stoneley_func_solid_fluid, solve_rayleigh_func_layer_over_halfspace
#from almanac.seismology import rayleigh_func_layer_over_halfspace

import matplotlib.pyplot as plt
import numpy as np

from Ouroboros.common import load_eigenfreq, read_input_file

def test_rayleigh_func_layer_over_halfspace():

    h = 30.0

    alpha_1 = 6.0
    beta_1  = 3.0
    rho_1   = 2.8

    alpha_2 = 8.0
    beta_2  = 4.0
    rho_2   = 3.3

    T       = 50.0
    f       = 1.0/T
    omega   = 2.0*np.pi*f
    
    c       = 3.5
    k       = omega/c

    #det = rayleigh_func_layer_over_halfspace(h, alpha_1, beta_1, rho_1,
    #            alpha_2, beta_2, rho_2, omega, k)
    
    dummy = solve_rayleigh_func_layer_over_halfspace(h, alpha_1, beta_1, rho_1,
                alpha_2, beta_2, rho_2, omega)



    return

def main():

    #rho_s =  5566.46  
    #vp_s  = 13716.62 
    #vs_s  =  7264.65 

    #rho_f =  9903.44 
    #vp_f  =  8064.79 
    
    # PREM, attenuation-corrected to 3 mHz.
    rho_s =  5566.46  
    vp_s  = 13685.91 
    vs_s  =  7221.47 

    rho_f =  9903.44 
    vp_f  =  8064.53

    c = solve_stoneley_func_solid_fluid(rho_s, vp_s, vs_s, rho_f, vp_f)

    # Get mode information.
    mode_type = 'S'
    path_input = '../../input/mineos/input_prem_noocean.txt'
    run_info = read_input_file(path_input)
    mode_info = load_eigenfreq(run_info, mode_type)
    n = mode_info['n']
    l = mode_info['l']
    f = mode_info['f']
    n_list = sorted(list(set(n)))

    #f_mHz_max = 7.0
    #f_Hz_max = 1.0E-3*f_mHz_max
    #omega_max = 2.0*np.pi*f_Hz_max

    l_max = np.max(l) 
    k_max = np.sqrt(l_max*(l_max + 1.0))

    omega_max = c*k_max/3480.0E3
    f_Hz_max = omega_max/(2.0*np.pi)
    f_mHz_max = 1.0E3*f_Hz_max

    fig = plt.figure(figsize = (8.0, 6.0), constrained_layout = True)
    ax  = plt.gca()

    color = 'k'
    c_scatter = color
    alpha = 1.0 

    for ni in n_list: 

        i = (n == ni)
        
        ax.plot(l[i], f[i], ls = '-', color = color, lw = 1, alpha = alpha)

    ax.scatter(l, f, s = 3, c = c_scatter, alpha = alpha, zorder = 10)

    ax.plot([0.0, l_max], [0.0, f_mHz_max], c = 'r')
    
    f_lims = [0.0, 7.0]
    l_lims = [-0.5, 30.0]
    ax.set_xlim(l_lims)
    ax.set_ylim(f_lims)
    
    font_size_label = 13
    ax.set_xlabel('Angular order, $\ell$', fontsize = font_size_label)
    ax.set_ylabel('Frequency (mHz)', fontsize = font_size_label)

    plt.show()

    return

if __name__ == '__main__':

    #main()
    test_rayleigh_func_layer_over_halfspace()
