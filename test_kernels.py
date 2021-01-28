import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

#from common import get_Ouroboros_out_dirs, get_r_fluid_solid_boundary, mkdir_if_not_exist, load_model, read_Ouroboros_input_file
from common import get_r_fluid_solid_boundary, load_model, read_Ouroboros_input_file
#from kernels import gravitational_acceleration, potential
#from post.read_output import load_eigenfreq_Ouroboros, load_eigenfunc_Ouroboros
from plot.plot_kernels_brute import get_kernel_brute
from post.read_output import load_eigenfreq_Ouroboros, load_eigenfunc_Ouroboros
from run_kernels import interp_n_parts
from kernels import G, gravitational_acceleration, kernel_spheroidal_rho_integral, potential, radial_derivative

def kernel_spheroidal_rho(r, U, V, l, omega, g_switch, rho = None, g = None, P = None):
    '''
    Eq. 9.15
    Spheroidal modes (W = 0)
    If rho, g and P are None, the kernel is calculated without the effect of
    gravity.
    '''

    # Calculate k and scale V by K.
    k2      = l*(l + 1.0)
    k       = np.sqrt(k2)

    # Factor of k.
    V       = k*V

    ## Factor of omega due to different normalisation.
    #U = U*omega

    # Radial derivative of potential.
    dPdr    = radial_derivative(r, P)
    
    # Calculate first term.
    a = -1.0*((omega*r)**2.0)*(U**2.0 + V**2.0)
    b =  8.0*np.pi*G*rho*((r*U)**2.0)
    c = 2.0*(r**2.0)*(U*dPdr + ((k*V*P)/r))
    # Apply appropriate limit as r -> 0
    c[0] = 0.0
    d = -2.0*g*r*U*(2.0*U - k*V)
    e = -8.0*np.pi*G*(r**2.0)*kernel_spheroidal_rho_integral(
                                    r, rho, U, V, k)

    
    out_arr = np.array([a, b, c, d, e])
    out_arr = out_arr/(2.0*omega)
    out_arr = out_arr/(2.0*np.pi) # Give sensitivity in Hz not rad per s.
    #K_rho = K_rho/(2.0*omega)

    #return a, b, c, d, e
    return out_arr

def main():

    # Parse the command-line arguments.
    parser = argparse.ArgumentParser()
    #
    parser.add_argument('path_to_input_file', help = 'File path (relative or absolute) to Ouroboros input file.')
    #parser.add_argument('mode_type', choices = ['R', 'S', 'T'], help = 'Mode type (radial, spheroidal, or toroidal).')
    parser.add_argument('n', type = int, help = 'Radial order.')
    parser.add_argument('l', type = int, help = 'Angular order.')
    #parser.add_argument('g_switch', type = int, choices = [0, 1, 2], help = 'Gravity switch.')
    args = parser.parse_args()

    mode_type = 'R'
    path_input = args.path_to_input_file
    n = args.n
    l = args.l
    #g_switch = args.g_switch
    
    # Load run information.
    run_info = read_Ouroboros_input_file(path_input)
    #run_info['g_switch'] = g_switch

    # Load the planetary model.
    model = load_model(run_info['path_model'])

    # Find the fluid-solid boundary points in the model.
    i_fluid_model, r_solid_fluid_boundary_model, i_fluid_solid_boundary_model =\
        get_r_fluid_solid_boundary(model['r'], model['v_s'])

    # Load frequency.
    f_mHz = load_eigenfreq_Ouroboros(run_info, mode_type, n_q = n, l_q = l) 
    f_Hz = f_mHz/1.0E3
    omega_rad_per_s = f_Hz*(2.0*np.pi)

    # Load eigenfunction.
    r, U = load_eigenfunc_Ouroboros(run_info, mode_type, n, l)
    V = np.zeros(U.shape)

    U = U*omega_rad_per_s

    ## Get planetary radius in km.
    #r_srf_km = r[-1]
    ## Calculate ratio for conversion to Mineos normalisation.
    #ratio = 1.0E-3*(r_srf_km**2.0)

    # Convert from km to m.
    r = r*1.0E3

    # Find indices of solid-fluid boundaries in the eigenfunction grid.
    i_fluid_solid_boundary = (np.where(np.diff(r) == 0.0))[0] + 1

    # Interpolate from the model grid to the output grid.
    rho  = interp_n_parts(r, model['r'], model['rho'], i_fluid_solid_boundary, i_fluid_solid_boundary_model)
    v_p  = interp_n_parts(r, model['r'], model['v_p'], i_fluid_solid_boundary, i_fluid_solid_boundary_model)
    v_s  = interp_n_parts(r, model['r'], model['v_s'], i_fluid_solid_boundary, i_fluid_solid_boundary_model)
    i_fluid = np.where(v_s == 0.0)[0]

    # Calculate the gravitational acceleration.
    # (SI units m/s2)
    g_model      = gravitational_acceleration(model['r'], model['rho'])

    # Interpolate the gravitational acceleration at the nodal points.
    g = np.interp(r, model['r'], g_model)

    ## Convert to Mineos normalisation.
    #U = U*ratio
    #V = V*ratio

    # Calculate gravitational potential.
    P = potential(r, U, V, l, rho)

    #a = -0.5*(omega_rad_per_s**3.0)*(r**2.0)*(U**2.0)

    # Change from sensitivity in
    # rad per s per (kg per m3) per m
    # to
    # Hz per (kg per m3) per m 
    #a = a/(2.0*np.pi)

    # Calculate terms in kernel.
    a, b, c, d, e = kernel_spheroidal_rho(r, U, V, l, omega_rad_per_s, run_info['g_switch'], rho = rho, g = g, P = P)

    #terms = np.array([a, b, c, d, e])

    ##a = a/(2.0*np.pi*omega_rad_per_s)
    #terms = terms/(2.0*np.pi*omega_rad_per_s)
    #
    ##a = a*(omega_rad_per_s**2.0)
    #terms = terms*(omega_rad_per_s**2.0)

    #scale = 1.04E9*np.pi
    ##a = a/scale
    #terms = terms/scale

    #a, b, c, d, e = terms

    r_bf, K_bf = get_kernel_brute(path_input, mode_type, n, l, 'rho')
    r_bf = r_bf*1.0E3

    a_max = np.max(np.abs(a))
    b_max = np.max(np.abs(b))
    c_max = np.max(np.abs(c))
    bf_max = np.nanmax(np.abs(K_bf))
    
    print('a', a_max)
    print('b', b_max)
    print('c', c_max)
    print(a_max/bf_max)

    fig = plt.figure()
    ax = plt.gca()

    ax.plot(a, r, label = 'a')
    ax.plot(b, r, label = 'b')
    ax.plot(c, r, label = 'c')
    ax.plot(d, r, label = 'd')
    ax.plot(e, r, label = 'e')
    #ax.plot(a + b, r, label = 'a + b')
    #ax.plot(a + c, r, label = 'a + c')
    #ax.plot(a + d, r, label = 'a + d')
    #ax.plot(a + e, r, label = 'a + e')
    #ax.plot(b + c, r, label = 'b + c')
    #ax.plot(a + b + c, r, label = 'a + b + c')
    #ax.plot(a + c + d, r, label = 'a + c + d')
    #ax.plot(a + c + d, r, label = 'a + c + d + e')
    #ax.plot(a + c + d + e, r, label = 'a + c + d + e')
    ax.plot(K_bf, r_bf, label = 'brute force')

    ax.legend()

    plt.show()

    ##aaa = np.nanmax(np.abs(K_bf))
    ##bbb = np.max(np.abs(a))
    ##print(aaa/bbb)

    #fig, ax_arr = plt.subplots(1, 9, sharey = True, figsize = (15.0, 10.0))

    #ax = ax_arr[0]

    #ax.plot(U, r, label = 'U')
    #ax.plot(V, r, label = 'V')
    #ax.plot(P, r, label = 'P')

    #ax.legend()

    #ax = ax_arr[1]

    #ax.plot(rho, r, label = 'rho')

    #ax.legend()

    #ax = ax_arr[2]
    #ax.plot(K_bf, r_bf*1.0E3, label = 'Brute force')

    #for ax_i, a_i, label_i in zip(ax_arr[2:7], [a, b, c, d, e], ['a', 'b', 'c', 'd', 'e']):

    #    ax_i.plot(a_i, r, label = label_i)

    #    ax_i.legend()

    #ax = ax_arr[7]

    #ax.plot(a + d, r, label = 'a + d')
    #ax.plot(K_bf, r_bf*1.0E3, label = 'Brute force')

    #ax = ax_arr[8]

    #ax.plot(a + b + c + d, r, label = 'a + b + c + d')
    #ax.plot(K_bf, r_bf*1.0E3, label = 'Brute force')
    #
    #ax.legend()

    ##plt.tight_layout()

    #plt.show()
    

    return

if __name__ == '__main__':

    main()
