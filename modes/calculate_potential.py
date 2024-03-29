'''
Calculating the gravitational potential.
'''

import argparse
import os

import numpy as np

from Ouroboros.common import (get_Ouroboros_out_dirs, 
                        get_r_fluid_solid_boundary, interp_n_parts,
                        load_eigenfreq_Ouroboros,
                        load_eigenfunc_Ouroboros, mkdir_if_not_exist,
                        load_model, read_Ouroboros_input_file)
from Ouroboros.constants import G

def potential_all_modes(run_info, mode_type, j_skip = None):
    '''
    Loop over all modes and calculate gravitational potential.
    '''

    # Set normalisation of eigenfunctions (and therefore potential).
    # Use Dahlen and Tromp normalisation function ('DT') so that we can
    # use expressions from Dahlen and Tromp without modification.
    # Use Ouroboros units so output is consistent with eigenfunctions.
    normalisation_args = {'norm_func' : 'DT', 'units' : 'ouroboros'}

    # Get the density profile.
    # r_rho km
    # rho   kg/m3
    r_rho, rho = get_rho(run_info, mode_type)

    # Get output directory.
    _, _, _, dir_output = \
        get_Ouroboros_out_dirs(run_info, mode_type)

    # Load mode information.
    mode_info = load_eigenfreq_Ouroboros(run_info, mode_type)
    n = mode_info['n']
    l = mode_info['l']
    f = mode_info['f']

    # Loop over modes.
    num_modes = len(n)
    for i in range(num_modes):

        if (j_skip is None) or not (i in j_skip):

            # To convert from Ouroboros normalisation to Dahlen and Tromp
            # normalisation we must provide the mode frequency in
            # rad per s.
            f_rad_per_s = f[i]*1.0E-3*2.0*np.pi
            if normalisation_args['norm_func'] == 'DT':
                normalisation_args['omega'] = f_rad_per_s

            # Calculate potential for this mode
            potential_wrapper(run_info, dir_output, mode_type, n[i], l[i], rho,
                    **normalisation_args)

    return

def potential_wrapper(run_info, dir_output, mode_type, n, l, rho, **normalisation_args):
    '''
    Calculate potential for a single mode.
    '''
    
    # Load eigenfunction.
    eigfunc_dict = load_eigenfunc_Ouroboros(run_info, mode_type, n, l,
                    **normalisation_args)
    r = eigfunc_dict['r']*1.0E-3 # Units of km.
    U = eigfunc_dict['U']

    if mode_type == 'S':

        V = eigfunc_dict['V']

    elif mode_type == 'R':

        V = np.zeros(U.shape)

    else:

        raise ValueError

    # Calculate potential.
    P = potential(r, U, V, l, rho)
    # Convert to Mineos normalisation function for consistency with other
    # outputs.
    P = P/normalisation_args['omega']

    # Load eigenfunction again with Ouroboros units.
    eigfunc_dict = load_eigenfunc_Ouroboros(run_info, mode_type, n, l,
                        norm_func = 'mineos', units = 'ouroboros',
                        omega = normalisation_args['omega'])

    # Save.
    file_eigenfunc = '{:>05d}_{:>05d}.npy'.format(n, l)
    path_out = os.path.join(dir_output, 'eigenfunctions', file_eigenfunc)

    if mode_type == 'S':

        out_arr = np.array([eigfunc_dict['r']*1.0E-3,
                            eigfunc_dict['U'],  eigfunc_dict['Up'],
                            eigfunc_dict['V'], eigfunc_dict['Vp'],
                            P, eigfunc_dict['Pp']])

    elif mode_type == 'R':

        out_arr = np.array([eigfunc_dict['r']*1.0E-3,
                            eigfunc_dict['U'], 
                            eigfunc_dict['Up'],
                            P, eigfunc_dict['Pp']])

    else:

        raise ValueError

    print("Saving to {:}".format(path_out))
    np.save(path_out, out_arr)

    return

def potential(r, U, V, l, rho):
    '''
    Dahlen and Tromp, Equation 8.55.
    '''

    nk  = len(r)
    P   = np.zeros(nk)
    # Evaluate potential at each radius.
    for i in range(nk):
        
        P[i] = P_of_r(i, r, U, V, l, rho)

    return P

def P_of_r(i, r, U, V, l, rho):
    '''
    Dahlen and Tromp, equation 8.55
    '''
    
    nk = len(r)
    
    # The integral is split into two parts.
    # If r == 0, only the upper part is non-zero.
    if (i == 0):

        I = potential_upper_integral(r[i], r, U, V, l, rho, contains_r0 = True)

    # If r == r_max, only the lower part is non-zero.
    elif (i == (nk - 1)):

        I = potential_lower_integral(r[i], r, U, V, l, rho, contains_r0 = True)

    # If r_max > r > 0, both the upper and lower parts of the integral are
    # non-zero.
    else:

        Il = potential_lower_integral(
            r[i], r[:(i + 1)], U[:(i + 1)], V[:(i + 1)], l, rho[:(i + 1)],
            contains_r0 = True)
        Iu = potential_upper_integral(
            r[i], r[i:], U[i:], V[i:], l, rho[i:])

        I  = Il + Iu

    # Multiply by pre-factor.
    pref = (-4.0*np.pi*G)/((2.0*l) + 1.0)
    P = pref*I
    
    return P

def potential_lower_integral(ri, r, U, V, l, rho, contains_r0 = False):
    '''
    First term in brackets on RHS of eq. 8.55.
    Note the factor of r^(-l - 1) has been moved inside the integral,
    because (a/b)^x is more accurate than (a^x)*(b^-x) if a/b ~ 1
    and x is large.
    '''
    
    k2  = l*(l + 1.0)
    k   = np.sqrt(k2)

    f   = rho*(l*U + k*V)*((r/ri)**(l + 1.0))
    
    # Apply limiting value at centre of the Earth.
    if contains_r0:
        
        f[0] = 0.0
    
    I = np.trapz(f, x = r)

    return I

def potential_upper_integral(ri, r, U, V, l, rho, contains_r0 = False):
    '''
    Second term in brackets on RHS of eq. 8.55.
    Note the factor of r^l has been moved inside the integral, because
    (a/b)^x is more accurate than (a^x)*(b^-x) if a/b ~ 1 and x is large.
    '''
    
    k2  = l*(l + 1.0)
    k   = np.sqrt(k2)
    f   = rho*(-1.0*(l + 1.0)*U + k*V)*((ri/r)**l)
    
    # Apply limiting value at centre of the Earth.
    if contains_r0:
        
        f[0] = 0.0

    I = np.trapz(f, x = r)
    
    return I

def get_rho(run_info, mode_type):
    '''
    Load a density model, interpolated at the eigenfunction radial points.
    '''

    # Load the planetary model.
    model = load_model(run_info['path_model'])

    # Load the radial coordinate.
    mode_info = load_eigenfreq_Ouroboros(run_info, mode_type)
    n = mode_info['n']
    l = mode_info['l']
    f = mode_info['f']
    dummy_value = 1.0
    eigfunc_dict = load_eigenfunc_Ouroboros(run_info, mode_type, n[0], l[0], omega = dummy_value)
    r = eigfunc_dict['r']*1.0E-3

    # Find the fluid-solid boundary points in the model.
    i_fluid_model, r_solid_fluid_boundary_model, i_fluid_solid_boundary_model =\
        get_r_fluid_solid_boundary(model['r']*1.0E-3, model['v_s'])

    # Find indices of solid-fluid boundaries in the eigenfunction grid.
    i_fluid_solid_boundary = (np.where(np.diff(r) == 0.0))[0] + 1

    # Interpolate from the model grid to the output grid.
    rho  = interp_n_parts(r, model['r']*1.0E-3, model['rho'],
                i_fluid_solid_boundary, i_fluid_solid_boundary_model)

    return r, rho

def main():

    # Read input arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_input_file", help = "File path (relative or absolute) to Ouroboros input file.")
    args = parser.parse_args()

    # Rename input arguments.
    path_input = args.path_to_input_file

    # Read input file.
    run_info = read_Ouroboros_input_file(path_input)
    
    # Loop over modes.
    for mode_type in run_info['mode_types']:

        potential_all_modes(run_info, mode_type)

    return

if __name__ == '__main__':

    main()
