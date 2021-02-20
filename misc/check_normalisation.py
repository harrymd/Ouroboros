import argparse

import numpy as np

from common import (    get_r_fluid_solid_boundary,
                        load_eigenfreq_Ouroboros, load_eigenfunc_Ouroboros,
                        load_eigenfreq_Mineos, load_eigenfunc_Mineos,
                        load_model,
                        read_Ouroboros_input_file,
                        read_Mineos_input_file)
from kernels.run_kernels import interp_n_parts

def main():

    # Read input arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_input_file", help = "File path (relative or absolute) to Ouroboros input file.")
    parser.add_argument("mode_type", choices = ['R', 'S', 'T'], help = 'Mode type (radial, spheroidal or toroidal).')
    parser.add_argument("n", type = int, help = "Check normalisation of mode with radial order n.")
    parser.add_argument("l", type = int, help = "Check normalisation of mode with angular order l (must be 0 for radial modes).")
    parser.add_argument("--i_toroidal", dest = "layer_number", help = "Check normalisation of a toroidal mode from the solid shell given by LAYER_NUMBER (0 is outermost solid shell).", type = int)
    parser.add_argument("--use_mineos", action = "store_true", help = "Check normalisation of a Mineos mode (default: Ouroboros).")
    parser.add_argument("--r_scale", type = float, help = 'Multiply radius by this quantity before calculating normalisation.')
    parser.add_argument("--eig_scale", type = float, help = 'Multiply eigenfunction(s) by this quantity before calculating normalisation.')
    parser.add_argument("--freq_scale", type = float, help = 'Multiply frequency (mHz) by this quantity before calculating normalisation.')
    parser.add_argument("--rho_scale", type = float, help = 'Multiply density by this quantity before calculating normalisation.')
    args = parser.parse_args()

    # Rename input arguments.
    path_input  = args.path_to_input_file
    mode_type   = args.mode_type
    n           = args.n
    l           = args.l
    i_toroidal  = args.layer_number
    use_mineos  = args.use_mineos
    r_scale     = args.r_scale
    eig_scale   = args.eig_scale
    freq_scale  = args.freq_scale
    rho_scale   = args.rho_scale

    if use_mineos:

        assert i_toroidal is None, 'The i_toroidal flag is not used with Mineos, try using mode_type == \'T\' for mantle toroidal modes and mode_type == \'I\' for inner-core toroidal modes.'
    
    # Read the input file and command-line arguments.
    if use_mineos:

        run_info = read_Mineos_input_file(path_input)

    else:

        run_info = read_Ouroboros_input_file(path_input)

    # Load the planetary model.
    model = load_model(run_info['path_model'])

    if rho_scale is not None:

        model['rho'] = model['rho']*rho_scale

    # Load frequency information.
    if use_mineos:
    
        f_mHz = load_eigenfreq_Mineos(run_info, mode_type, n_q = n, l_q = l)
        
    else:
        
        f_mHz = load_eigenfreq_Ouroboros(run_info, mode_type, n_q = n, l_q = l, i_toroidal = i_toroidal)

    if freq_scale is not None:

        print('Multiplying frequency by: {:>7.2e}'.format(freq_scale))
        f_mHz = f_mHz*freq_scale

    # Convert frequency to radians per second.
    f_Hz = f_mHz*1.0E-3
    f_rad_per_s = f_Hz*(2.0*np.pi)

    # Load eigenfunction(s).
    if use_mineos:

        if mode_type == 'R':
            
            print('Loading raw values of r and U from Ouroboros output files.')
            r, U, _ = load_eigenfunc_Mineos(run_info, mode_type, n, l)
            r = r[::-1]
            U = U[::-1]

        elif mode_type == 'S': 

            r, U, _, V, _, _, _ = load_eigenfunc_Mineos(run_info, mode_type, n, l)
            r = r[::-1]
            U = U[::-1]
            V = V[::-1]

        elif mode_type in ['T', 'I']:

            r, W, _ = load_eigenfunc_Mineos(run_info, mode_type, n, l)
            r = r[::-1]
            W = W[::-1]

    else:

        if mode_type == 'R':
            
            print('Loading raw values of r and U from Ouroboros output files.')
            r, U = load_eigenfunc_Ouroboros(run_info, mode_type, n, l)
            U[0] = 0.0 # Value of U at planet core appears to be buggy for R modes.

        elif mode_type == 'S': 

            r, U, V = load_eigenfunc_Ouroboros(run_info, mode_type, n, l)

        elif mode_type == 'T':

            r, W = load_eigenfunc_Ouroboros(run_info, mode_type, n, l, i_toroidal = i_toroidal)

    if mode_type in ['S', 'T']:

        k = np.sqrt(l*(l + 1.0))

    #
    print('Converting radius to m')
    if use_mineos:

        pass

    else:

        r = r*1.0E3
    
    # Apply scaling.
    if r_scale is not None:
        
        print('Multiplying radius by {:>10.3e}'.format(r_scale))
        r = r*r_scale
        model['r'] = model['r']*r_scale

    if eig_scale is not None:
        
        print('Multiplying eigenfunction(s) by {:>10.3e}'.format(eig_scale))
        if mode_type == 'R':

            #U = U/np.sqrt(1.0E9)
            U = U*eig_scale

        elif mode_type == 'S':

            U = U*eig_scale
            V = V*eig_scale

        elif mode_type == 'T':

            W  = W*eig_scale
    
    if mode_type == 'R':

        max_abs_eigfunc = np.max(np.abs(U))

    elif mode_type == 'S':

        max_abs_U = np.max(np.abs(U))
        max_abs_V = np.max(np.abs(V))
        max_abs_eigfunc = np.max([max_abs_U, max_abs_V])

    elif mode_type == 'T':

        max_abs_eigfunc = np.max(np.abs(W))

    if mode_type in ['R', 'S']:

        # Find indices of solid-fluid boundaries in the eigenfunction grid.
        i_fluid_solid_boundary = (np.where(np.diff(r) == 0.0))[0] + 1

        if use_mineos:

            print("Warning: Assuming inner two discontinuities are CMB and ICB.")
            i_fluid_solid_boundary = i_fluid_solid_boundary[0:2]

        # Find the fluid-solid boundary points in the model.
        i_fluid_model, r_solid_fluid_boundary_model, i_fluid_solid_boundary_model =\
            get_r_fluid_solid_boundary(model['r'], model['v_s'])

        # Interpolate from the model grid to the output grid.
        #rho  = interp_n_parts(r*1.0E3, model['r'], model['rho'], i_fluid_solid_boundary, i_fluid_solid_boundary_model)

        rho  = interp_n_parts(r, model['r'], model['rho'], i_fluid_solid_boundary, i_fluid_solid_boundary_model)

    else:

        rho = np.interp(r, model['r'], model['rho'])

    print(r[0], r[-1])

    print('Calculating normalisation')
    print('Mode {:>5d} {:>1} {:>5d}'.format(n, mode_type, l))
    print('Freq.: {:>9.5f}'.format(f_mHz))
    print('omega: {:>9.5f}'.format(f_rad_per_s))
    print('Max. density: {:>10.3f}'.format(np.max(rho)))
    print('Max. radius (from model file):   {:>10.3f}'.format(np.max(model['r'])))
    print('Max. radius (from eigfunc file): {:>10.3f}'.format(np.max(r)))
    print('Max. abs. eigfunc: {:>.3e}'.format(max_abs_eigfunc))
    
    # Integrate (each section integrated separately).
    # Define integration function.
    if mode_type == 'R':

        print('I             = integral( rho * (U^2) * (r^2) )')
        function = rho*(U**2.0)*(r**2.0)

    elif mode_type == 'S':

        print('I             = integral( rho * (U^2 + (k*V)^2) * (r^2) )')
        function = rho*(U**2.0 + (k*V)**2.0)*(r**2.0)

    elif mode_type == 'T':

        print('I             = integral( rho * ((k*W)^2) * (r^2) )')
        function = rho*((k*W)**2.0)*(r**2.0)

    if mode_type in ['R', 'S']:

        i = np.array([0, *i_fluid_solid_boundary, len(r)], dtype = np.int)
        n_segment = len(i) - 1
        #
        integral = 0.0
        for j in range(n_segment):
            
            i1 = i[j]
            i2 = i[j + 1]

            integral = integral + np.trapz(function[i1 : i2], x = r[i1 : i2])

    else:

        integral = np.trapz(function, x = r)

    print('I             = {:>10.3e}'.format(integral))
    print('I * (omega^2) = {:>10.3e}'.format(integral*(f_rad_per_s**2.0))) 

    return

if __name__ == '__main__':

    main()
