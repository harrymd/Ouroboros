import argparse

import numpy as np

from common import (    get_r_fluid_solid_boundary,
                        load_eigenfreq,
                        load_eigenfunc,
                        load_model,
                        read_input_file)
from kernels.run_kernels import interp_n_parts

def main():
    '''

    Examples:
    
    If we load the eigenfunctions with the Mineos units (--eig_norm_units mineos) then to get a correct normalisation we must scale the variables as follows:
    r_scale = 1.0/6371000
    f_scale = 1.0/(1.0754 * 10^-3)
    rho_scale = 1.0/5515.0
    
    We can then choose whether the normalisation applies to the eigenfunctions including the factor of k (--eig_norm mineos)
    
    >>> python3 misc/check_normalisation.py ../../input/mineos/input_Mineos_stoneley_noq.txt --use_mineos --r_scale 0.00000015696123 --freq_scale 929.886553 --rho_scale 0.00018132366 --eig_norm mineos --eig_norm_units mineos S 2 8
    >>> I             = integral( rho * (U^2 + (k*V)^2) * (r^2) )
    >>> I * (omega^2) =  9.943e-01
    
    or without the factor of k (--eig_norm DT)
    
    >>> python3 misc/check_normalisation.py ../../input/mineos/input_Mineos_stoneley_noq.txt --use_mineos --r_scale 0.00000015696123 --freq_scale 929.886553 --rho_scale 0.00018132366 --eig_norm DT --eig_norm_units mineos S 2 8
    >>> I             = integral( rho * (U^2 + V^2) * (r^2) )
    >>> I             =  9.943e-01
    
    When using Ouroboros units (--eig_norm_units ouroboros) then the radial coordinate must be converted to km to get a correct normalisation:
    
    >>> python3 misc/check_normalisation.py ../../input/Ouroboros/input_Ouroboros_prem_noq_noocean.txt --r_scale 1.0E-3 --freq_scale 1.0 --rho_scale 1.0 --eig_norm mineos --eig_norm_units ouroboros S 3 5
    >>> I             = integral( rho * (U^2 + (k*V)^2) * (r^2) )
    >>> I * (omega^2) =  1.000e+00
    
    No adjustment are necessary to get the correct normalisation in SI units (because the radial coordinate array is always loaded in units of m):
    
    >>> python3 misc/check_normalisation.py ../../input/Ouroboros/input_Ouroboros_prem_noq_noocean.txt --r_scale 1.0 --freq_scale 1.0 --rho_scale 1.0 --eig_norm mineos --eig_norm_units SI S 4 7
    >>> I             = integral( rho * (U^2 + (k*V)^2) * (r^2) )
    >>> I * (omega^2) =  1.000e+00
    '''

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
    parser.add_argument("--eig_norm", choices = ['mineos', 'DT'], default = 'DT', help = "Specify normalisation to be applied when loading the eigenfunctions. \'mineos\' is the normalisation function used by Mineos and Ouroboros. \'DT\' is the normalisation function used in the Dahlen and Tromp textbook. It does not include the factor of k. See also the --units flag. For more detail, see Ouroboros/doc/Ouroboros_normalisation_notes.pdf.")
    parser.add_argument("--eig_norm_units", choices = ['SI', 'ouroboros', 'mineos'], default = 'mineos', help = 'Specify units used when applying normalisation to eigenfunction. \'SI\' is SI units. \'mineos\' is Mineos units. \'ouroboros\' is Ouroboros units. See also the --norm_func flag. For more detail, see Ouroboros/doc/Ouroboros_normalisation_notes.pdf.')
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
    eig_norm    = args.eig_norm
    eig_norm_units = args.eig_norm_units

    if use_mineos:

        assert i_toroidal is None, 'The i_toroidal flag is not used with Mineos, try using mode_type == \'T\' for mantle toroidal modes and mode_type == \'I\' for inner-core toroidal modes.'
    
    # Read the input file and command-line arguments.
    run_info = read_input_file(path_input)

    # Load the planetary model.
    model = load_model(run_info['path_model'])

    if rho_scale is not None:

        model['rho'] = model['rho']*rho_scale

    # Load frequency information.
    mode_info = load_eigenfreq(run_info, mode_type, n_q = n, l_q = l)
    f_mHz = mode_info['f']

    if freq_scale is not None:

        print('Multiplying frequency by: {:>7.2e}'.format(freq_scale))
        f_mHz = f_mHz*freq_scale

    # Convert frequency to radians per second.
    f_Hz = f_mHz*1.0E-3
    f_rad_per_s = f_Hz*(2.0*np.pi)

    load_eigfunc_norm_args = {'norm_func' : eig_norm, 'units' : eig_norm_units}
    if eig_norm == 'DT':
        load_eigfunc_norm_args['omega'] = f_rad_per_s

    # Load eigenfunction(s).
    eigfunc_dict = load_eigenfunc(run_info, mode_type, n, l, norm_args = load_eigfunc_norm_args)
    r = eigfunc_dict['r']
    if mode_type == 'R':
        
        U = eigfunc_dict['U']

    elif mode_type == 'S':

        U = eigfunc_dict['U']
        V = eigfunc_dict['V']

    elif mode_type in ['T', 'I']:

        W = eigfunc_dict['W']

    if mode_type in ['S', 'T']:

        k = np.sqrt(l*(l + 1.0))

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

    print('\n')
    if mode_type in ['S', 'T']:

        # Integrate (each section integrated separately).
        # Define integration function.
        if mode_type == 'S':

            print('I             = integral( rho * (U^2 + V^2) * (r^2) )')
            function = rho*(U**2.0 + V**2.0)*(r**2.0)

        elif mode_type == 'T':

            print('I             = integral( rho * (W^2) * (r^2) )')
            function = rho*(W**2.0)*(r**2.0)
        
        if mode_type == 'S':

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
