import argparse

import numpy as np

from common import get_r_fluid_solid_boundary, load_model, read_Ouroboros_input_file
from post.read_output import load_eigenfreq_Ouroboros, load_eigenfunc_Ouroboros
from run_kernels import interp_n_parts

def main():

    # Read input arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_input_file", help = "File path (relative or absolute) to Ouroboros input file.")
    parser.add_argument("mode_type", choices = ['R', 'S', 'T'], help = 'Mode type (radial, spheroidal or toroidal).')
    parser.add_argument("n", type = int, help = "Plot mode with radial order n.")
    parser.add_argument("l", type = int, help = "Plot mode with angular order l (must be 0 for radial modes).")
    parser.add_argument("--toroidal", dest = "layer_number", help = "Plot toroidal modes for the solid shell given by LAYER_NUMBER (0 is outermost solid shell).", type = int)
    parser.add_argument("--mineos", action = "store_true", help = "Plot Mineos modes (default: Ouroboros).")
    args = parser.parse_args()

    # Rename input arguments.
    path_input = args.path_to_input_file
    mode_type  = args.mode_type
    n           = args.n
    l           = args.l
    i_toroidal = args.layer_number
    use_mineos = args.mineos

    # Read the input file and command-line arguments.
    run_info = read_Ouroboros_input_file(path_input)
    run_info['use_mineos'] = use_mineos
    #Ouroboros_info, mode_type, n, l, i_toroidal = prep_Ouroboros_info()
    #run_info, mode_type, n, l, i_toroidal = prep_run_info(args)

    # Load the planetary model.
    model = load_model(run_info['path_model'])

    f_mHz = load_eigenfreq_Ouroboros(run_info, mode_type, n_q = n, l_q = l, i_toroidal = i_toroidal)
    f_Hz = f_mHz*1.0E-3
    f_rad_per_s = f_Hz*(2.0*np.pi)

    if mode_type == 'R':

        r, U = load_eigenfunc_Ouroboros(run_info, mode_type, n, l)
        U[0] = 0.0 # Value of U at planet core appears to be buggy for R modes.

    elif mode_type == 'S': 

        r, U, V = load_eigenfunc_Ouroboros(run_info, mode_type, n, l)

    elif mode_type == 'T':

        r, W = load_eigenfunc_Ouroboros(run_info, mode_type, n, l, i_toroidal = i_toroidal)

    r = r*1.0E3 # Convert to m.
    U = U/np.sqrt(1.0E9)

    # Find indices of solid-fluid boundaries in the eigenfunction grid.
    i_fluid_solid_boundary = (np.where(np.diff(r) == 0.0))[0] + 1

    # Find the fluid-solid boundary points in the model.
    i_fluid_model, r_solid_fluid_boundary_model, i_fluid_solid_boundary_model =\
        get_r_fluid_solid_boundary(model['r'], model['v_s'])

    # Interpolate from the model grid to the output grid.
    #rho  = interp_n_parts(r*1.0E3, model['r'], model['rho'], i_fluid_solid_boundary, i_fluid_solid_boundary_model)
    rho  = interp_n_parts(r, model['r'], model['rho'], i_fluid_solid_boundary, i_fluid_solid_boundary_model)

    print(rho[0], rho[-1])
    #import matplotlib.pyplot as plt
    #fig = plt.figure()
    #ax = plt.gca()
    #ax.plot(r*1.0E3, rho)
    #ax.plot(model['r'], model['rho'])
    #plt.show()

    #import sys
    #sys.exit()

    #i = np.where(np.diff(r) == 0.0)[0]
    i = np.array([0, *i_fluid_solid_boundary, len(r)], dtype = np.int)
    n_segment = len(i) - 1
    
    function = rho*(U**2.0)*(r**2.0)
    integral = 0.0
    for j in range(n_segment):
        
        i1 = i[j]
        i2 = i[j + 1]

        integral = integral + np.trapz(function[i1 : i2], x = r[i1 : i2])

    #print(integral*(f_Hz**2.0))
    print(integral*(f_rad_per_s**2.0))

    #import matplotlib.pyplot as plt
    #fig = plt.figure()
    #ax = plt.gca()

    #for j in range(n_segment):
    #    
    #    i1 = i[j]
    #    i2 = i[j + 1]

    #    #ax.plot(r[i1 : i2], function[i1 : i2])
    #    ax.plot(r[i1 : i2], rho[i1 : i2])

    #ax.axhline()

    #plt.show()

    return

if __name__ == '__main__':

    main()
