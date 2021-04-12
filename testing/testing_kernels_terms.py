import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from Ouroboros.common import get_path_adjusted_model, get_r_fluid_solid_boundary, interp_n_parts, load_eigenfreq, load_eigenfunc, load_model, load_model_full, read_input_file
from Ouroboros.kernels.kernels import kernel_spheroidal_mu

def main():

    # Parse input arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("path_input", help = "File path (relative or absolute) to Ouroboros input file.")
    parser.add_argument("n", type = int)
    parser.add_argument("l", type = int)
    input_args = parser.parse_args()
    path_input = input_args.path_input
    n_q = input_args.n
    l_q = input_args.l
    name_input = os.path.splitext(os.path.basename(path_input))[0]

    # Read the input file.
    run_info = read_input_file(path_input)

    # Kernels are calculated without attenuation.
    run_info['use_attenuation'] = False
    mode_type = 'S'

    # Load the planetary model.
    model_path = get_path_adjusted_model(run_info)
    model = load_model(model_path)

    # Find the fluid-solid boundary points in the model.
    i_fluid_model, r_solid_fluid_boundary_model, i_fluid_solid_boundary_model =\
        get_r_fluid_solid_boundary(model['r'], model['v_s'])

    # Load modes.
    mode_info = load_eigenfreq(run_info, mode_type) 
    n_list = mode_info['n']
    l_list = mode_info['l']
    f_mHz_list = mode_info['f']
    f_Hz_list = f_mHz_list*1.0E-3
    f_rad_per_s_list = f_Hz_list*2.0*np.pi

    # Define normalisation arguments.
    i = np.where((n_list == n_q) & (l_list == l_q))[0]
    norm_args = {'norm_func' : 'DT', 'units' : 'SI', 'omega' : f_rad_per_s_list[i]}

    eigfunc_dict = load_eigenfunc(run_info, 'S', n_q, l_q, norm_args = norm_args)
        
    # Find indices of solid-fluid boundaries in the eigenfunction grid.
    i_fluid_solid_boundary = (np.where(np.diff(eigfunc_dict['r']) == 0.0))[0] + 1

    # Interpolate from the model grid to the output grid.
    rho  = interp_n_parts(eigfunc_dict['r'], model['r'], model['rho'], i_fluid_solid_boundary, i_fluid_solid_boundary_model)
    v_p  = interp_n_parts(eigfunc_dict['r'], model['r'], model['v_p'], i_fluid_solid_boundary, i_fluid_solid_boundary_model)
    v_s  = interp_n_parts(eigfunc_dict['r'], model['r'], model['v_s'], i_fluid_solid_boundary, i_fluid_solid_boundary_model)
    i_fluid = np.where(v_s == 0.0)[0]

    # Calculate the kernels.
    a, b, c = kernel_spheroidal_mu(eigfunc_dict['r'],
                    eigfunc_dict['U'], eigfunc_dict['V'],
                    eigfunc_dict['Up'], eigfunc_dict['Vp'],
                    l_q, f_rad_per_s_list[i], i_fluid = i_fluid,
                    return_terms = True)

    model_full = load_model_full(model_path)
    # Throw away unnecessary items.
    keys_to_keep = ['T_ref', 'r', 'mu', 'ka', 'Q_mu', 'Q_ka']
    model_full = {key : model_full[key] for key in keys_to_keep}
    model_interpolated = dict()
    model_interpolated['r'] = eigfunc_dict['r']
    for key in ['mu', 'ka', 'Q_mu', 'Q_ka']:

        model_interpolated[key] = np.interp(eigfunc_dict['r'], model_full['r'], model_full[key])

    model_full  = model_interpolated

    import matplotlib.pyplot as plt

    fig, ax_arr = plt.subplots(1, 2, figsize = (14.0, 8.5), constrained_layout = True)

    #ax.plot(a, eigfunc_dict['r'], label = 'a')
    #ax.plot(b, eigfunc_dict['r'], label = 'b')
    #ax.plot(c, eigfunc_dict['r'], label = 'c')

    ax = ax_arr[0]

    ax.plot(model_full['mu'], eigfunc_dict['r'], label = 'mu')

    ax = ax_arr[1]

    ax.plot(model_full['mu']*a, eigfunc_dict['r'], label = 'mu * a')
    ax.plot(model_full['mu']*b, eigfunc_dict['r'], label = 'mu * b')
    ax.plot(model_full['mu']*c, eigfunc_dict['r'], label = 'mu * c')
    ax.plot(model_full['mu']*(a + b + c), eigfunc_dict['r'], label = 'mu * sum')


    ax.legend()

    plt.show()

    return

if __name__ == '__main__':

    main()
