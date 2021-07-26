import argparse

import matplotlib.pyplot as plt
import numpy as np

from Ouroboros.common import load_model_full

def main():

    # 
    var_options = ['rho', 'v_p', 'v_pv', 'v_ph', 'v_s', 'v_sv', 'v_sh', 'Q_ka', 'Q_mu', 'eta', 'mu', 'ka']
    label_dict = {  'rho'   : 'Density (kg m$^{-3}$)',
                    'v_p'   : 'P-wave speed (m s$^{-1}$)',
                    'v_pv'  : 'Vertically-polarised P-wave speed (m s$^{-1}$)',
                    'v_ph'  : 'Horizontally-polarised P-wave speed (m s$^{-1}$)',
                    'v_s'   : 'S-wave speed (m s$^{-1}$)',
                    'v_sv'  : 'Vertically-polarised S-wave speed (m s$^{-1}$)',
                    'v_sh'  : 'Horizontally-polarised S-wave speed (m s$^{-1}$)',
                    'Q_ka'  : '$Q_{\kappa}$, bulk quality factor',
                    'Q_mu'  : '$Q_{\mu}$, shear quality factor',
                    'eta'   : 'Radial anistropy parameter, $\eta$',
                    'mu'    : 'Shear modulus, $\mu$ (Pa)',
                    'ka'    : 'Bulk modulus, $\kappa$ (Pa)'
                    }
    
    # Read input arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("path_model",
                help = "File path (relative or absolute) to model file.")
    parser.add_argument("--vars", choices = var_options,
                nargs = '+', help = "Specify which variables to plot (space-separated list).")
    #
    args = parser.parse_args()
    #
    path_model = args.path_model
    var_list = args.vars
    
    var_list_default = ['rho', 'v_p', 'v_s']
    if var_list is None:

        var_list = var_list_default

    n_vars = len(var_list)

    # Load model.
    model = load_model_full(path_model)

    # Create figure.
    fig_width_inches = (4.0 * n_vars)
    fig, ax_arr = plt.subplots(1, n_vars, figsize = (fig_width_inches, 8.0),
                    constrained_layout = True, sharey = True)

    if n_vars == 1:

        ax_arr = [ax_arr]

    font_size_label = 13

    for i in range(n_vars): 

        ax = ax_arr[i]

        ax.plot(model[var_list[i]], model['r'])
        ax.set_xlabel(label_dict[var_list[i]], fontsize = font_size_label)

    # Tidy axes.
    ax = ax_arr[0]
    ax.set_ylabel('Radius (m)', fontsize = font_size_label)
    ax.set_ylim([model['r'][0], model['r'][-1]])

    plt.show()

    return

if __name__ == '__main__':

    main()
