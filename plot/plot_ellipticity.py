import argparse

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from Ouroboros.common import filter_mode_list, load_eigenfunc, load_eigenfreq, read_input_file 

def main():

    # Read input arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("path_input", help = "File path (relative or absolute) to input file.")
    parser.add_argument("path_mode_list", help = "File path (relative or absolute) to mode list file.")
    parser.add_argument("--path_out", help = "Output figure path.")
    args = parser.parse_args()
    path_input = args.path_input    
    path_mode_list = args.path_mode_list
    path_out = args.path_out

    # Read input file(s).
    run_info = read_input_file(path_input)
    
    # Read modes.
    mode_type = 'S'
    mode_info = load_eigenfreq(run_info, mode_type)

    # Filter mode list.
    mode_info = filter_mode_list({'S' : mode_info}, path_mode_list)['S']

    # Define eigenfunction normalisation.
    norm_func = 'DT'
    units = 'mineos'
    normalisation_args = {'norm_func' : norm_func, 'units' : units}

    # Load eigenfunction information at surface.
    num_modes = len(mode_info['n'])
    U = np.zeros(num_modes)
    V = np.zeros(num_modes)
    #
    for i in range(num_modes):
        
        # Unpack variables.
        n = mode_info['n'][i]
        l = mode_info['l'][i]
        f = mode_info['f'][i]
        f_rad_per_s = f*1.0E-3*2.0*np.pi
        normalisation_args['omega'] = f_rad_per_s

        # Load eigenfunction.
        eigfunc_dict = load_eigenfunc(run_info, mode_type, n, l,
                            norm_args = normalisation_args)

        U[i] = eigfunc_dict['U'][-1]
        V[i] = eigfunc_dict['V'][-1]

    i_neg_U = np.where(U < 0.0)[0]
    U[i_neg_U] = U[i_neg_U]*-1.0
    V[i_neg_U] = V[i_neg_U]*-1.0

    scale = 1.0E3
    U = U*scale
    V = V*scale

    fig = plt.figure(figsize = (5.0, 8.0), constrained_layout = True)
    ax  = plt.gca()

    ax.scatter(U, V)

    ax.set_aspect(1.0)
    
    label_individual_modes = True
    if label_individual_modes:

        label_str_format = '$_{{{:d}}}S_{{{:d}}}$'
        for i in range(num_modes):

            # Unpack variables.
            n = mode_info['n'][i]
            l = mode_info['l'][i]

            # Define label string.
            label_str = label_str_format.format(n, l)

            # Create label.
            ax.annotate(label_str, (U[i], V[i]), xytext = (5.0, 5.0), textcoords = 'offset pixels')
    
    min_U = np.min(U)
    min_V = np.min(V)

    max_U = np.max(U)
    max_V = np.max(V)

    U_range = max_U - min_U
    V_range = max_V - min_V
    buff = 0.05
    U_lims = [min_U - buff*U_range, max_U + buff*U_range]
    V_lims = [min_V - buff*V_range, max_V + buff*V_range]

    ax.set_xlim(U_lims)
    ax.set_ylim(V_lims)

    font_size_label = 13
    ax.set_xlabel('Vertical component, $U$', fontsize = font_size_label)
    ax.set_ylabel('Horizontal component, $V$', fontsize = font_size_label)

    # Move left y-axis and bottim x-axis to centre, passing through (0,0)
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    
    # Eliminate upper and right axes
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    
    # Show ticks in the left and lower axes only
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    ax.xaxis.set_major_locator(ticker.MultipleLocator(2.0))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(2.0))
    
    label_groups = False
    if label_groups:

        labels = ['R0', 'R1', 'R2', 'Stoneley', 'Mixed']
        label_coords = [[5.83, -4.90], [3.71, 1.72], [2.33, 5.39], [0.98, 0.44], [1.77, 3.67]]
        label_kwargs = {'fontsize' : 13, 'ha' : 'center', 'va' : 'center'}
        n_labels = len(labels)
        for i in range(n_labels):

            ax.text(*label_coords[i], labels[i], **label_kwargs)

    if path_out is not None:

        print('Writing to {:}'.format(path_out))
        plt.savefig(path_out, dpi = 300)

    plt.show()

    return

if __name__ == '__main__':

    main()
