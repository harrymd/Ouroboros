import argparse
import os

import numpy as np

from Ouroboros.common import load_eigenfreq, read_input_file

def main():

    # Read input arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("path_input", help = "File path (relative or absolute) to input file.")
    args = parser.parse_args()
    path_input = args.path_input

    # Read input file.
    run_info = read_input_file(path_input)

    # Get mode information.
    mode_info = load_eigenfreq(run_info, 'S')

    f_qi = 5.40 # Centering on second quasi-intersection.
    #f_half_width = 0.5
    f_half_width = 0.714 # Gives an interval of 1.0 mHz after 15% tapering
                            # at each each end.
    f_min = f_qi - f_half_width
    f_max = f_qi + f_half_width
    
    dir_out = '../../input/mineos/mode_lists/'

    name_out = 'rayleigh_0_near_qi2.txt'
    path_out = os.path.join(dir_out, name_out)
    i_choose = np.where((mode_info['f'] < f_max) & (mode_info['f'] > f_min) & (mode_info['n'] == 0))[0]
    print('Writing {:}'.format(path_out))
    with open(path_out, 'w') as out_id:
        
        for i in i_choose:
            
            out_id.write('{:>5d} {:>5d}\n'.format(mode_info['n'][i], mode_info['l'][i]))

    name_out = 'rayleigh_1_near_qi2.txt'
    path_out = os.path.join(dir_out, name_out)
    i_choose = np.where((mode_info['f'] < f_max) & (mode_info['f'] > f_min) & (mode_info['n'] == 1))[0]
    print('Writing {:}'.format(path_out))
    with open(path_out, 'w') as out_id:
        
        for i in i_choose:
            
            out_id.write('{:>5d} {:>5d}\n'.format(mode_info['n'][i], mode_info['l'][i]))

    name_out = 'qi2.txt'
    path_out = os.path.join(dir_out, name_out)
    i_choose = np.where((mode_info['f'] < f_max) & (mode_info['f'] > f_min) & ((mode_info['n'] == 2) | mode_info['n'] == 3))[0]
    print('Writing {:}'.format(path_out))
    with open(path_out, 'w') as out_id:
        
        for i in i_choose:
            
            out_id.write('{:>5d} {:>5d}\n'.format(mode_info['n'][i], mode_info['l'][i]))

    name_out = 'all_modes_near_qi2.txt'
    path_out = os.path.join(dir_out, name_out)
    i_choose = np.where((mode_info['f'] < f_max) & (mode_info['f'] > f_min))[0]
    print('Writing {:}'.format(path_out))
    with open(path_out, 'w') as out_id:
        
        for i in i_choose:
            
            out_id.write('{:>5d} {:>5d}\n'.format(mode_info['n'][i], mode_info['l'][i]))

    name_out = 'stoneley_near_qi2.txt'
    path_out = os.path.join(dir_out, name_out)
    n_list = [ 2,  2,  2,  2,  3,  3,  3]
    l_list = [22, 23, 24, 25, 26, 27, 28]
    num_modes = len(n_list)
    print('Writing {:}'.format(path_out))
    with open(path_out, 'w') as out_id:

        for i in range(num_modes):

            out_id.write('{:>5d} {:>5d}\n'.format(n_list[i], l_list[i]))

    name_out = 'rayleigh_2_near_qi2.txt'
    path_out = os.path.join(dir_out, name_out)
    n_list = [ 3,  3,  3,  3,  3,  2,  2,  2,  2]
    l_list = [21, 22, 23, 24, 25, 26, 27, 28, 29]
    num_modes = len(n_list)
    print('Writing {:}'.format(path_out))
    with open(path_out, 'w') as out_id:

        for i in range(num_modes):

            out_id.write('{:>5d} {:>5d}\n'.format(n_list[i], l_list[i]))

    name_out = 'n2_near_qi2.txt'
    path_out = os.path.join(dir_out, name_out)
    i_choose = np.where((mode_info['f'] < f_max) & (mode_info['f'] > f_min) & (mode_info['n'] == 2))[0]
    print('Writing {:}'.format(path_out))
    with open(path_out, 'w') as out_id:
        
        for i in i_choose:
            
            out_id.write('{:>5d} {:>5d}\n'.format(mode_info['n'][i], mode_info['l'][i]))

    name_out = 'n3_near_qi2.txt'
    path_out = os.path.join(dir_out, name_out)
    i_choose = np.where((mode_info['f'] < f_max) & (mode_info['f'] > f_min) & (mode_info['n'] == 3))[0]
    print('Writing {:}'.format(path_out))
    with open(path_out, 'w') as out_id:
        
        for i in i_choose:
            
            out_id.write('{:>5d} {:>5d}\n'.format(mode_info['n'][i], mode_info['l'][i]))

    name_out = 'rayleigh_2_and_2S25_near_qi2.txt'
    path_out = os.path.join(dir_out, name_out)
    n_list = [ 3,  3,  3,  3,  3,  2,  2,  2,  2,  2]
    l_list = [21, 22, 23, 24, 25, 26, 27, 28, 29, 25]
    num_modes = len(n_list)
    print('Writing {:}'.format(path_out))
    with open(path_out, 'w') as out_id:

        for i in range(num_modes):

            out_id.write('{:>5d} {:>5d}\n'.format(n_list[i], l_list[i]))

    name_out = 'rayleigh_2_and_neighbouring_3_stoneley_modes_near_qi2.txt'
    path_out = os.path.join(dir_out, name_out)
    n_list = [ 3,  3,  3,  3,  3,  2,  2,  2,  2,   2,  2,  3]
    l_list = [21, 22, 23, 24, 25, 26, 27, 28, 29,  24, 25, 26]
    num_modes = len(n_list)
    print('Writing {:}'.format(path_out))
    with open(path_out, 'w') as out_id:

        for i in range(num_modes):

            out_id.write('{:>5d} {:>5d}\n'.format(n_list[i], l_list[i]))

    name_out = 'rayleigh_2_very_close_to_qi2.txt'
    path_out = os.path.join(dir_out, name_out)
    n_list = [ 3,  3,  2]
    l_list = [24, 25, 26]
    num_modes = len(n_list)
    print('Writing {:}'.format(path_out))
    with open(path_out, 'w') as out_id:

        for i in range(num_modes):

            out_id.write('{:>5d} {:>5d}\n'.format(n_list[i], l_list[i]))

    return

if __name__ == '__main__':

    main()
