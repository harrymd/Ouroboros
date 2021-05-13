'''
Convert a Mineos eigenvalue file to Ouroboros format.
'''
import argparse
import os

from Ouroboros.common import load_eigenfreq, read_input_file

def main():

    fmt = '{:>10d} {:>10d} {:>19.12e} {:>19.12e} {:>19.12e}\n'

    # Read input arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("path_input", help = "File path (relative or absolute) to input file.")
    parser.add_argument("mode_type", choices = ['R', 'S', 'T', 'I', 'all'])
    args = parser.parse_args()
    # Rename input arguments.
    path_input = args.path_input    
    mode_type = args.mode_type

    # Read input file(s).
    run_info = read_input_file(path_input)

    if mode_type == 'all':

        mode_type_list = run_info['mode_types']

    else:

        mode_type_list = [mode_type]

    # Write in Ouroboros format for each mode type.
    for mode_type in mode_type_list:

        # Get mode information.
        mode_info = load_eigenfreq(run_info, mode_type)

        n_modes = len(mode_info['n'])
        
        file_out = 'eigenvalues_{:}.txt'.format(mode_type)
        path_out = os.path.join(run_info['dir_run'], file_out)
        print('Writing to {:}'.format(path_out))
        with open(path_out, 'w') as out_id:

            for i in range(n_modes):

                out_id.write(fmt.format(mode_info['n'][i],
                                        mode_info['l'][i],
                                        mode_info['f'][i],
                                        mode_info['f'][i],
                                        mode_info['Q'][i]))

    return

if __name__ == '__main__':

    main()
