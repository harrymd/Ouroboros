'''
Wrapper for calculating the normal modes of a given Earth model.
'''

import os
import argparse
from shutil import copyfile, move
import time

import numpy as np

from Ouroboros.common import (get_Ouroboros_out_dirs, get_path_adjusted_model,
        load_eigenfreq_Ouroboros, load_eigenfunc_Ouroboros,
        read_Ouroboros_input_file, mkdir_if_not_exist, load_model_full,
        write_model)
from Ouroboros.kernels.run_kernels import kernels_wrapper
from Ouroboros.modes.attenuation_correction import (
        apply_atten_correction_all_modes, create_adjusted_model,
        dispersion_correction)
from Ouroboros.modes.compute_modes import (radial_modes, spheroidal_modes,
        toroidal_modes)
from Ouroboros.modes.calculate_potential import potential_all_modes
from Ouroboros.modes.calculate_gradient import (gradient_all_modes_R_or_S,
        gradient_all_modes_T)

# Scripts for iterative attenuation correction. -------------------------------
def rescale_eigenfunctions(run_info, mode_type, rescale_factor, j_skip = None):
    '''
    Rescale eigenfunctions of a given mode type. Replaces the original
    output files.
    '''

    # Find the directory containing the eigenfunctions (which is contained
    # within the eigenvalue directory) based on the Ouroboros parameters.
    _, _, _, dir_eigval      = get_Ouroboros_out_dirs(run_info, mode_type)
    dir_eigenfuncs  = os.path.join(dir_eigval, 'eigenfunctions')

    # Set the normalisation.
    normalisation_args = {'norm_func' : 'mineos', 'units' : 'ouroboros'}

    # Get list of modes.
    mode_info = load_eigenfreq_Ouroboros(run_info, mode_type, i_toroidal = None)
    n = mode_info['n']
    l = mode_info['l']
    f_mHz = mode_info['f']
    f_rad_per_s = f_mHz*1.0E-3*2.0*np.pi

    # Count number of modes.
    num_modes = len(n)
    #assert(len(rescale_factor) == num_modes,
    #    'Rescale factor list must be same length as list of modes.')
    
    # Get a list of variables.
    if mode_type == 'S':

        vals = ['U', 'Up', 'V', 'Vp', 'P', 'Pp']

    elif mode_type == 'R':

        vals = ['U', 'Up', 'P', 'Pp']

    elif mode_type == 'T':

        raise NotImplementedError

    else:

        raise ValueError

    # Loop over modes.
    for i in range(num_modes):

        if not (i in j_skip):

            # Load eigenfunction.
            normalisation_args['omega'] = f_rad_per_s[i]

            eigfunc_dict = load_eigenfunc_Ouroboros(run_info, mode_type,
                                n[i], l[i], **normalisation_args)
            eigfunc_dict['r'] = eigfunc_dict['r']*1.0E-3 # Convert to km.

            # Rescale.
            for val in vals:

                eigfunc_dict[val] = eigfunc_dict[val]*rescale_factor[i]
                
            # Save re-scaled eigenfunction.
            file_eigenfunc  = '{:>05d}_{:>05d}.npy'.format(n[i], l[i])
            path_eigenfunc  = os.path.join(dir_eigenfuncs, file_eigenfunc)
            out_arr = [eigfunc_dict[val] for val in vals]
            out_arr = [eigfunc_dict['r'], *out_arr]
            print("Re-writing {:}".format(path_eigenfunc))
            print(rescale_factor[i])
            np.save(path_eigenfunc, out_arr)

    return

def relabel_modes(run_info, mode_type):
    '''
    After attenuation correction is applied, modes change in frequency, and
    this may necessitate re-labelling of the n-index. By definition, n is the
    0-based index of the mode in a list of modes of the same l-value when the
    list is sorted by increasing frequency.
    '''

    # Find the directory containing the eigenfunctions (which is contained
    # within the eigenvalue directory) based on the Ouroboros parameters.
    _, _, _, dir_output      = get_Ouroboros_out_dirs(run_info, mode_type)

    # Load mode information.
    mode_info = load_eigenfreq_Ouroboros(run_info, mode_type)
    n_old = mode_info['n']
    l = mode_info['l']
    f_old = mode_info['f_0']
    f_new = mode_info['f']
    Q_new = mode_info['Q']

    # Get list of unique l-values.
    l_vals = np.sort(np.unique(l))

    # Prepare output arrays.
    num_modes = len(f_new)
    n_new_list = np.zeros(num_modes, dtype = np.int)
    l_list = np.zeros(num_modes, dtype = np.int)
    
    # Loop over l-values.
    i_sort = np.zeros(num_modes, dtype = np.int)
    k = 0
    for l_i in l_vals:

        # Find modes with the current l values.
        i = np.where(l == l_i)[0]
        num_i = len(i)

        # Sort these modes by their new frequency.
        j = np.argsort(f_new[i])
        i_sort[k : k + num_i] = i[j]
        
        # Determine their new n-values. 
        n_old_i = n_old[i]
        n_min_old_i = np.min(n_old_i)
        n_new_i = np.array(range(n_min_old_i, n_min_old_i + num_i), dtype = np.int)

        # Store their l-value (unchanged) and new n-value.
        l_list    [k : k + num_i] = l_i
        n_new_list[k : k + num_i] = n_new_i

        # Move to next point in index.
        k = k + num_i
    
    # Sort the arrays according to the new system.
    f_old = f_old[i_sort]
    n_old = n_old[i_sort]
    f_new = f_new[i_sort] 
    Q_new = Q_new[i_sort]
    
    # Write a list of the re-labelling.
    fmt = '{:>10d} {:>10d} {:>10d} {:>10d}\n'
    path_relabel = os.path.join(dir_output, 'mode_relabelling.txt')
    print("Writing {:}".format(path_relabel))
    with open(path_relabel, 'w') as out_id:
        
        out_id.write('{:>10} {:>10} {:>10} {:>10}\n'.format('Original n', 'Original l', 'New n', 'New l'))
        for k in range(num_modes):

            out_id.write(fmt.format(n_old[k], l_list[k], n_new_list[k], l_list[k]))
    
    # Re-write the eigenvalues file.
    #path_eigvals = os.path.join(dir_output, 'eigenvalues_new.txt')
    path_eigvals = os.path.join(dir_output, 'eigenvalues.txt')
    print("Re-writing {:}".format(path_eigvals))
    fmt = '{:>10d} {:>10d} {:>19.12e} {:>19.12e} {:>19.12e}\n'
    with open(path_eigvals, 'w') as out_id:

        for k in range(num_modes):

            out_id.write(fmt.format(n_new_list[k], l_list[k], f_old[k], f_new[k], Q_new[k])) 
    
    # Re-label the eigenfunctions.
    relabel_eigenfunctions(run_info, mode_type)

    return

def relabel_eigenfunctions(run_info, mode_type):
    '''
    After running relabel_modes(), eigenfunction files have to be relabelled
    too.
    '''

    # Find the directory containing the eigenfunctions (which is contained
    # within the eigenvalue directory) based on the Ouroboros parameters.
    _, _, _, dir_output      = get_Ouroboros_out_dirs(run_info, mode_type)
    dir_eigenfuncs = os.path.join(dir_output, 'eigenfunctions')
    path_relabel = os.path.join(dir_output, 'mode_relabelling.txt')
    n_old, l_old, n_new, l_new = np.loadtxt(path_relabel, skiprows = 1, dtype = np.int).T

    num_modes = len(n_old)

    # Loop over modes and move files to temporary locations.
    for i in range(num_modes):

        if (n_old[i] != n_new[i]) or (l_old[i] != l_new[i]):

            file_eigenfunc_old = '{:>05d}_{:>05d}.npy'.format(n_old[i], l_old[i])
            path_eigenfunc_old = os.path.join(dir_eigenfuncs, file_eigenfunc_old)

            file_eigenfunc_tmp  = '{:>05d}_{:>05d}_tmp.npy'.format(n_old[i], l_old[i])
            path_eigenfunc_tmp  = os.path.join(dir_eigenfuncs, file_eigenfunc_tmp)
            
            print('cp {:} {:}'.format(path_eigenfunc_old, path_eigenfunc_tmp))
            copyfile(path_eigenfunc_old, path_eigenfunc_tmp)

    # Loop over modes and move temporary files to new locations.
    for i in range(num_modes):

        if (n_old[i] != n_new[i]) or (l_old[i] != l_new[i]):

            file_eigenfunc_tmp  = '{:>05d}_{:>05d}_tmp.npy'.format(n_old[i], l_old[i])
            path_eigenfunc_tmp  = os.path.join(dir_eigenfuncs, file_eigenfunc_tmp)

            file_eigenfunc_new  = '{:>05d}_{:>05d}.npy'.format(n_new[i], l_new[i])
            path_eigenfunc_new  = os.path.join(dir_eigenfuncs, file_eigenfunc_new)
            
            print('mv {:} {:}'.format(path_eigenfunc_tmp, path_eigenfunc_new))
            move(path_eigenfunc_tmp, path_eigenfunc_new)

    return

def iterative_attenuation_correction(run_info, mode_type, max_abs_f_diff_mHz_thresh = 1.0E-6, n_iterations_max = 5):
    '''
    Attenuation causes frequency shifts. This is dependent on the
    eigenfunctions through the sensitivity kernels. But the eigenfunctions
    depend on the frequency due to normalisation. Therefore we apply an
    iterative attenuation correction.
    1. Use sensitivity kernels to correct frequencies.
    2. Use new frequencies to re-scale eigenfunctions and kernels.
    3. If no convergence, go back to step 1.
    '''

    # Load the frequencies from the initial calculation.
    mode_info_old = load_eigenfreq_Ouroboros(run_info, mode_type)

    num_modes = len(mode_info_old['n'])
    j_skip = []

    # Loop until mode frequencies have converged or maximum number of
    # iterations is reached.
    for i in range(n_iterations_max):

        print("Iteration {:>3d} (max {:>3d})".format(i + 1, n_iterations_max))
        print("{:>5d} modes converged so far.".format(len(j_skip)))

        # Calculate potential.
        potential_all_modes(run_info, mode_type, j_skip = j_skip)

        # Calculate gradient.
        gradient_all_modes(run_info, mode_type, j_skip = j_skip)

        # Calculate kernels.
        kernels_wrapper(run_info, mode_type, j_skip = j_skip)

        # Apply attenuation correction.
        apply_atten_correction_all_modes(run_info, mode_type)

        # Compare frequencies.
        f_old_mHz = mode_info_old['f']
        mode_info_new = load_eigenfreq_Ouroboros(run_info, mode_type)
        f_new_mHz = mode_info_new['f']

        # Re-scale eigenfunctions.
        #rescale_factor = (f_old_mHz/f_new_mHz)**2.0
        rescale_factor = (f_old_mHz/f_new_mHz)
        rescale_eigenfunctions(run_info, mode_type, rescale_factor, j_skip = j_skip)
        
        # Check convergence.
        abs_f_diff_mHz = np.abs(f_new_mHz - f_old_mHz)
        j_skip = np.where(abs_f_diff_mHz < max_abs_f_diff_mHz_thresh)[0]
        if len(j_skip) == num_modes:

            print("Convergence.")
            break

        else:

            if i == (n_iterations_max - 1):

                print("Iterative attenuation correction failed to converge.")

            else:

               mode_info_old = mode_info_new 

    return

# Main wrapper for mode calculation. ------------------------------------------
def main():
    '''
    Control script for running Ouroboros.
    Run from the command line using
    python3 modes/calculate_modes.py example/input/example_input_Ouroboros_modes.txt
    '''
    
    # Make announcement.
    print('Ouroboros')

    # Parse input arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_input_file", help = "File path (relative or absolute) to Ouroboros input file.")
    input_args = parser.parse_args()
    Ouroboros_input_file = input_args.path_to_input_file
    name_input = os.path.splitext(os.path.basename(Ouroboros_input_file))[0]

    # Read the input file.
    Ouroboros_info = read_Ouroboros_input_file(Ouroboros_input_file)

    # Create frequency-adjusted model if applying attenuation.
    if Ouroboros_info['use_attenuation']:

        create_adjusted_model(Ouroboros_info)

    # Set the 'g_switch' string: 0 -> noG, 1 -> G, 2 -> GP.
    g_switch_strs = ['noGP', 'G', 'GP']
    g_switch_str = g_switch_strs[Ouroboros_info['grav_switch']]

    # Loop over mode types.
    Ouroboros_info['dirs_type'] = dict()
    for mode_type in Ouroboros_info['mode_types']:
        
        # Start timer.
        start_time = time.time()

        # Set the 'switch' string, e.g. 'S_GP'.
        if mode_type == 'T':

            switch = None 

        else:

            switch = '{:}_{:}'.format(mode_type, g_switch_str)

        Ouroboros_info['switch'] = switch
        
        # Prepare output directories.
        dir_model, dir_run, dir_g, dir_type = get_Ouroboros_out_dirs(Ouroboros_info, mode_type)
        for dir_ in [Ouroboros_info['dir_output'], dir_model, dir_run, dir_g, dir_type]:
            
            if dir_ is not None:
                
                mkdir_if_not_exist(dir_)

        Ouroboros_info['dirs_type'][mode_type] = dir_type

        # Copy the input file to the output directory.
        copyfile(Ouroboros_input_file, os.path.join(dir_type, name_input))

        # Run the code.
        if mode_type == 'T':

            toroidal_modes(Ouroboros_info)
        
        elif mode_type == 'R':
            
            radial_modes(Ouroboros_info)

        elif mode_type == 'S':

            spheroidal_modes(Ouroboros_info)

        else:

            raise ValueError

        # Report time elapsed.
        elapsed_time = time.time() - start_time
        path_time = os.path.join(dir_type, 'time.txt')
        print('Time used: {:>.3f} s, saving time info to {:}'.format(elapsed_time, path_time))
        with open(path_time, 'w') as out_id:

            out_id.write('{:>.3f} seconds'.format(elapsed_time))

        # If using attenuation, apply attenuation correction to frequencies.
        if Ouroboros_info['use_attenuation']:

            iterative_attenuation_correction(Ouroboros_info, mode_type)
            relabel_modes(Ouroboros_info, mode_type)

        else:

            if mode_type in ['R', 'S']:

                # Calculate potential.
                potential_all_modes(Ouroboros_info, mode_type)

            if mode_type in ['R', 'S']:

                # Calculate gradient.
                gradient_all_modes_R_or_S(Ouroboros_info, mode_type)

            elif mode_type == 'T':

                # Calculate gradient.
                gradient_all_modes_T(Ouroboros_info)

            # Calculate kernels.
            kernels_wrapper(Ouroboros_info, mode_type)
        
    return

if __name__ == '__main__':

    main()
