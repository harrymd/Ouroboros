'''
Python wrapper for calculating modes using Mineos command-line functions.
'''

import argparse
import os
import subprocess

from Ouroboros.common import (get_Mineos_out_dirs, jcom_to_mode_type_dict,
                        mkdir_if_not_exist, mode_types_to_jcoms,
                        read_Mineos_input_file, load_model)

# Set default names of files written by wrapper scripts.
default_in_file_minos_bran = 'minos_bran_in.txt'
default_in_file_eigcon = 'eigcon_in.txt'
default_in_file_green = 'green_in.txt'
default_out_file_minos_bran = 'minos_bran_out'

# Identify names of executables.
executable_minos_bran = 'minos_bran'
executable_eigcon = 'eigcon'
executable_eigen2asc = 'eigen2asc'

# Running minos_bran (calculates normal modes). -------------------------------
def jcom_to_minos_bran_out_file(dir_project, jcom, file_out_name):
    '''
    Get the path to the minos bran output file.
    '''

    mode_type_char = jcom_to_mode_type_dict[jcom]
    file_out_plain = '{:}_{:}.txt'.format(file_out_name, mode_type_char)
    file_out_bin = '{:}_{:}.bin'.format(file_out_name, mode_type_char)
    
    # Get paths from directory and file name.
    file_out_plain  = os.path.join(dir_project, file_out_plain)
    file_out_bin    = os.path.join(dir_project, file_out_bin)
        
    return file_out_plain, file_out_bin

def write_minos_bran_parameter_file(run_info, jcom):
    '''
    Write the input file for minos_bran.
    '''
    
    # Get the paths of the minos_bran output files.
    file_out_plain, file_out_bin = jcom_to_minos_bran_out_file(
                                                run_info['dir_run'],
                                                jcom,
                                                run_info['out_file_minos_bran'])
                                               
    
    # Format strings for the parameter file.
    fmts    = [     '{:}',
                    '{:}',
                    '{:}',
                    '{:12.8e} {:12.8e}', '{:1d}',
                    '{:6d} {:6d} {:12.8e} {:12.8e} {:6d} {:6d}']

    # Variables to be written to the parameter file.
    inputs  = [ (run_info['path_model'],),
                (file_out_plain,),
                (file_out_bin,),
                (run_info['eps'], run_info['f_grav_cutoff']),
                (jcom,),
                (*run_info['l_lims'], *run_info['f_lims'], *run_info['n_lims'])] 
                
    # Create the lines of the parameter file.
    lines = []
    for fmt, inp in zip(fmts, inputs):
        
        line = fmt.format(*inp)
        lines.append(line)
    
    # Write the parameter file.
    in_path_minos_bran = os.path.join(run_info['dir_run'], run_info['in_file_minos_bran'])
    print('Writing to {:}'.format(in_path_minos_bran))
    with open(in_path_minos_bran, 'w') as param_file_id:
        
        for line in lines:
            
            param_file_id.write(line + '\n')

    return
    
def run_minos_bran(run_info, skip = False):
    '''
    Wrapper for the minos_bran script from the Mineos package.
    
    Input
    
    skip    Will detect if the output files already exist, and,
            if so, do nothing.
    '''
    
    # Loop over different mode types.
    for jcom in run_info['jcoms']:
        
        # Skip mode calculation if output files already exist.
        if skip:
            
            # Get names of output files.
            file_out_plain, file_out_bin = jcom_to_minos_bran_out_file(
                                                run_info['dir_run'],
                                                jcom,
                                                run_info['out_file_minos_bran'])


            # Check for existence of output files.
            if      os.path.exists(file_out_plain) \
                and os.path.exists(file_out_bin):
                
                print('Skipping minos_bran: Output files already exist.')
                continue
        
        # Write the minos bran parameter file.
        write_minos_bran_parameter_file(run_info, jcom)
        
        # Call the minos_bran command.
        in_path_minos_bran = os.path.join(run_info['dir_run'], run_info['in_file_minos_bran'])
        cmd = '{} < {}'.format(executable_minos_bran, in_path_minos_bran)
        print(cmd)
        subprocess.call(cmd, shell = True)
        
    return

# Running eigcon (post-processing for minos_bran, produces binary database). --
def jcom_to_eigcon_db_path(dir_project, jcom, db_name):
    '''
    For a given mode type (jcom), returns the path to the database file
    created by eigcon.
    '''

    mode_type_char = jcom_to_mode_type_dict[jcom]
    db_file = '{:}{:}'.format(db_name, mode_type_char)
    
    # Get the path from directory and file name.
    db_path = os.path.join(dir_project, db_file)
    
    return db_path

def write_eigcon_parameter_file(run_info, jcom, db_name):
    '''
    Writes the parameter files for the eigcon code.
    '''

    # Find the path of the databse file produced by eigcon.
    db_path = jcom_to_eigcon_db_path(run_info['dir_run'], jcom, db_name)

    # Get the paths to the minos_bran output files.
    txt_file_in_path, bin_file_in_path = jcom_to_minos_bran_out_file(
                                                run_info['dir_run'],
                                                jcom,
                                                run_info['out_file_minos_bran'])

    # Make a list of format strings.
    fmts    = [     '{:1d}', '{:}', '{:12.8e}', '{:}', '{:}', '{:}']

    # A list of input variables.
    inputs  = [ jcom, run_info['path_model'], run_info['max_depth'],
                txt_file_in_path, bin_file_in_path,
                db_path]

    # Write the lines to a list.
    lines = []
    for fmt, inp in zip(fmts, inputs):
        
        line = fmt.format(inp)
        lines.append(line)
    
    # Write the eigcon parameter file.
    in_path_eigcon = os.path.join(run_info['dir_run'], run_info['in_file_eigcon'])
    print('Writing {:}'.format(in_path_eigcon))
    with open(in_path_eigcon, 'w') as param_file_id:
        
        for line in lines:
            
            param_file_id.write(line + '\n')

    return

def run_eigcon(run_info, skip = False, db_name = ''):
    '''
    A wrapper for the Mineos script eigcon.
    '''
    
    # Loop over different mode types.
    for jcom in run_info['jcoms']:
        
        # Check if the output files exist, and skip this step if they do.
        if skip:
            
            # Find the path of the databse file produced by eigcon and check
            # if it exists.
            db_path = jcom_to_eigcon_db_path(run_info['dir_run'], jcom, db_name) + '.eigen'
            if      os.path.exists(db_path):
                
                print('Skipping eigcon: Output file already exists.')
                continue
        
        # Write the eigcon parameter file.
        write_eigcon_parameter_file(run_info, jcom, db_name)

        # Run eigcon.
        in_path_eigcon = os.path.join(run_info['dir_run'], run_info['in_file_eigcon'])
        cmd = '{} < {}'.format(executable_eigcon, in_path_eigcon)
        print(cmd)
        subprocess.call(cmd, shell = True)

    return

# Running eigen2asc (converts binary database to text files). -----------------
def jcom_to_eigen2asc_out_dir(dir_project, jcom):
    '''
    Returns path to directory containing eigen2asc output.
    '''
    
    mode_type_char = jcom_to_mode_type_dict[jcom]
    eigen2asc_out_dir = 'eigen_txt_{:}'.format(mode_type_char)
    
    # Get path to directory.
    eigen2asc_out_dir = os.path.join(dir_project, eigen2asc_out_dir)
    
    return eigen2asc_out_dir

def run_eigen2asc(run_info, jcoms, db_name = '', skip = False):
    '''
    Runs the Mineos script eigen2asc which converts binary eigenfunction
    files to text format.
    '''
    
    # Loop over mode types (jcoms).
    for jcom in jcoms:

        # Find the name of the database file produced by eigcon.
        db_file = jcom_to_eigcon_db_path(run_info['dir_run'], jcom, db_name)

        # Find the name of the output directory to be used by eigen2asc.
        out_dir = jcom_to_eigen2asc_out_dir(run_info['dir_run'], jcom)
        
        # Check if output files already exist, and skip if they do.
        if skip:
            
            if os.path.exists(out_dir):
                
                print('Skipping eigen2asc: Output directory already exists.')
                continue
    
        # Run eigen2asc.
        cmd = '{} {:6d} {:6d} {:6d} {:6d} {} {}'.format(
                    executable_eigen2asc,
                    *run_info['n_lims'],
                    *run_info['l_lims'],
                    db_file,
                    out_dir)
        print(cmd)
        subprocess.call(cmd, shell = True)

    return
    
# Wrapper scripts. ------------------------------------------------------------
def calculate_modes_with_mineos_wrapper(path_input_file):
    '''
    Wrapper script for calculate_modes_with_mineos().
    '''

    # Read the input file.
    run_info = read_Mineos_input_file(path_input_file)

    # Find the gravity cut-off frequency.
    if run_info['grav_switch'] == 0:

        raise NotImplementedError('Mineos does not support neglect of gravity.')

    if run_info['grav_switch'] == 1:
        
        # Neglect gravitational terms for all modes.
        run_info['f_grav_cutoff'] = 0.0

    elif run_info['grav_switch'] == 2:
        
        # Include gravitational terms for all modes.
        # We do this by setting the cutoff frequency to an arbitarily high value.
        run_info['f_grav_cutoff'] = 1000.0*run_info['f_lims'][1]

    else:

        raise ValueError('Value of grav_switch {:} not recognised.'.format(run_info['grav_switch']))

    # Set mineos switch controlling which mode types are calculated.
    run_info['jcoms'] = mode_types_to_jcoms(run_info['mode_types'])

    # Find the numerical value of the cut-off depth if 'all' flag was used
    # (i.e. not cut-off, store eigenfunctions for entire planet).
    if run_info['max_depth'] == 'all':

        raise ValueError('Mineos has a bug when summing radial modes if max_depth is too large (the exact value is not known).')
        model = load_model(run_info['path_model'])
        r_srf_km = model['r'][-1]*1.0E-3 

    # Set directory names.
    dir_model_out, run_info['dir_run'] = get_Mineos_out_dirs(run_info)

    # Create directories if needed.
    for dir_ in [run_info['dir_output'], dir_model_out, run_info['dir_run']]:
        
        mkdir_if_not_exist(dir_)

    # Set names of some default input and output files.
    run_info['in_file_minos_bran'] = default_in_file_minos_bran
    run_info['out_file_minos_bran'] = default_out_file_minos_bran
    run_info['in_file_eigcon'] = default_in_file_eigcon

    # Calculate the modes.
    calculate_modes_with_mineos(run_info)

    return

def calculate_modes_with_mineos(run_info, skip = False):
    '''
    Calculate modes with Mineos.
    '''
        
    # Run minos_bran, which calculates the normal modes.
    run_minos_bran(run_info, skip = skip)

    # Run eigcon, which post-processes the minos_bran output, creating
    # a database file.
    run_eigcon(run_info, skip = skip)

    # Run eigen2asc, which converts binary eigenfunction database into
    # text files.
    run_eigen2asc(run_info, run_info['jcoms'], skip = skip)

    return

def main():

    # Parse input arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_input_file", help = "File path (relative or absolute) to Mineos input file.")
    input_args = parser.parse_args()
    path_to_input_file = input_args.path_to_input_file

    # Calculate the modes.
    calculate_modes_with_mineos_wrapper(path_to_input_file)

    return

if __name__ == '__main__':
    
    main()
