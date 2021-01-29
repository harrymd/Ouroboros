# Import modules from standard library.
import argparse
import os
import subprocess

# Import custom modules.
from common import read_Mineos_input_file

# ---------------------------------------
def run_simpledit(channel_ascii_path, channel_db_path):

    cmd = '{:} {:} {:}'.format(executable_simpledit, channel_ascii_path, channel_db_path)
    print(cmd) 
    subprocess.call(cmd, shell = True)

    return

def write_green_parameter_file(dir_project, path_eigen_db_list, path_channel_db, cmt_path, out_db_path, jcoms = [2, 3], f_lims = [10.0, 260.0], n_samples = 8000):
    '''
    Writes the input parameter file for the program green.
    '''

    # c
    
    # Make a list of the lines of the parameter file.
    lines = [   path_channel_db,
                path_eigen_db_list,
                cmt_path,
                '{:12.8f} {:12.8f}'.format(f_lims[0], f_lims[1]),
                '{:8d}'.format(n_samples),
                out_db_path]

    in_path_green = os.path.join(dir_project, in_file_green)
    with open(in_path_green, 'w') as param_file:
        
        for line in lines:
            
            param_file.write(line + '\n')

    return
            
def write_eigen_db_list_file(dir_project, jcoms, eigen_db_list_file):
    

    eigen_db_list_path = os.path.join(dir_project, eigen_db_list_file)
    jcom_dict = {2 : 'T', 3 : 'S'}
    
    with open(eigen_db_list_path, 'w') as out_ID:
        
        for jcom in jcoms:
            
            out_ID.write(os.path.join(dir_project, jcom_dict[jcom]) + '\n')
            
def run_green(run_info, summation_info, path_channel_db, path_channel_ascii, path_cmt, path_green_out_db):
        #dir_project, channel_db_path, channel_ascii_path, cmt_path, out_db_path, jcoms = [2, 3], eigen_db_list_name = 'eigen_db_list.txt', f_lims = [10.0, 260.0], n_samples = 8000, skip = False, ):

    if skip:
        
        #if os.path.exists(path_green_db + '.wfdisc'):
        if os.path.exists(out_db_path + '.wfdisc'):
            
            print('Skipping green: Output files already exist.')
            return
    
    path_eigen_db_list = os.path.join(dir_project, eigen_db_list_name)
        
    # Convert a text file which lists stations and channels into a CSS
    # database format used by Mineos.
    run_simpledit(channel_ascii_path, channel_db_path)
    
    write_green_parameter_file(
        dir_project,
        path_eigen_db_list,
        channel_db_path,
        cmt_path,
        out_db_path,
        jcoms           = jcoms,
        f_lims          = f_lims,
        n_samples       = n_samples)

    write_eigen_db_list_file(dir_project, jcoms, eigen_db_list_name)
        
    in_path_green = os.path.join(dir_project, in_file_green)
    cmd = '{} < {}'.format(executable_green, in_path_green)
    print(cmd)
    subprocess.call(cmd, shell = True)

    return

# ---------------------------------------------------------------------
def write_syndat_parameter_file(cmt_path, green_db_path, syndat_param_path, syndat_out_path, plane = 0, datatype = 0):
    
    lines = [   cmt_path,
                '{:1d}'.format(plane),
                green_db_path,
                syndat_out_path, 
                '{:1d}'.format(datatype)]
    
    with open(syndat_param_path, 'w') as out_id:
        
        for line in lines:
            
            out_id.write(line + '\n')

    return

def run_syndat(cmt_path, green_db_path, syndat_out_path, syndat_param_path, plane = 0, datatype = 0, skip = False):
    '''
    Input
    
    plane       0, 1 or 2   See Mineos manual.
    datatype    0, 1 or 2   Acceleration, velocity or displacement.
    '''
    
    #if skip:
    #    
    #    if os.path.exists(out_db_path + '.wfdisc'):
    #        
    #        print('Skipping syndat: Output file already exists.')
    #        return
        
    write_syndat_parameter_file(
                cmt_path,
                green_db_path,
                syndat_param_path,
                syndat_out_path,
                plane = plane,
                datatype = datatype)
                
    print(cmd)
    cmd = '{:} < {}'.format(executable_syndat, syndat_param_path)
    
    subprocess.call(cmd, shell = True)

# ---------------------------------------------------------------------
def run_creat_origin(cmt_path, path_db):
    
    cmd = '{:} {} {}'.format(executable_creat_origin, cmt_path, path_db)
    print(cmd) 
    subprocess.call(cmd, shell = True)

    return

# ---------------------------------------------------------------------
def run_cucss2sac(dir_name, name_syndat_db, name_syndat_sac, skip = False):
    
    if skip:
        
        if os.path.exists(os.path.join(dir_name, name_syndat_sac)):
            
            print('Skipping run_cucss2sac: Output directory already exists.')
            return
    
    ret_dir = os.getcwd()
    
    # Change directory because it doesn't seem to work with absolute
    # paths.
    os.chdir(dir_name)
    
    try:
    
        cmd = '{:} {} {}'.format(executable_cucss2sac, name_syndat_db, name_syndat_sac)
        print(cmd)
        subprocess.call(cmd, shell = True)
        
        os.chdir(ret_dir)
        
    except:
        
        os.chdir(ret_dir)
        raise

# -----------------------------------------------------------------------------
def read_Mineos_summation_input_file(path_input):
    
    # Read the summation input file.
    print('Reading {:}'.format(path_input))
    with open(path_input, 'r') as in_id:

        channel_file = in_id.readline().split()[1]

    # Store the information in a dictionary.
    summation_info = dict()
    summation_info['channel_file'] = channel_file

    return summation_info

def summation_wrapper(path_mode_input, path_summation_input):

    # Read the mode input file.
    run_info = read_Mineos_input_file(path_mode_input)

    # Read the summation input file. 
    summation_info = read_Mineos_summation_input_file(path_summation_input)

    print(summation_info)

    import sys
    sys.exit()

    # Define output files.
    # path_channel_db       Mineos variable in_dbname.
    #                       Path to the binary database file storing the
    #                       .site information (station location etc) and
    #                       .sitechan relations (channels associated with
    #                       each site.
    # path_eigen_db_list    Mineos variable db_list.
    #                       Path to the file listing the eigenfunction
    #                       database files.
    # path_cmt              Mineos variable cmt_event.
    #                       Path to the text file containing the moment-tensor
    #                       information for one earthquake.
    # path_green_out_db     Mineos variable out_dbname.
    #                       Path to the binary .wfdisc database which stores
    #                       the Green's functions.

    # pannel_channel_ascii  Text database file storing ???
    # green_out_db_path     Bindary database file ???
    path_channel_db = os.path.join(dir_model_out, 'channel_db')
    path_channel_ascii = os.path.join(dir_mineos_in, 'station_lists', 'stations.txt')
    # Define output path for database file created by green.
    green_out_db_path = os.path.join(run_info['dir_run'], 'green')

    #run_green(dir_project, path_channel_db, path_channel_ascii, path_cmt, out_db_path, jcoms = jcoms, skip = skip)
    run_green(dir_project, path_channel_db, path_channel_ascii, path_cmt, green_out_db_path, jcoms = jcoms, skip = skip, f_lims = w_lims, n_samples = n_samples)

    import sys
    sys.exit()

    #path_green_db = os.path.join(dir_project, 'green')
    #run_create_origin(path_cmt, path_green_db)
    run_create_origin(path_cmt, green_out_db_path)

    path_syndat_in = os.path.join(dir_project, 'syndat_in.txt')
    path_syndat_out = os.path.join(dir_project, 'syndat_out')
    run_syndat(path_cmt, green_out_db_path, path_syndat_out, path_syndat_in, datatype = 0, skip = skip)

    run_create_origin(path_cmt, path_syndat_out)
    run_cucss2sac(dir_project, 'syndat_out', 'sac', skip = skip)

    return

def main():

    # Parse input arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("path_mode_input", help = "File path (relative or absolute) to Mineos mode input file.")
    parser.add_argument("path_summation_input", help = "File path (relative or absolute) to Mineos summation input file.")
    input_args = parser.parse_args()
    path_mode_input = input_args.path_mode_input
    path_summation_input = input_args.path_summation_input
    
    # Do the summation.
    summation_wrapper(path_mode_input, path_summation_input)

    return

if __name__ == '__main__':

    main()
