# Import modules from standard library.
import argparse
import os
from shutil import copyfile
import subprocess

# Import third-party modules.
from obspy import read

# Import custom modules.
from common import get_Mineos_out_dirs, get_Mineos_summation_out_dirs, jcom_to_mode_type_dict, mkdir_if_not_exist, mode_types_to_jcoms, read_Mineos_input_file, read_Mineos_summation_input_file, read_channel_file
from misc.cmt_io import read_mineos_cmt

# Set default names of files written by wrapper scripts.
default_file_green_in = 'green_in.txt'
default_file_syndat_in = 'syndat_in.txt'
default_file_syndat_out = 'syndat_out'

# Identify names of executables.
executable_green = 'green'
executable_syndat = 'syndat'
executable_simpledit = 'simpledit'
executable_creat_origin = 'creat_origin' # Note missing 'e'.
executable_cucss2sac = 'cucss2sac'

# Running green (calculation of Green's functions). --------------------------- 
def run_simpledit(channel_ascii_path, channel_db_path):
    '''
    Wrapper for the simpledit script, which converts a text station/channel
    listing into a CSS database format used by mineos.

    channel_ascii_path The staname field cannot contain interior apostrophes.
    '''

    # Run the command.
    cmd = '{:} {:} {:}'.format(executable_simpledit, channel_ascii_path, channel_db_path)
    print(cmd) 
    subprocess.call(cmd, shell = True)

    return

def write_green_parameter_file(run_info, summation_info):
        #dir_project, path_eigen_db_list, path_channel_db, cmt_path, out_db_path, jcoms = [2, 3], f_lims = [10.0, 260.0], n_samples = 8000):
    '''
    Writes the input parameter file for the program green.
    '''

    # Make a list of the lines of the parameter file.
    lines = [   summation_info['path_channel_db'],
                summation_info['path_eigen_db_list'],
                summation_info['path_cmt'],
                '{:12.8f} {:12.8f}'.format(*summation_info['f_lims']),
                '{:8d}'.format(summation_info['n_samples']),
                summation_info['path_green_out_db']]

    # Write the parameter file.
    print('Writing {:}'.format(summation_info['in_path_green']))
    with open(summation_info['in_path_green'], 'w') as param_file:
        
        for line in lines:
            
            param_file.write(line + '\n')

    return
            
def write_eigen_db_list_file(run_info, summation_info):
        #dir_project, jcoms, eigen_db_list_file):
    '''
    Writes a text file listing the normal-mode database files to be
    used in calculating Green's functions during mode summation.
    '''

    # Write one line per jcom.
    print('Writing {:}'.format(summation_info['path_eigen_db_list']))
    with open(summation_info['path_eigen_db_list'], 'w') as out_ID:
        
        for jcom in run_info['jcoms']:
            
            out_ID.write(os.path.join(run_info['dir_run'], jcom_to_mode_type_dict[jcom]) + '\n')

    return
            
def run_green(run_info, summation_info, skip = False):
    '''
    Wrapper for green, which calculates Green's functions (impulse response
    functions) given a list of channels (station location and orientation)
    and an earthquake source.
    '''
        #, path_channel_db, path_channel_ascii, path_cmt, path_green_out_db):
        #dir_project, channel_db_path, channel_ascii_path, cmt_path, out_db_path, jcoms = [2, 3], eigen_db_list_name = 'eigen_db_list.txt', f_lims = [10.0, 260.0], n_samples = 8000, skip = False, ):

    # Check for output files and skip calculation if they already exist.
    if skip:
        
        #if os.path.exists(path_green_db + '.wfdisc'):
        if os.path.exists(out_db_path + '.wfdisc'):
            
            print('Skipping green: Output files already exist.')
            return
    
    # Convert a text file which lists stations and channels into a CSS
    # database format used by Mineos.
    run_simpledit(summation_info['path_channels'], summation_info['path_channel_db'])

    # Define the path to the file which lists the normal mode databases to
    # be included.
    file_eigen_db_list = 'eigen_db_list.txt'
    summation_info['path_eigen_db_list'] = \
        os.path.join(summation_info['dir_cmt'], file_eigen_db_list)

    # Write the input file for the green function. 
    summation_info['in_path_green'] = os.path.join(summation_info['dir_cmt'], summation_info['file_green_in'])
    write_green_parameter_file(run_info, summation_info)
        #dir_project,
        #path_eigen_db_list,
        #channel_db_path,
        #cmt_path,
        #out_db_path,
        #jcoms           = jcoms,
        #f_lims          = f_lims,
        #n_samples       = n_samples)

    # Write the file listing the normal-mode mode databases.
    write_eigen_db_list_file(run_info, summation_info)

    # Run green. 
    cmd = '{} < {}'.format(executable_green, summation_info['in_path_green'])
    print(cmd)
    subprocess.call(cmd, shell = True)

    return

# Running creat_origin (adds header information to Green function database). --
def run_creat_origin(cmt_path, path_db):
    '''
    Wrapper for creat_origin, which adds header information to Green
    function database.
    '''
    
    # Run creat_origin.
    cmd = '{:} {} {}'.format(executable_creat_origin, cmt_path, path_db)
    print(cmd) 
    subprocess.call(cmd, shell = True)

    return

# Running syndat (convolve Green's functions with moment tensor). -------------
def write_syndat_parameter_file(summation_info):
    '''
    Writes the input file for the syndat function.
    '''
#cmt_path, green_db_path, syndat_param_path, syndat_out_path, plane = 0, datatype = 0):
    
    # Define the lines to be written to file.
    # (See Mineos manual, section 3.4.1.)
    lines = [   summation_info['path_cmt'],
                '{:1d}'.format(summation_info['plane']),
                summation_info['path_green_out_db'],
                summation_info['path_syndat_out'],
                '{:1d}'.format(summation_info['data_type'])]
    
    # Write the parameter file.
    print('{:}'.format(summation_info['path_syndat_in']))
    with open(summation_info['path_syndat_in'], 'w') as out_id:
        
        # Write line by line.
        for line in lines:
            
            out_id.write(line + '\n')

    return

def run_syndat(summation_info, plane = 0):
    #cmt_path, green_db_path, syndat_out_path, syndat_param_path, plane = 0, datatype = 0, skip = False):
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
        

    # Write the syndat input file.
    write_syndat_parameter_file(summation_info)

                #cmt_path,
                #green_db_path,
                #syndat_param_path,
                #syndat_out_path,
                #plane = plane,
                #datatype = datatype)
                
    # Run syndat.
    cmd = '{:} < {}'.format(executable_syndat, summation_info['path_syndat_in'])
    print(cmd)
    subprocess.call(cmd, shell = True)

# Running cucss2sac (converts CSS database of seismograms to text/SAC format).-
def run_cucss2sac(dir_name, name_syndat_db, name_syndat_sac = 'sac', skip = False):
    '''
    Wrapper for cucss2sac, which converts a CSS database of seismograms
    to SAC or text format.
    '''
    
    # Check for output files and skip if they already exist.
    if skip:
        
        if os.path.exists(os.path.join(dir_name, name_syndat_sac)):
            
            print('Skipping run_cucss2sac: Output directory already exists.')
            return
    
    # Change directory because it doesn't seem to work with absolute
    # paths.
    return_dir = os.getcwd()
    print('cd {:}'.format(dir_name))
    os.chdir(dir_name)
    
    try:
    
        # Run cucss2sac.
        cmd = '{:} {} {}'.format(executable_cucss2sac, name_syndat_db, name_syndat_sac)
        print(cmd)
        subprocess.call(cmd, shell = True)
        
        # Return to starting directory.
        os.chdir(return_dir)
        
    except:
        
        # If there was an error, return to starting directory.
        os.chdir(return_dir)

        # Re-raise the error.
        raise

    return

def path_to_sac(dir_output, path_cmt, station, channel):

    dir_sac = os.path.join(dir_output, 'sac')

    # Read moment tensor file.
    cmt_info = read_mineos_cmt(path_cmt)

    # Get path to SAC file.
    src_time_year_day_str = cmt_info['datetime_ref'].strftime('%Y%j')
    src_time_hms_str = cmt_info['datetime_ref'].strftime('%H:%M:%S').replace('0', ' ')
    file_sac = 'syndat_out.{:}:{:}.{:}.{:}.SAC'.format(src_time_year_day_str,
                src_time_hms_str, station, channel)
    path_sac = os.path.join(dir_sac, file_sac)

    return path_sac

def sac2mseed(dir_output, path_cmt, path_channel):
    
    station_list = read_channel_file(path_channel)

    first_iteration = True

    for station in station_list:

        channel_list = station_list[station]
        for channel in channel_list:

            path_sac = path_to_sac(dir_output, path_cmt, station, channel)
            stream_new = read(path_sac)

            if first_iteration:

                stream = stream_new
                first_iteration = False

            else:

                stream = stream + stream_new

    path_out = os.path.join(dir_output, 'sac', 'stream.mseed')
    print('Writing {:}'.format(path_out))
    stream.write(path_out)

    return

# Wrapper scripts. ------------------------------------------------------------
def summation_wrapper(path_mode_input, path_summation_input, skip = False, green2sac = False):

    # Read the mode input file.
    run_info = read_Mineos_input_file(path_mode_input)
    run_info['jcoms'] = mode_types_to_jcoms(run_info['mode_types'])

    # Read the summation input file. 
    summation_info = read_Mineos_summation_input_file(path_summation_input)
    if summation_info['f_lims'] == 'same':

        summation_info['f_lims'] = run_info['f_lims']

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
    run_info['dir_model'], run_info['dir_run'] = get_Mineos_out_dirs(run_info) 
    summation_info = get_Mineos_summation_out_dirs(run_info, summation_info,
                        file_green_in = default_file_green_in,
                        file_syndat_in = default_file_syndat_in,
                        file_syndat_out = default_file_syndat_out)

    for dir_key in ['dir_summation', 'dir_channels', 'dir_cmt']:
    
        mkdir_if_not_exist(summation_info[dir_key])

    # pannel_channel_ascii  Text database file storing ???
    # green_out_db_path     Bindary database file ???
    #path_channel_db = os.path.join(dir_model_out, 'channel_db')
    #path_channel_ascii = os.path.join(dir_mineos_in, 'station_lists', 'stations.txt')
    # Define output path for database file created by green.
    #green_out_db_path = os.path.join(run_info['dir_run'], 'green')

    #run_green(dir_project, path_channel_db, path_channel_ascii, path_cmt, out_db_path, jcoms = jcoms, skip = skip)

    # Calculate Green's functions.
    run_green(run_info, summation_info, skip = skip)

            #dir_project, path_channel_db, path_channel_ascii, path_cmt, green_out_db_path, jcoms = jcoms, skip = skip, f_lims = w_lims, n_samples = n_samples)

    #path_green_db = os.path.join(dir_project, 'green')
    #run_create_origin(path_cmt, path_green_db)

    # Run the creat_origin script, which fills out header information in the
    # Green's functions database.
    run_creat_origin(summation_info['path_cmt'], summation_info['path_green_out_db'])

    if green2sac:
    
        print('cp {:}.site {:}'.format(summation_info['path_channel_db'], os.path.join(summation_info['dir_cmt'], '{:}.site'.format('green'))))
        copyfile('{:}.site'.format(summation_info['path_channel_db']), os.path.join(summation_info['dir_cmt'], '{:}.site'.format('green')))
        run_cucss2sac(summation_info['dir_cmt'],
                'green',
                name_syndat_sac = 'green_sac')

    # Run syndat, which convolves Green's functions with moment tensor.
    #path_syndat_in = os.path.join(dir_project, 'syndat_in.txt')
    #path_syndat_out = os.path.join(dir_project, 'syndat_out')
    run_syndat(summation_info)
            #path_cmt, green_out_db_path, path_syndat_out, path_syndat_in, datatype = 0, skip = skip)

    # Run creat_origin (again), to fill out header information in the
    # seismogram database.
    run_creat_origin(summation_info['path_cmt'], summation_info['path_syndat_out'])

    # Run cucss2sac, which converts the CSS database of synthetic seismograms
    # into text files.
    print('cp {:}.site {:}'.format(summation_info['path_channel_db'], os.path.join(summation_info['dir_cmt'], '{:}.site'.format(summation_info['file_syndat_out']))))
    copyfile('{:}.site'.format(summation_info['path_channel_db']), os.path.join(summation_info['dir_cmt'], '{:}.site'.format(summation_info['file_syndat_out'])))
    run_cucss2sac(summation_info['dir_cmt'], summation_info['file_syndat_out'], skip = skip)

    ## For convenience, store the SAC output in a miniSEED file.
    sac2mseed(summation_info['dir_cmt'], summation_info['path_cmt'], summation_info['path_channels'])

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
    summation_wrapper(path_mode_input, path_summation_input, green2sac = True)

    return

if __name__ == '__main__':

    main()
