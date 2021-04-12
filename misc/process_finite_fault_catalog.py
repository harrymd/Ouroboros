from datetime import datetime, timedelta
from glob import glob
import os
from shutil import copyfile

import matplotlib.pyplot as plt
import numpy as np
import xml.etree.ElementTree as ET

from libcomcat.search import get_event_by_id, search
from libcomcat.exceptions import ProductNotFoundError
#from Ouroboros.parse_quakeml import parse_usgs_xml

from Ouroboros.misc.cmt_io import read_mineos_cmt

dir_base = '/Users/hrmd_work/Documents/research/stoneley/'

def download_finite_faults():
    '''
    https://github.com/usgs/libcomcat/blob/master/notebooks/Classes.ipynb
    '''

    # Define output directory.
    dir_finite = os.path.join(dir_base, 'input', 'finite_faults')

    # Get a list of earthquakes.
    earthquakes = search(   starttime = datetime(1900, 1, 1, 0, 0),
                            #endtime =   datetime(2100, 1, 1, 0, 0),
                            endtime =   datetime(2017, 1, 1, 0, 0),
                            minmagnitude = 8.0,
                            orderby = 'time', limit = 1000) #1000)

    # Loop over candidate earthquakes.
    datetime_strs = []
    n_candidates = len(earthquakes)
    j = 0
    for i in range(n_candidates):

        # Check for product
        #product = earthquake.hasProduct('shakemap')

        try:

            # Get finite CMT solution.
            earthquake = earthquakes[i]
            detail = earthquake.getDetailEvent()

            finite_fault = detail.getProducts('finite-fault')[0]
            finite_cmt_text = finite_fault.getContentBytes('CMTSOLUTION')[0]

            # Write to file.
            file_out = '{:>05d}.txt'.format(j)
            path_out = os.path.join(dir_finite, file_out)
            with open(path_out, 'wb') as out_id:
                
                print('Writing {:}'.format(path_out))
                out_id.write(finite_cmt_text)

            # Store the datetime information.
            datetime_strs.append(' '.join(str(earthquake).split()[1:3]))

            # Increment loop.
            j = j + 1

        except ProductNotFoundError: 

            pass

    # Write the datetime information.
    num_finite = j
    path_datetimes = os.path.join(dir_finite, 'datetimes.txt')
    print('Writing {:}'.format(path_datetimes))
    with open(path_datetimes, 'w') as out_id:

        for i in range(num_finite):

            out_id.write('{:}\n'.format(datetime_strs[i]))
    
    return

def merge_finite_and_point():

    # Get a list of the date/time of the finite events (in string format).
    dir_finite = os.path.join(dir_base, 'input', 'finite_faults')
    path_datetimes = os.path.join(dir_finite, 'datetimes.txt')
    datetime_strs = []
    with open(path_datetimes, 'r') as in_id:

        for line in in_id:

            datetime_strs.append(line.strip())

    # Convert to datetime format.
    finite_datetimes = np.array([datetime.strptime(datetime_strs[i], '%Y-%m-%d %H:%M:%S.%f') for i in range(num_finite)])

    # Get a list of the date/time for the point source events.
    dir_point = os.path.join(dir_base, 'input', 'global_cmt')
    file_event_regex = '[0-9]'*5 + '.txt'
    event_names_list_unsorted = glob(os.path.join(dir_point, file_event_regex))
    num_points = len(event_names_list_unsorted)
    
    point_datetimes = np.zeros(num_points, dtype = datetime)
    for i in range(num_points):
        
        path_cmt = os.path.join(dir_point, '{:>05d}.txt'.format(i))
        cmt = read_mineos_cmt(path_cmt)

        point_datetimes[i] = cmt['datetime_ref']
    
    # Merge the finite and point source events based on closest time match.
    # 'thresh' is the maximum time separation allowed in seconds.
    thresh = 30.0
    dir_merged = os.path.join(dir_base, 'input', 'cmts_merged')
    k = 0
    for i in range(num_finite):
        
        # For given finite event, find time difference with all point events.
        t_diff_list = (np.abs(finite_datetimes[i] - point_datetimes)) 

        # Find finite-point pair with smallest time gap.
        j = np.argmin(t_diff_list)
        t_diff = t_diff_list[j]

        # If time gap is small enough, the match is successful.
        if t_diff.total_seconds() < thresh:

            print(i, 'successful')

            # Move the finite file to new destination (note relabelling).
            path_finite_old = os.path.join(dir_finite, '{:>05d}.txt'.format(i))
            path_finite_new = os.path.join(dir_merged, 'finite_{:>05d}.txt'.format(k))
            copyfile(path_finite_old, path_finite_new)

            # Move the point file to new destinate (note relabelling).
            path_point_old = os.path.join(dir_point, '{:>05d}.txt'.format(j))
            path_point_new = os.path.join(dir_merged, 'point_{:>05d}.txt'.format(k))
            copyfile(path_point_old, path_point_new)

            k = k + 1

        # If time gap is too large, there is no match for this finite event.
        else:

            print(i, 'unsuccessful')

    return

def merge_with_radiation_patterns():

    # Get list of radiation pattern directories.
    dir_radiation = os.path.join(dir_base, 'output', 'radiation_patterns', 'database')
    dir_pattern_regex = os.path.join(dir_radiation, 'Surface-Wave_Radiation_Patterns_*')
    dir_pattern_list_unsorted = glob(dir_pattern_regex)
    
    # Loop over radiation patterns and get event times.
    num_patterns = len(dir_pattern_list_unsorted)
    pattern_id = np.zeros(num_patterns, dtype = np.int32)
    pattern_datetime  = np.zeros(num_patterns, dtype = datetime)
    for i in range(num_patterns):
        
        # Get pattern ID (an integer).
        dir_pattern = dir_pattern_list_unsorted[i]
        dir_name = os.path.basename(os.path.normpath(dir_pattern))
        pattern_id[i]  = int(dir_name.split('_')[-1])

        # Read datetime from XML metadata file.
        path_xml = os.path.join(dir_pattern, 'meta_data.xml')
        tree = ET.parse(path_xml)
        root = tree.getroot()
        LocationTime = root.find('LocationTime')
        StartTime = LocationTime.find('StartTime')
        pattern_datetime[i] = datetime.strptime(StartTime.text, '%Y-%m-%dT%H:%M:%S')
    
    # Account for half-hour shift.
    pattern_datetime = pattern_datetime + timedelta(seconds = (30.0*60.0))

    # Find the merged CMT files.
    dir_merged = os.path.join(dir_base, 'input', 'cmts_merged')
    path_merged_regex = os.path.join(dir_merged, 'finite_' + '[0-9]'*5 + '.txt')
    path_merged_list_unsorted = glob(path_merged_regex)
    num_merged = len(path_merged_list_unsorted)

    # Look for matches.
    # Merge the finite source and radiation pattern based on closest time match.
    # 'thresh' is the maximum time separation allowed in seconds.
    thresh = 30.0 
    k = 0
    for i in range(num_merged):

        path_merged = os.path.join(dir_merged, 'point_{:>05d}.txt'.format(i))
        cmt = read_mineos_cmt(path_merged)

        # For given finite event, find time difference with all point events.
        t_diff_list = (np.abs(cmt['datetime_ref'] - pattern_datetime)) 

        # Find finite-point pair with smallest time gap.
        j = np.argmin(t_diff_list)
        t_diff = t_diff_list[j]

        # If time gap is small enough, the match is successful.
        if t_diff.total_seconds() < thresh:

            print(i, 'successful')
            path_pattern_regex = os.path.join(dir_pattern_list_unsorted[j], '*.txt')
            path_pattern_glob = glob(path_pattern_regex)
            assert len(path_pattern_glob) == 1
            path_pattern = path_pattern_glob[0]
            path_pattern_new = os.path.join(dir_merged, 'pattern_{:>05d}.txt'.format(i))
            copyfile(path_pattern, path_pattern_new)

        else:

            print(i, 'unsuccessful')

    return

def main():

    #download_finite_faults()
    #merge_finite_and_point()
    merge_with_radiation_patterns()

    return

if __name__ == '__main__':

    main()
