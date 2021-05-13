'''
Tools for working with CMT (centroid moment tensor) input and output.
'''

import argparse
import datetime
import os

import numpy as np

def ndk_to_mineos_cmt(path_ndk, path_mineos_cmt, dt = 1.0):
    '''
    Converts a centroid moment tensor (CMT) file from NDK format
    https://www.ldeo.columbia.edu/~gcmt/projects/CMT/catalog/allorder.ndk_explained
    into Mineos format (see Mineos manual, section 3.3.2).

    dt  Sets the time interval (in seconds) for Mineos synthetic seismograms.
    '''

    # Read the NDK file.
    cmt_info = read_ndk(path_ndk)

    # Write the Mineos file.
    write_mineos_cmt(path_mineos_cmt, cmt_info, dt = dt)

    return

def read_ndk(path_ndk):
    '''
    Reads a centroid moment tensor in NDK format.
    https://www.ldeo.columbia.edu/~gcmt/projects/CMT/catalog/allorder.ndk_explained
    '''

    with open(path_ndk, 'r') as in_id:

        # First line: Hypocenter line.
        line = in_id.readline()

        catalog = line[0:3]
        date_str = line[5:15]
        time_str = line[16:26]
        lat_hypo_str = line[27:33]
        lon_hypo_str = line[34:41]
        depth_hypo_str = line[42:47]
        mag_strs = line[48:55]
        loc_str = line[56:80]

        # Second line: CMT info (1).
        line = in_id.readline()

        cmt_name = line[0:15]
        data_list_str = line[17:61]
        src_type_str = line[62:68]
        moment_rate_func_str = line[69:80]
        half_duration_str = moment_rate_func_str.split()[1]

        # Third line: CMT info (2).
        line = in_id.readline()

        depth_type_str = line[59:63]
        timestamp_str = line[64:80]

        line = line.split()

        t_centroid_str = line[1]
        t_centroid_uncertainty_str = line[2]
        lat_centroid_str = line[3]
        lat_centroid_uncertainty_str = line[4]
        lon_centroid_str = line[5]
        lon_centroid_uncertainty_str = line[6]
        depth_centroid_str = line[7]
        depth_centroid_uncertainty_str = line[8]

        # Fourth line: CMT info (3).
        line = in_id.readline()

        line = line.split()
        exponent_str = line[0]
        Mrr_str = line[1]
        Mtt_str = line[3]
        Mpp_str = line[5]
        Mrt_str = line[7]
        Mrp_str = line[9]
        Mtp_str = line[11]
        #
        Mrr_sig_str = line[2]
        Mtt_sig_str = line[4]
        Mpp_sig_str = line[6]
        Mrt_sig_str = line[8]
        Mrp_sig_str = line[10]
        Mtp_sig_str = line[12]

        # Fifth line: CMT info (4)
        line = in_id.readline()

        line = line.split()
        version = line[0]
        eigval_0_str = line[1]
        eigval_1_str = line[4]
        eigval_2_str = line[7]
        #
        plunge_0_str = line[2]
        plunge_1_str = line[5]
        plunge_2_str = line[8]
        #
        azimuth_0_str = line[3]
        azimuth_1_str = line[6]
        azimuth_2_str = line[9]
        #
        scalar_moment_str = line[10]
        #
        strike_0_str = line[11]
        dip_0_str = line[12]
        rake_0_str = line[13]
        #
        strike_1_str = line[14]
        dip_1_str = line[15]
        rake_1_str = line[16]

    # Convert to correct types.

    # Convert date and time strings to datetime object.
    hhmm_str = time_str[0:5]
    ss_s_str = time_str[6:]
    secs = float(ss_s_str)
    secs_int = int(np.floor(secs))
    microsecs_int = int(round(1.0E6*(secs - secs_int)))

    # Global CMT catalog has some cases where secs = 60.
    if secs_int == 60:
        
        secs_int = 0
        hours_int = int(hhmm_str[0:2])
        minutes_int = int(hhmm_str[3:])
        minutes_int = minutes_int + 1
        hhmm_str = '{:>02d}:{:>02d}'.format(hours_int, minutes_int)

    hhmmss_str = '{:}:{:>02d}'.format(hhmm_str, secs_int) 

    assert microsecs_int < 1000000

    datetime_str = '{:} {:} {:>06d}'.format(date_str, hhmmss_str, microsecs_int)
    datetime_ref = datetime.datetime.strptime(datetime_str, '%Y/%m/%d %H:%M:%S %f')

    # Convert hypocentre coords to floats.
    lat_hypo = float(lat_hypo_str)
    lon_hypo = float(lon_hypo_str)
    depth_hypo = float(depth_hypo_str)

    #
    half_duration = float(half_duration_str)

    # Convert centroid information to floats.
    t_centroid = float(t_centroid_str)
    lat_centroid = float(lat_centroid_str)
    lon_centroid = float(lon_centroid_str)
    depth_centroid = float(depth_centroid_str)
    #
    t_centroid_uncertainty = float(t_centroid_uncertainty_str)
    lat_centroid_uncertainty = float(lat_centroid_uncertainty_str)
    lon_centroid_uncertainty = float(lon_centroid_uncertainty_str)
    depth_centroid_uncertainty = float(depth_centroid_uncertainty_str)

    # Convert moment tensoir components to floats.
    exponent = int(exponent_str)
    #
    Mrr = float(Mrr_str)
    Mtt = float(Mtt_str)
    Mpp = float(Mpp_str)
    Mrt = float(Mrt_str)
    Mrp = float(Mrp_str)
    Mtp = float(Mtp_str)
    #
    Mrr_sig = float(Mrr_sig_str)
    Mtt_sig = float(Mtt_sig_str)
    Mpp_sig = float(Mpp_sig_str)
    Mrt_sig = float(Mrt_sig_str)
    Mrp_sig = float(Mrp_sig_str)
    Mtp_sig = float(Mtp_sig_str)

    # Moment tensor principal-axis information.
    eigval_0 = float(eigval_0_str)
    eigval_1 = float(eigval_1_str)
    eigval_2 = float(eigval_2_str)
    #
    plunge_0 = float(plunge_0_str)
    plunge_1 = float(plunge_1_str)
    plunge_2 = float(plunge_2_str)
    #
    azimuth_0 = float(azimuth_0_str)
    azimuth_1 = float(azimuth_1_str)
    azimuth_2 = float(azimuth_2_str)
    #
    scalar_moment = float(scalar_moment_str)
    #
    strike_0 = float(strike_0_str)
    dip_0 = float(dip_0_str)
    rake_0 = float(rake_0_str)
    #
    strike_1 = float(strike_1_str)
    dip_1 = float(dip_1_str)
    rake_1 = float(rake_1_str)

    # Store in a dictionary.
    cmt_info = dict()
    #
    cmt_info['lat_hypo'] = lat_hypo
    cmt_info['lon_hypo'] = lon_hypo
    cmt_info['depth_hypo'] = depth_hypo
    cmt_info['datetime_ref'] = datetime_ref
    #
    cmt_info['half_duration'] = half_duration
    #
    cmt_info['t_centroid'] = t_centroid
    cmt_info['lat_centroid'] = lat_centroid
    cmt_info['lon_centroid'] = lon_centroid
    cmt_info['depth_centroid'] = depth_centroid
    #
    cmt_info['t_centroid_uncertainty'] = t_centroid_uncertainty
    cmt_info['lat_centroid_uncertainty'] = lat_centroid_uncertainty
    cmt_info['lon_centroid_uncertainty'] = lon_centroid_uncertainty
    cmt_info['depth_centroid_uncertainty'] = depth_centroid_uncertainty
    #
    cmt_info['exponent'] = exponent
    #
    cmt_info['Mrr'] = Mrr
    cmt_info['Mtt'] = Mtt
    cmt_info['Mpp'] = Mpp
    cmt_info['Mrt'] = Mrt
    cmt_info['Mrp'] = Mrp
    cmt_info['Mtp'] = Mtp
    #
    cmt_info['Mrr_sig'] = Mrr_sig
    cmt_info['Mtt_sig'] = Mtt_sig
    cmt_info['Mpp_sig'] = Mpp_sig
    cmt_info['Mrt_sig'] = Mrt_sig
    cmt_info['Mrp_sig'] = Mrp_sig
    cmt_info['Mtp_sig'] = Mtp_sig
    #
    cmt_info['eigval_0'] = eigval_0
    cmt_info['eigval_1'] = eigval_1
    cmt_info['eigval_2'] = eigval_2
    #
    cmt_info['plunge_0'] = plunge_0
    cmt_info['plunge_1'] = plunge_1
    cmt_info['plunge_2'] = plunge_2
    #
    cmt_info['azimuth_0'] = azimuth_0
    cmt_info['azimuth_1'] = azimuth_1
    cmt_info['azimuth_2'] = azimuth_2
    #
    cmt_info['scalar_moment'] = scalar_moment
    #
    cmt_info['strike_0'] = strike_0
    cmt_info['dip_0'] = dip_0
    cmt_info['rake_0'] = rake_0
    #
    cmt_info['strike_1'] = strike_1
    cmt_info['dip_1'] = dip_1
    cmt_info['rake_1'] = rake_1

    return cmt_info

def write_mineos_cmt(path_mineos_cmt, cmt_info, dt = 1.0):
    '''
    Writes a Mineos CMT file from a dictionary containing the relevant
    variables.
    For specification of Mineos CMT file, see section 3.3.2 of the Mineos
    manual.

    dt Time step (seconds), used by Mineos for synthetic seismograms.
    '''

    # Convert date to string.
    year = cmt_info['datetime_ref'].year

    date_str_no_secs = cmt_info['datetime_ref'].strftime('%Y %j %H %M')
    secs = cmt_info['datetime_ref'].second + 1.0E-6*cmt_info['datetime_ref'].microsecond
    sec_str = '{:>5.2f}'.format(secs)
    date_str = '{:} {:}'.format(date_str_no_secs, sec_str)

    # Get the scalar moment with correct exponent as a float.
    scale_factor = 10.0**(cmt_info['exponent'])
    scalar_moment = cmt_info['scalar_moment']*scale_factor

    # Get the event ID string.
    ev_id = os.path.basename(path_mineos_cmt).split('.')[0]
    if len(ev_id) > 8:
        ev_id = ev_id[0:8]
    elif len(ev_id) < 8:
        ev_id = ev_id.rjust(8, '_')
    
    # Write the file (it is just one line).
    line = '{:} {:} {:>6.2f} {:>7.2f} {:>6.2f} {:>9.5e} {:>6.2f} {:>7.2e} {:>7.2f} {:>7.2f} {:>7.2f} {:>7.2f} {:>7.2f} {:>7.2f} {:>7.2e} {:>3.0f} {:>3.0f} {:>3.0f} {:>3.0f} {:>3.0f} {:>3.0f}'.format(ev_id, date_str, cmt_info['lat_centroid'], cmt_info['lon_centroid'], cmt_info['depth_centroid'], dt, cmt_info['half_duration'], scalar_moment, cmt_info['Mrr'], cmt_info['Mtt'], cmt_info['Mpp'], cmt_info['Mrt'], cmt_info['Mrp'], cmt_info['Mtp'], scale_factor, cmt_info['strike_0'], cmt_info['dip_0'], cmt_info['azimuth_0'], cmt_info['strike_1'], cmt_info['dip_1'], cmt_info['azimuth_1'])
    
    with open(path_mineos_cmt, 'w') as out_id:

        out_id.write(line)

    return

def read_mineos_cmt(path_mineos_cmt):
    '''
    Reads a CMT file written in Mineos format (see the Mineos manual, section
    3.3.2).
    '''

    # Read the file (it is only one line).
    with open(path_mineos_cmt, 'r') as in_id:

        data_strs = in_id.readline().split()

    # Parse the lines into a dictionary of CMT information.
    cmt_info = dict()

    # Read the event ID (string).
    cmt_info['ev_id'] = data_strs[0]

    # Read the date and time (datetime object)
    date_str_fmt = '%Y %j %H %M %S.%f'
    date_time_str = ' '.join(data_strs[1:5]) + ' ' + '{:>09.6f}'.format(float(data_strs[5]))
    datetime_ref = datetime.datetime.strptime(date_time_str, date_str_fmt)
    cmt_info['datetime_ref'] = datetime_ref

    # Read the remaining information (floats).
    keys = ['lat_centroid', 'lon_centroid', 'depth_centroid', 'dt',
            'half_duration', 'scalar_moment', 'Mrr', 'Mtt', 'Mpp', 'Mrt',
            'Mrp', 'Mtp', 'scale_factor', 'strike_0', 'dip_0', 'azimuth_0',
            'strike_1', 'dip_1', 'azimuth_1']
    #
    for i, key in enumerate(keys):

        j = i + 6 
        cmt_info[key] = float(data_strs[j])

    return cmt_info

def main():
    
    # Read input arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_ndk", help = "Path to CMT file in .ndk format.")
    parser.add_argument("--path_mineos", help = "Path to output CMT file in Mineos format.")
    parser.add_argument("--mineos_dt", type = float, help = "Time step (s) to be written to Mineos-format CMT file.")
    args = parser.parse_args()
    path_ndk = args.path_ndk
    path_mineos = args.path_mineos
    mineos_dt = args.mineos_dt
    default_mineos_dt = 1.0
    if mineos_dt is None:

        mineos_dt = default_mineos_dt

    # Convert from NDK format to Mineos format.
    ndk_to_mineos_cmt(path_ndk, path_mineos, dt = mineos_dt)

    return

if __name__ == '__main__':

    main()
