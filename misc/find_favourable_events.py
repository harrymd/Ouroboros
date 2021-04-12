import argparse
from glob import glob
import os

import matplotlib.pyplot as plt
import numpy as np

# Plotting.
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point

from Ouroboros.common import load_eigenfreq, load_eigenfunc, read_input_file
from Ouroboros.misc.cmt_io import read_mineos_cmt
from Ouroboros.summation.run_summation import associated_Legendre_func_series_no_CS_phase, calculate_azimuth, calculate_epicentral_distance, calculate_source_coefficients_spheroidal, calculate_coeffs_spheroidal, get_eigenfunc, get_Mineos_out_dirs

path_input = '/Users/hrmd_work/Documents/research/stoneley/input/mineos/input_prem_noocean.txt'
cmt_type = 'point2' # 'testing', 'point', 'point2', or 'finite'

def testing_cmt_list():

    Mrr_list    = [ 1.0, 0.0,  0.0, 0.0, 0.0, 0.0]
    Mtt_list    = [ 0.0, 1.0,  1.0, 0.0, 0.0, 0.0]
    Mpp_list    = [ 0.0, 1.0, -1.0, 0.0, 0.0, 0.0]
    Mrt_list    = [ 0.0, 0.0,  0.0, 1.0, 0.0, 0.0]
    Mrp_list    = [ 0.0, 0.0,  0.0, 0.0, 1.0, 0.0]
    Mtp_list    = [ 0.0, 0.0,  0.0, 0.0, 0.0, 1.0]

    num_cmts = len(Mrr_list)
    lat_list    = np.zeros(num_cmts) + 90.0
    lon_list    = np.zeros(num_cmts)
    z_list      = np.zeros(num_cmts)
    tau_list    = np.zeros(num_cmts) + 10.0
    moment_list = np.zeros(num_cmts) + 1.0
    scale_list  = np.zeros(num_cmts) + 1.0E27

    num_cmts = len(lat_list)

    cmt_list = [{'lat_centroid'     : lat_list[i],
                 'lon_centroid'     : lon_list[i],
                 'depth_centroid'   : z_list[i],
                 'half_duration'    : tau_list[i],
                 'scalar_moment'    : moment_list[i],
                 'Mrr'              : Mrr_list[i],
                 'Mtt'              : Mtt_list[i],
                 'Mpp'              : Mpp_list[i],
                 'Mrt'              : Mrt_list[i],
                 'Mrp'              : Mrp_list[i],
                 'Mtp'              : Mtp_list[i],
                 'scale_factor'     : scale_list[i]}
                 for i in range(num_cmts)]

    cmt_list = [cmt_list[i] for i in [0, 3, 2]]

    return cmt_list

def mode_list():

    mode_type = 'S'
    n_list = [ 2,  3,  0]
    l_list = [25, 25, 48]

    return mode_type, n_list, l_list

def run_point():

    # Read input file.
    run_info = read_input_file(path_input)

    # Create mode list.
    mode_type, n_list, l_list = mode_list()
    num_modes = len(n_list)

    # Get frequencies.
    mode_info = load_eigenfreq(run_info, mode_type)
    f_mHz_list = np.zeros(num_modes)
    for i in range(num_modes):
        
        j = np.where((mode_info['n'] == n_list[i]) &
                     (mode_info['l'] == l_list[i]))[0][0]
        f_mHz_list[i] = mode_info['f'][j]
    f_Hz_list = f_mHz_list*1.0E-3
    f_rad_per_s_list = 2.0*np.pi*f_Hz_list

    # Moment tensor list.
    if cmt_type == 'testing':

        cmt_list = testing_cmt_list()
        num_events = len(cmt_list)

    elif cmt_type == 'point':

        dir_event = '/Users/hrmd_work/Documents/research/stoneley/input/global_cmt/'
        file_event_regex = '[0-9]'*5 + '.txt'
        event_names_list_unsorted = glob(os.path.join(dir_event, file_event_regex))
        num_events = len(event_names_list_unsorted)

        num_events = 3

    # Create grid of points.
    #num_pts_lon = 720 
    num_pts_lon = 72 
    num_pts_lat = (num_pts_lon//2) + 1 
    lon_grid_deg = np.linspace(0.0, 360.0, num = num_pts_lon + 1)[:-1]
    lat_grid_deg = np.linspace(-90.0, 90.0, num = num_pts_lat)
    #
    Phi_grid_deg = lon_grid_deg
    Theta_grid_deg = 90.0 - lat_grid_deg
    #
    Phi_grid_rad = np.deg2rad(Phi_grid_deg)
    Theta_grid_rad = np.deg2rad(Theta_grid_deg)
    #
    cosTheta_grid = np.cos(Theta_grid_rad)
    sinTheta_grid = np.sin(Theta_grid_rad)
    #
    cosPhi_grid = np.cos(Phi_grid_rad)
    sinPhi_grid = np.sin(Phi_grid_rad)
    cos2Phi_grid = np.cos(2.0*Phi_grid_rad)
    sin2Phi_grid = np.sin(2.0*Phi_grid_rad)
    sin_cos_Phi_array = np.array([cosPhi_grid, sinPhi_grid,
                            cos2Phi_grid, sin2Phi_grid])

    z_receiver = 0.0 # Units of m.
    pulse_type = 'triangle'
    r_planet = 6371000.0

    l_max = np.max(l_list)
    assert l_max > 2
    Plm_series_grid = np.zeros((num_pts_lat, 3, l_max + 1))
    Plm_prime_series_grid = np.zeros((num_pts_lat, 3, l_max + 1))
    for k in range(num_pts_lat):

        Plm_series_grid[k, :, :], Plm_prime_series_grid[k, :, :] = \
                associated_Legendre_func_series_no_CS_phase(
                        2, l_max, cosTheta_grid[k])

    # Create output array.
    A = np.zeros((num_events, num_modes, num_pts_lon, num_pts_lat))
    
    # Loop over moment tensor list.
    for h in range(num_events):

        print('Event {:>5d} of {:>5d}'.format(h + 1, num_events))
        
        # Get moment tensor information.
        if testing:

            cmt = cmt_list[h]

        else:

            file_event = '{:>05d}.txt'.format(h)
            path_event = os.path.join(dir_event, file_event)
            cmt = read_mineos_cmt(path_event)

        cmt['r_centroid'] = r_planet*1.0E-3 - cmt['depth_centroid']

        # Loop over mode list.
        for i in range(num_modes):
            
            # Unpack.
            n = n_list[i]
            l = l_list[i]
            f_rad_per_s = f_rad_per_s_list[i]

            # Determine eigenfunctions at source and receiver depths.
            eigfunc_source, eigfunc_receiver, r_planet = \
                get_eigenfunc(run_info, mode_type,
                    n,
                    l,
                    f_rad_per_s,
                    cmt['depth_centroid']*1.0E3, # km to m.
                    z_receiver = z_receiver,
                    response_correction_params = None)

            # Excitation coefficients determined by source location.
            source_coeffs = \
                calculate_source_coefficients_spheroidal(
                    l, f_rad_per_s, cmt, eigfunc_source, pulse_type)

            # Loop over longitude grid.
            for j in range(num_pts_lon):

                # Loop over latitude grid.
                for k in range(num_pts_lat):

                    # Overall coefficients including receiver location.
                    coeffs = calculate_coeffs_spheroidal(
                                source_coeffs, eigfunc_receiver, l,
                                sinTheta_grid[k], Plm_series_grid[k, :, l],
                                Plm_prime_series_grid[k, :, l],
                                sin_cos_Phi_array[:, j])

                    A[h, i, j, k] = coeffs['r']
    
    _, dir_out = get_Mineos_out_dirs(run_info)
    path_out = os.path.join(dir_out, 'testing_ratios.npy')
    print('Writing to {:}'.format(path_out))
    np.save(path_out, A)

    return

def run_finite():

    # Read input file.
    run_info = read_input_file(path_input)

    # Create mode list.
    mode_type, n_list, l_list = mode_list()
    num_modes = len(n_list)

    # Get frequencies.
    mode_info = load_eigenfreq(run_info, mode_type)
    f_mHz_list = np.zeros(num_modes)
    for i in range(num_modes):
        
        j = np.where((mode_info['n'] == n_list[i]) &
                     (mode_info['l'] == l_list[i]))[0][0]
        f_mHz_list[i] = mode_info['f'][j]
    f_Hz_list = f_mHz_list*1.0E-3
    f_rad_per_s_list = 2.0*np.pi*f_Hz_list

    # Get number of events.
    dir_event = '/Users/hrmd_work/Documents/research/stoneley/input/finite_faults/'
    file_event_regex = '[0-9]'*5 + '.txt'
    event_names_list_unsorted = glob(os.path.join(dir_event, file_event_regex))
    num_events = len(event_names_list_unsorted)

    num_events = 1

    # Create grid of points.
    num_pts_lon = 720 
    num_pts_lat = (num_pts_lon//2) + 1 
    lon_grid_deg = np.linspace(0.0, 360.0, num = num_pts_lon + 1)[:-1]
    lat_grid_deg = np.linspace(-90.0, 90.0, num = num_pts_lat)

    # Define constants.
    z_receiver = 0.0 # Units of m.
    pulse_type = 'triangle'
    r_planet = 6371000.0
    l_max = np.max(l_list)
    assert l_max > 2

    # Get eigenfunction at receiver locations (the same in each case
    # because it is controlled only by the depth).
    eigfunc_receiver_list = []
    for i in range(num_modes):

        eigfunc_receiver_dict = get_eigenfunc_given_z(run_info, mode_type,
                                    n_list[i], l_list[i], f_rad_per_s_list[i],
                                    z_receiver, r_planet)

        eigfunc_receiver_list.append(eigfunc_receiver_dict)

    # Create output array.
    A = np.zeros((num_events, num_modes, num_pts_lon, num_pts_lat), dtype = np.complex)
    
    # Loop over moment tensor list.
    for h in range(num_events):

        print('Event {:>5d} of {:>5d}'.format(h + 1, num_events))
        
        # Load list of CMT patches.
        file_event = '{:>05d}.txt'.format(h)
        path_event = os.path.join(dir_event, file_event)
        cmt_list, t_offset_list = read_finite(path_event, r_planet*1.0E-3)
        #cmt_list = cmt_list[0:5]
        #t_offset_list = t_offset_list[0:5]

        # Get eigenfunction at source locations.
        num_patches = len(cmt_list)
        eigfunc_source_list = []
        for i in range(num_modes):


            
            eigfunc_source_sublist = []
            for p in range(num_patches):

                eigfunc_source_dict = get_eigenfunc_given_z(run_info, mode_type,
                                            n_list[i], l_list[i], f_rad_per_s_list[i],
                                            cmt_list[p]['depth']*1.0E3, # Convert to m
                                            r_planet)

                eigfunc_source_sublist.append(eigfunc_source_dict)

            eigfunc_source_list.append(eigfunc_source_sublist)

        # Get excitation coefficients determined by source location.
        source_coeffs_list = []
        for i in range(num_modes):

            source_coeffs_sub_list = []
            for p in range(num_patches):

                # Excitation coefficients determined by source location.
                source_coeffs = \
                    calculate_source_coefficients_spheroidal(
                        l_list[i], f_rad_per_s_list[i], cmt_list[p],
                        eigfunc_source_list[i][p], pulse_type)

                source_coeffs_sub_list.append(source_coeffs)

            source_coeffs_list.append(source_coeffs_sub_list)
        
        # Get angular coordinates.
        Theta_deg_array = np.zeros((num_pts_lon, num_pts_lat, num_patches))
        Phi_deg_array = np.zeros((num_pts_lon, num_pts_lat, num_patches))
        cosTheta_array = np.zeros((num_pts_lon, num_pts_lat, num_patches))
        for j in range(num_pts_lon):

            print('lon: ', j)

            for k in range(num_pts_lat):

                for p in range(num_patches):

                    Theta_deg_array[j, k, p], cosTheta_array[j, k, p] = \
                        calculate_epicentral_distance(
                            cmt_list[p]['longitude'], cmt_list[p]['latitude'],
                            lon_grid_deg[j], lat_grid_deg[k],
                            io_in_degrees = True)

                    Phi_deg_array[j, k, p] = calculate_azimuth(
                            cmt_list[p]['longitude'], cmt_list[p]['latitude'],
                            lon_grid_deg[j], lat_grid_deg[k],
                            io_in_degrees = True)

        # Get Legendre functions.
        Phi_rad_array = np.deg2rad(Phi_deg_array)
        Theta_rad_array = np.deg2rad(Theta_deg_array)
        sinTheta_array = np.sin(Theta_rad_array)
        cosPhi_array = np.cos(Phi_rad_array)
        sinPhi_array = np.sin(Phi_rad_array)
        cos2Phi_array = np.cos(2.0*Phi_rad_array)
        sin2Phi_array = np.sin(2.0*Phi_rad_array)
        #
        sin_cos_Phi_array = np.array([cosPhi_array, sinPhi_array,
                            cos2Phi_array, sin2Phi_array])
        #
        Plm_series_grid = np.zeros((num_pts_lon, num_pts_lat, num_patches, 3, l_max + 1))
        Plm_prime_series_grid = np.zeros((num_pts_lon, num_pts_lat, num_patches, 3, l_max + 1))
        for j in range(num_pts_lon):

            for k in range(num_pts_lat):

                for p in range(num_patches):

                    Plm_series_grid[j, k, p, :, :], Plm_prime_series_grid[j, k, p, :, :] = \
                                associated_Legendre_func_series_no_CS_phase(
                                        2, l_max, cosTheta_array[j, k, p])

        # Loop over modes.
        for i in range(num_modes):
            
            l = l_list[i]
            print('Mode {:>5d} or {:>5d}'.format(i + 1, num_modes))

            # Loop over longitude grid.
            for j in range(num_pts_lon):

                # Loop over latitude grid.
                for k in range(num_pts_lat):

                    # Loop over patches.
                    for p in range(num_patches):

                        # Overall coefficients including receiver location.
                        coeffs = calculate_coeffs_spheroidal(
                                    source_coeffs_list[i][p],
                                    eigfunc_receiver_list[i],
                                    l,
                                    sinTheta_array[j, k, p],
                                    Plm_series_grid[j, k, p, :, l],
                                    Plm_prime_series_grid[j, k, p, :, l],
                                    sin_cos_Phi_array[:, j, k, p])

                        # Correct phase.
                        A_patch = coeffs['r']
                        phase = f_rad_per_s_list[i]*t_offset_list[p]
                        A_patch_phased = A_patch*np.exp(1.0j*phase)

                        # Store.
                        A[h, i, j, k] = A[h, i, j, k] + A_patch_phased

    
    # Save.
    _, dir_out = get_Mineos_out_dirs(run_info)
    path_out = os.path.join(dir_out, 'testing_ratios_finite.npy')
    print('Writing to {:}'.format(path_out))
    np.save(path_out, A)

    return

def run_point_new():

    # Read input file.
    run_info = read_input_file(path_input)

    # Create mode list.
    mode_type, n_list, l_list = mode_list()
    num_modes = len(n_list)

    # Get frequencies.
    mode_info = load_eigenfreq(run_info, mode_type)
    f_mHz_list = np.zeros(num_modes)
    for i in range(num_modes):
        
        j = np.where((mode_info['n'] == n_list[i]) &
                     (mode_info['l'] == l_list[i]))[0][0]
        f_mHz_list[i] = mode_info['f'][j]
    f_Hz_list = f_mHz_list*1.0E-3
    f_rad_per_s_list = 2.0*np.pi*f_Hz_list

    # Get number of events.
    dir_event = '/Users/hrmd_work/Documents/research/stoneley/input/cmts_merged/'
    file_event_regex = 'point_' + '[0-9]'*5 + '.txt'
    event_names_list_unsorted = glob(os.path.join(dir_event, file_event_regex))
    num_events = len(event_names_list_unsorted)
    
    num_events = 1 

    # Create grid of points.
    num_pts_lon = 720 
    #num_pts_lon = 72 
    num_pts_lat = (num_pts_lon//2) + 1 
    lon_grid_deg = np.linspace(0.0, 360.0, num = num_pts_lon + 1)[:-1]
    lat_grid_deg = np.linspace(-90.0, 90.0, num = num_pts_lat)

    # Define constants.
    z_receiver = 0.0 # Units of m.
    pulse_type = 'triangle'
    r_planet = 6371000.0
    l_max = np.max(l_list)
    assert l_max > 2

    # Get eigenfunction at receiver locations (the same in each case
    # because it is controlled only by the depth).
    eigfunc_receiver_list = []
    for i in range(num_modes):

        eigfunc_receiver_dict = get_eigenfunc_given_z(run_info, mode_type,
                                    n_list[i], l_list[i], f_rad_per_s_list[i],
                                    z_receiver, r_planet)

        eigfunc_receiver_list.append(eigfunc_receiver_dict)

    # Create output array.
    A = np.zeros((num_events, num_modes, num_pts_lon, num_pts_lat))
    
    # Loop over moment tensor list.
    for h in range(num_events):

        print('Event {:>5d} of {:>5d}'.format(h + 1, num_events))
        
        # Load list of CMT patches.
        file_event = 'point_{:>05d}.txt'.format(h)
        path_event = os.path.join(dir_event, file_event)
        cmt = read_mineos_cmt(path_event)
        cmt['r_centroid'] = r_planet*1.0E-3 - cmt['depth_centroid']

        # Get eigenfunction at source locations.
        eigfunc_source_list = []
        for i in range(num_modes):
            
            eigfunc_source_dict = get_eigenfunc_given_z(run_info, mode_type,
                                            n_list[i], l_list[i], f_rad_per_s_list[i],
                                            cmt['depth_centroid']*1.0E3, # Convert to m
                                            r_planet)

            eigfunc_source_list.append(eigfunc_source_dict)

        # Get excitation coefficients determined by source location.
        source_coeffs_list = []
        for i in range(num_modes):

            # Excitation coefficients determined by source location.
            source_coeffs = \
                calculate_source_coefficients_spheroidal(
                    l_list[i], f_rad_per_s_list[i], cmt,
                    eigfunc_source_list[i], pulse_type)

            source_coeffs_list.append(source_coeffs)
        
        # Get angular coordinates.
        Theta_deg_array = np.zeros((num_pts_lon, num_pts_lat))
        Phi_deg_array = np.zeros((num_pts_lon, num_pts_lat))
        cosTheta_array = np.zeros((num_pts_lon, num_pts_lat))
        for j in range(num_pts_lon):

            for k in range(num_pts_lat):

                Theta_deg_array[j, k], cosTheta_array[j, k] = \
                    calculate_epicentral_distance(
                        cmt['lon_centroid'], cmt['lat_centroid'],
                        lon_grid_deg[j], lat_grid_deg[k],
                        io_in_degrees = True)

                Phi_deg_array[j, k] = calculate_azimuth(
                        cmt['lon_centroid'], cmt['lat_centroid'],
                        lon_grid_deg[j], lat_grid_deg[k],
                        io_in_degrees = True)

        # Get Legendre functions.
        Phi_rad_array = np.deg2rad(Phi_deg_array)
        Theta_rad_array = np.deg2rad(Theta_deg_array)
        sinTheta_array = np.sin(Theta_rad_array)
        cosPhi_array = np.cos(Phi_rad_array)
        sinPhi_array = np.sin(Phi_rad_array)
        cos2Phi_array = np.cos(2.0*Phi_rad_array)
        sin2Phi_array = np.sin(2.0*Phi_rad_array)
        #
        sin_cos_Phi_array = np.array([cosPhi_array, sinPhi_array,
                            cos2Phi_array, sin2Phi_array])
        #
        Plm_series_grid = np.zeros((num_pts_lon, num_pts_lat, 3, l_max + 1))
        Plm_prime_series_grid = np.zeros((num_pts_lon, num_pts_lat, 3, l_max + 1))
        for j in range(num_pts_lon):

            for k in range(num_pts_lat):

                Plm_series_grid[j, k, :, :], Plm_prime_series_grid[j, k, :, :] = \
                            associated_Legendre_func_series_no_CS_phase(
                                    2, l_max, cosTheta_array[j, k])

        # Loop over modes.
        for i in range(num_modes):
            
            l = l_list[i]

            # Loop over longitude grid.
            for j in range(num_pts_lon):

                # Loop over latitude grid.
                for k in range(num_pts_lat):

                    # Overall coefficients including receiver location.
                    coeffs = calculate_coeffs_spheroidal(
                                source_coeffs_list[i],
                                eigfunc_receiver_list[i],
                                l,
                                sinTheta_array[j, k],
                                Plm_series_grid[j, k, :, l],
                                Plm_prime_series_grid[j, k, :, l],
                                sin_cos_Phi_array[:, j, k])

                    # Store.
                    A[h, i, j, k] = coeffs['r']

    
    # Save.
    _, dir_out = get_Mineos_out_dirs(run_info)
    path_out = os.path.join(dir_out, 'testing_ratios_point2.npy')
    print('Writing to {:}'.format(path_out))
    np.save(path_out, A)

    return

def read_finite(path_cmt, r_planet_km):
    
    cmt_list = []
    t_offset_list = []
    with open(path_cmt, 'r') as in_id:
        
        line = in_id.readline()
        while line:
                    
            point_header = line
            name_str = in_id.readline().split()[-1] 
            t_offset = float(in_id.readline().split()[-1])
            half_duration = float(in_id.readline().split()[-1])
            latitude = float(in_id.readline().split()[-1])
            longitude = float(in_id.readline().split()[-1])
            depth = float(in_id.readline().split()[-1])
            Mrr = float(in_id.readline().split()[-1])
            Mtt = float(in_id.readline().split()[-1])
            Mpp = float(in_id.readline().split()[-1])
            Mrt = float(in_id.readline().split()[-1])
            Mrp = float(in_id.readline().split()[-1])
            Mtp = float(in_id.readline().split()[-1])

            t_offset_list.append(t_offset)

            cmt = {'half_duration' : half_duration,
                    'latitude'      : latitude,
                    'longitude'     : longitude,
                    'depth'         : depth,
                    'r_centroid'    : r_planet_km - depth,
                    'Mrr'           : Mrr,
                    'Mtt'           : Mtt,
                    'Mpp'           : Mpp,
                    'Mrt'           : Mrt,
                    'Mrp'           : Mrp,
                    'Mtp'           : Mtp,
                    'scale_factor'  : 1.0}

            cmt_list.append(cmt)
            
            line = in_id.readline()

    t_offset_list = np.array(t_offset_list)
    
    return cmt_list, t_offset_list

def get_eigenfunc_given_z(run_info, mode_type, n, l, f_rad_per_s, z_query, r_planet, response_correction_params = None):

    norm_args = {'norm_func' : 'DT', 'units' : 'SI', 'omega' : f_rad_per_s}

    # Load eigenfunction information for this mode.
    if mode_type in ['R', 'S']:

        eigenfunc_dict = load_eigenfunc(run_info, mode_type, n, l, norm_args = norm_args)
        
    else:
        
        print('Mode summation only implemented for R and S modes.')
        raise NotImplementedError

    # Convert r to depth.
    z = r_planet - eigenfunc_dict['r']
    
    # Find the eigenfunctions and gradients at the depth of the source
    # and dict.
    eigfunc_dict = dict()
    if mode_type == 'S':

        if response_correction_params is not None:
        
            keys = ['U', 'V', 'Up', 'Vp', 'P']

        else:

            keys = ['U', 'V', 'Up', 'Vp']

    elif mode_type == 'R':

        keys = ['U', 'Up']

    else:
        
        print('Not implented yet for T modes.')
        raise NotImplementedError

    # Reverse everything for interpolation.
    z = z[::-1]
    for key in keys:

        eigenfunc_dict[key] = eigenfunc_dict[key][::-1]
    
    assert z[-1] > z[0], 'Depth must be increasing for np.interp'
    for key in keys:

        eigfunc_dict[key] = np.interp(z_query, z, eigenfunc_dict[key])

    # Apply seismometer response correction (if requested).
    if response_correction_params is not None:
        
        assert mode_type in ['R', 'S']
        # Unpack response correction parameters.
        g = response_correction_params['g']
        #
        U = eigfunc_dict['U']
        if mode_type == 'R':

            P = 0.0

        elif mode_type == 'S':

            P = eigfunc_dict['P']

        U_free, U_pot, V_tilt, V_pot = seismometer_response_correction(
                                        l, f_rad_per_s, r_planet, g,
                                        U, P)
        eigfunc_dict['U'] = U + U_free + U_pot

        if mode_type == 'S':

            V = eigfunc_dict['V']
            eigfunc_dict['V'] = V + V_tilt + V_pot

    return eigfunc_dict

def plot(i_event, i_mode, i_second_mode = None):

    # Choose event and mode.
    h = i_event
    i = i_mode

    # Read input file.
    run_info = read_input_file(path_input)

    # Get mode information.
    mode_type, n_list, l_list = mode_list()
    n = n_list[i]
    l = l_list[i]

    if i_second_mode is not None:

        n2 = n_list[i_second_mode]
        l2 = l_list[i_second_mode]

    # Get moment tensor information. 
    if cmt_type == 'testing':

        cmt_list = testing_cmt_list()
        cmt = cmt_list[h]

    elif cmt_type == 'point':

        dir_event = '/Users/hrmd_work/Documents/research/stoneley/input/global_cmt/'
        file_event = '{:>05d}.txt'.format(h)
        path_event = os.path.join(dir_event, file_event)
        cmt = read_mineos_cmt(path_event)

    elif cmt_type == 'point2':

        dir_event = '/Users/hrmd_work/Documents/research/stoneley/input/cmts_merged/'
        file_event = 'point_{:>05d}.txt'.format(h)
        path_event = os.path.join(dir_event, file_event)
        cmt = read_mineos_cmt(path_event)
        
        show_radiation_line = True
        if show_radiation_line:

            from Ouroboros.summation.sum_over_branches import gc_path_points_through_antipode, read_radiation_pattern

            # Read radiation pattern.
            path_radiation_pattern = os.path.join(dir_event, 'pattern_{:>05d}.txt'.format(i))
            radiation_info = read_radiation_pattern(path_radiation_pattern)

            az = radiation_info['azi_span']
            amp = radiation_info['band_0']['amp_rayl']

            i_max_amp = np.argmax(amp)
            az_max_amp = az[i_max_amp]
            
            gc_points = gc_path_points_through_antipode(cmt['lon_centroid'],
                            cmt['lat_centroid'], az_max_amp, half_n = 100,
                            geodesic_object = 'sphere_earth')

        else:

            gc_points = None

    elif cmt_type == 'finite':
        
        pass

    # Create title.
    mode_str = '$_{{{:d}}}${:}$_{{{:d}}}$'.format(n, mode_type, l)
    if cmt_type in ['testing', 'point', 'point2']:

        cmt_str = '$rr$ = {:>.1f}, $\\theta \\theta$ = {:>.1f}, $\phi\phi$ = {:>.1f}, $r\\theta$ = {:>.1f}, ${{r\phi}}$ = {:>.1f}, $\\theta\phi$ = {:>.1f}'.format(
                    cmt['Mrr'], cmt['Mtt'], cmt['Mpp'], cmt['Mrt'], cmt['Mrp'], cmt['Mtp'])

    else:

        cmt_str = 'Complex source'

    if i_second_mode is None:

        title = '{:}\n{:}'.format(mode_str, cmt_str)

    else:

        mode_str2 = '$_{{{:d}}}${:}$_{{{:d}}}$'.format(n2, mode_type, l2)
        title = 'Ratio {:} / {:} \n{:}'.format(mode_str, mode_str2, cmt_str)

    # Load array.
    _, dir_out = get_Mineos_out_dirs(run_info)
    if cmt_type in ['testing', 'point']:

        name_out = 'testing_ratios.npy'

    elif cmt_type == 'point2':

        name_out = 'testing_ratios_point2.npy'

    elif cmt_type == 'finite':

        name_out = 'testing_ratios_finite.npy'

    path_out = os.path.join(dir_out, name_out)
    print('Reading {:}'.format(path_out))
    A_array = np.load(path_out)
    A_grid = A_array[h, i, :, :]

    if cmt_type == 'finite':

        A_grid = np.real(A_grid)

    if i_second_mode is not None:

        A_grid2 = A_array[h, i_second_mode, :, :]

        if cmt_type == 'finite':

            A_grid = np.real(A_grid2)

    else:

        A_grid2 = None

    # Create figure.
    if i_second_mode is None:

        name_fig = 'coeff_mode_{:>03d}_{:}_{:>03d}_event_{:>03d}.png'.format(n, mode_type, l, h)

    else:

        name_fig ='ratio_{:>03d}_{:}_{:>03d}_over_{:>03d}_{:}_{:>03d}_event_{:>03d}.png'.format(n, mode_type, l, n2, mode_type, l2, h)

    path_fig = os.path.join(dir_out, name_fig)

    # Infer lat, lon grids.
    num_pts_lon, num_pts_lat = A_grid.shape
    lon_grid_deg = np.linspace(0.0, 360.0, num = num_pts_lon + 1)[:-1]
    lat_grid_deg = np.linspace(-90.0, 90.0, num = num_pts_lat)
    #
    plot_coeffs(lon_grid_deg, lat_grid_deg, A_grid, coeffs2 = A_grid2, title = title, path_fig = path_fig, gc_points = gc_points)
    
    return

def plot_coeffs(lon, lat, coeffs, coeffs2 = None, title = None, path_fig = None, gc_points = None):
    
    coeffs = np.abs(coeffs)

    if coeffs2 is not None:
        
        coeffs2 = np.abs(coeffs2)
        ratio = coeffs/coeffs2
        var = ratio

    else:

        var = coeffs

    #projection = ccrs.Mollweide()
    projection = ccrs.Robinson()

    fig = plt.figure(figsize = (8.5, 4.5))
    ax = fig.add_subplot(1, 1, 1, projection = projection)

    #coeffs_max = np.max(np.abs(coeffs.flatten()))
    #vmin, vmax = 0.3*coeffs_max*np.array([-1.0, 1.0])
    #levels = np.linspace(vmin, vmax, num = 11)
    #c_norm = mpl.colors.Normalize(vmin = -0.3*coeffs_max, vmax = 0.3*coeffs_max)
    #cmap = mpl.cm.get_cmap('seismic')
    
    i_high_lat = np.where(np.abs(lat) > 70.0)[0]
    var_masked = var.copy()
    var_masked[:, i_high_lat] = 0.0
    if coeffs2 is None:
        
        var_max = np.max(np.abs(var_masked.flatten()))

    else:

        var_max = 10.0*np.median(np.abs(var_masked.flatten())) 
        
    vmin = 0.0
    #vmax = 10.0**(np.ceil(np.log10(var_max)))
    vmax = var_max
    levels = np.linspace(vmin, vmax, num = 11)
    if coeffs2 is None:

        c_norm = mpl.colors.Normalize(vmin = vmin, vmax = vmax)
        extend = 'max'
        label = 'Coefficient'

    else:

        c_norm = mpl.colors.LogNorm(vmin = vmax*1.0E-3, vmax = vmax)
        extend = 'both'
        label = 'Ratio'

    cmap = mpl.cm.get_cmap('magma')
    
    var, lon = add_cyclic_point(var.T, coord = lon)
    #h = ax.contourf(lon, lat, var,
    #            levels = levels,
    #            transform = ccrs.PlateCarree(central_longitude = 0.0),
    #            cmap = cmap,
    #            norm = c_norm,
    #            extend = 'max')
    h = ax.pcolormesh(lon, lat, var,
                transform = ccrs.PlateCarree(central_longitude = 0.0),
                cmap = cmap,
                norm = c_norm)
    
    #Lon, Lat = np.meshgrid(lon, lat)
    #ax.scatter(Lon, Lat, transform = ccrs.PlateCarree(central_longitude = 0.0), s = 1, c = 'white')
    
    if gc_points is not None:
        
        #ax.scatter(gc_points[1, :], gc_points[2, :], transform = ccrs.PlateCarree())
        ax.plot(gc_points[1, :], gc_points[2, :], marker = '.', transform = ccrs.PlateCarree())
        print(gc_points.shape)

    ax.set_global()

    divider = make_axes_locatable(ax)
    ax_cb = divider.new_horizontal(size="2%", pad=0.05, axes_class=plt.Axes)
    fig.add_axes(ax_cb)
    fig.colorbar(h, cax=ax_cb, label = label, extend = extend)

    if title is not None:
        
        font_size_title = 14
        ax.set_title(title, fontsize = font_size_title)

    #ax.coastlines()
    ax.add_feature(cartopy.feature.COASTLINE, color = 'grey')

    fig.canvas.draw()
    plt.tight_layout()

    if path_fig is not None:
        
        print('Saving to {:}'.format(path_fig))
        plt.savefig(path_fig, dpi = 300)

    plt.show()

    return

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices = ['run', 'plot'], help = "Choose what to do.")
    parser.add_argument("--modes", type = int, nargs = '*', help = "Which mode to plot. If two modes are given, plot ratio of first over second.")
    parser.add_argument("--event", type = int, help = "Which event to plot.")
    args = parser.parse_args()
    task = args.task
    modes = args.modes
    event = args.event

    if task == 'run':

        assert modes is None, 'Flag --modes is for use with \'plot\' option.'
        assert event is None, 'Flag --event is for use with \'plot\' option.'

    elif task == 'plot':

        assert modes is not None, 'When plotting, must specify one or two modes to plot with --modes flag.'
        assert event is not None, 'When plotting, must specify an event to plot with --event flag.'
        assert len(modes) in [1, 2], 'You specified {:} modes, should be 1 or 2.'.format(len(modes))

        i_mode = modes[0]
        if len(modes) == 1:

            i_second_mode = None

        else:

            i_second_mode = modes[1]

    if task == 'run':

        if cmt_type == 'point':

            run_point()

        elif cmt_type == 'point2':

            run_point_new()

        elif cmt_type == 'finite':

            run_finite()

    elif task == 'plot':

        plot(event, i_mode, i_second_mode = i_second_mode)

    return

if __name__ == '__main__':

    main()
