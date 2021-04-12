import argparse
import copy
import datetime
import os
from string import Formatter

from matplotlib.animation import FuncAnimation
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from obspy.core import read as obspy_read
from obspy.core.utcdatetime import UTCDateTime
from obspy.signal.filter import lowpass, bandpass
from obspy.core.stream import Stream
from obspy.core.trace import Trace

import pandas
#from scipy.interpolate import interp2d
from scipy.interpolate import griddata

from Ouroboros.common import (filter_mode_list, load_eigenfreq,
                            get_Mineos_out_dirs, get_Ouroboros_summation_out_dirs,
                            read_channel_file, read_input_file, read_summation_input_file)
from Ouroboros.misc.cmt_io import read_mineos_cmt
from Ouroboros.plot.plot_gather import plot_gather

# Define radii.
r_srf = 6371.0
r_cmb = 3480.0
r_icb = 1221.5

def read_even_grid(path_grid):
    
    data = np.loadtxt(path_grid, usecols = (1, 2, 3, 4))
    lon = data[:, 0]
    lat = data[:, 1]
    delta = data[:, 2]
    r = data[:, 3]

    check_plot = False 
    if check_plot:

        x = r*np.cos(delta)
        y = r*np.sin(delta)

        fig = plt.figure(figsize = (10.0, 10.0))
        ax  = plt.gca()
        ax.scatter(x, y, alpha = 0.5)
        ax.set_aspect(1.0)

        plt.show()

    return lon, lat, delta, r

def add_epi_dist(stream, path_station_file):

    n_traces = len(stream)
    epi_dist = np.zeros(n_traces)
    with open(path_station_file, 'r') as in_id:

        for i in range(n_traces):

            station_info = in_id.readline().split()
            epi_dist_str = station_info[-1].replace("'", "")
            epi_dist[i] = float(epi_dist_str)

            in_id.readline()


    r_planet = 6371.0
    circumference = 2.0*np.pi*r_planet
    half_circ = circumference/2.0
    i_negative = np.where(epi_dist > half_circ)[0]
    epi_dist[i_negative] = -1.0*(circumference - epi_dist[i_negative])
    
    # Convert from km to m.
    epi_dist = epi_dist*1.0E3

    for i in range(n_traces):

        station_rel_coords = {'epi_dist_m' : epi_dist[i]}

        stream.traces[i].stats['rel_coords'] = station_rel_coords

    return stream

def add_geometric_info(stream, x_profile, z_profile):
    
    n_x = len(x_profile)
    n_z = len(z_profile)

    p = 0

    for j in range(n_x):

        for k in range(n_z):

            station_id = '{:>05d}'.format(p)

            trace = stream.select(station = station_id)[0]
            
            coords = {  'epi_dist_m' : x_profile[j]*1.0E3,
                        'depth_m' : z_profile[k]*1.0E3,
                        'x_index' : j,
                        'z_index' : k}

            trace.stats['rel_coords'] = coords

            p = p + 1

    return stream

def merge_numpy_arrays(dir_output, path_channels, output_type, i_t_min, i_t_max):

    # Get name of variable for output file.
    var_name_dict = {'displacement' : 's', 'velocity' : 'v', 'acceleration' : 'a'}
    var_name = var_name_dict[output_type]

    # Get name of output directory.
    dir_np_arrays = os.path.join(dir_output, 'np_arrays')

    #
    path_merged_array = os.path.join(dir_np_arrays, '{:}_from_{:>06d}_to_{:>06d}.npy'.format(var_name, i_t_min, i_t_max))

    if os.path.exists(path_merged_array):

        print('{:} already exists, loading'.format(path_merged_array))
        s = np.load(path_merged_array)

        return s

    # Get station information.
    name_stations_data_frame = 'stations.pkl'
    path_stations_data_frame = os.path.join(dir_output, name_stations_data_frame)
    stations = pandas.read_pickle(path_stations_data_frame)

    # Load channel information.
    channel_dict = read_channel_file(path_channels)

    # Get station list.
    station_list = list(stations.index)
    num_stations = len(station_list)
    
    n_t = i_t_max - i_t_min
    n_channels = np.max([len(channel_dict[st]['channels']) for st in station_list])
    s = np.zeros((num_stations, n_channels, n_t))
    for i in range(num_stations):

        station = station_list[i]

        channels = channel_dict[station]['channels']
        for j, channel in enumerate(channels):

            name_trace = '{:}_{:}_{:}.npy'.format(var_name, station, channel)
            path_trace = os.path.join(dir_np_arrays, name_trace)

            s[i, j, :] = np.load(path_trace)[i_t_min : i_t_max]

    print('Writing {:}'.format(path_merged_array))
    np.save(path_merged_array, s)

    return s

def circle(radius, n_pts = 100, repeat_first = True):

    if repeat_first:

        theta_span = np.linspace(0.0, 2.0*np.pi, num = n_pts)

    else:

        theta_span = np.linspace(0.0, 2.0*np.pi, num = n_pts + 1)[:-1]

    x = radius*np.cos(theta_span)
    y = radius*np.sin(theta_span)

    p = np.array([x, y])

    return p

def strfdelta(tdelta, fmt='{D:02}d {H:02}h {M:02}m {S:02}s', inputtype='timedelta'):
    """
    https://stackoverflow.com/questions/538666
    Convert a datetime.timedelta object or a regular number to a custom-
    formatted string, just like the stftime() method does for datetime.datetime
    objects.

    The fmt argument allows custom formatting to be specified.  Fields can
    include seconds, minutes, hours, days, and weeks.  Each field is optional.

    Some examples:
        '{D:02}d {H:02}h {M:02}m {S:02}s' --> '05d 08h 04m 02s' (default)
        '{W}w {D}d {H}:{M:02}:{S:02}'     --> '4w 5d 8:04:02'
        '{D:2}d {H:2}:{M:02}:{S:02}'      --> ' 5d  8:04:02'
        '{H}h {S}s'                       --> '72h 800s'

    The inputtype argument allows tdelta to be a regular number instead of the
    default, which is a datetime.timedelta object.  Valid inputtype strings:
        's', 'seconds',
        'm', 'minutes',
        'h', 'hours',
        'd', 'days',
        'w', 'weeks'
    """

    # Convert tdelta to integer seconds.
    if inputtype == 'timedelta':
        remainder = int(tdelta.total_seconds())
    elif inputtype in ['s', 'seconds']:
        remainder = int(tdelta)
    elif inputtype in ['m', 'minutes']:
        remainder = int(tdelta)*60
    elif inputtype in ['h', 'hours']:
        remainder = int(tdelta)*3600
    elif inputtype in ['d', 'days']:
        remainder = int(tdelta)*86400
    elif inputtype in ['w', 'weeks']:
        remainder = int(tdelta)*604800

    f = Formatter()
    desired_fields = [field_tuple[1] for field_tuple in f.parse(fmt)]
    possible_fields = ('W', 'D', 'H', 'M', 'S')
    constants = {'W': 604800, 'D': 86400, 'H': 3600, 'M': 60, 'S': 1}
    values = {}
    for field in possible_fields:
        if field in desired_fields and field in constants:
            values[field], remainder = divmod(remainder, constants[field])
    return f.format(fmt, **values)

def get_delta_section(depth_of_section, delta_lims, n_pts_section, dir_output,  delta_in, r_in, s_in):

    n_t = s_in.shape[1]
    r_in_unique = np.unique(r_in)

    # Get coordinates of output profile.
    r_of_section = (6371.0 - depth_of_section)
    delta_section = np.linspace(*delta_lims, num = n_pts_section)
    delta_section = np.deg2rad(delta_section)
    r_section = np.zeros(n_pts_section) + r_of_section

    path_section = os.path.join(dir_output, 'delta_section_{:>05.1f}.npy'.format(depth_of_section))
    if os.path.exists(path_section):

        s_section = np.load(path_section)

    else:

        if r_of_section in r_in_unique:

            j_interp = np.where(r_in == r_of_section)[0]

            # Interpolate.
            s_section = np.zeros((n_pts_section, n_t))
            for i in range(n_t):
                
                s_section[:, i] = np.interp(delta_section, delta_in[j_interp], s_in[j_interp, i])
                
        else:

            x_in = r_in*np.sin(delta_in)
            y_in = r_in*np.cos(delta_in)

            # Get indices of different regions of input data.
            j_mantle = np.where((r_in <= r_srf) & (r_in > r_cmb))[0]
            j_outer_core = np.where((r_in <= r_cmb) & (r_in > r_icb))[0]
            j_inner_core = np.where((r_in <= r_icb))[0]
            input_index_list = [j_mantle, j_outer_core, j_inner_core]

            x_section = r_section*np.sin(delta_section)
            y_section = r_section*np.cos(delta_section)

            # Choose the correct region.
            if r_of_section > r_cmb:

                j_interp = j_mantle

            elif (r_of_section <= r_cmb) & (r_of_section > r_icb):

                j_interp = j_outer_core

            elif (r_of_section <= r_icb):

                j_interp = j_inner_core

            # Interpolate.
            s_section = np.zeros((n_pts_section, n_t))
            for i in range(n_t):
                
                s_section[:, i] = griddata( (x_in[j_interp], y_in[j_interp]), s[j_interp, i],
                                            (x_section, y_section),
                                            method = 'linear')

        np.save(path_section, s_section)

    return r_section, delta_section, s_section

def get_depth_section(delta_of_section, depth_lims, r_spacing_appx, dir_output, delta_in, r_in, s_in):
    
    delta_of_section_deg = delta_of_section
    delta_of_section_rad = np.deg2rad(delta_of_section)

    n_t = s_in.shape[1]
    r_in_unique = np.unique(r_in)

    # Get coordinates of output profile.
    dr_mantle = r_srf - r_cmb
    dr_outer_core = r_cmb - r_icb
    dr_inner_core = r_icb
    #
    #
    n_r_mantle = int(np.round(dr_mantle/r_spacing_appx)) + 1
    n_r_outer_core = int(np.round(dr_outer_core/r_spacing_appx)) + 1
    n_r_inner_core = int(np.round(dr_inner_core/r_spacing_appx)) + 1
    #
    r_mantle        = np.linspace(r_srf, r_cmb, num = n_r_mantle)
    r_outer_core    = np.linspace(r_cmb, r_icb, num = n_r_outer_core)
    r_inner_core    = np.linspace(r_icb, 0.0, num = n_r_inner_core)
    #
    # Apply small shift at boundaries to avoid interpolation ambiguities and
    # and avoid singularity at centre of Earth.
    buff = 1.0
    r_mantle[-1]     = r_mantle[-1] + 1.0
    r_outer_core[0]  = r_outer_core[0] - 1.0
    r_outer_core[-1] = r_outer_core[-1] + 1.0
    r_inner_core[0]  = r_inner_core[0] - 1.0
    #
    r_section = np.concatenate([r_mantle, r_outer_core, r_inner_core])
    n_pts_section = len(r_section)
    #
    delta_section = np.zeros(n_pts_section) + delta_of_section_rad

    path_section = os.path.join(dir_output, 'depth_section_{:>05.1f}.npy'.format(delta_of_section_deg))
    if os.path.exists(path_section):

        s_section = np.load(path_section)

    else:
        
        # --
        x_in = r_in*np.sin(delta_in)
        y_in = r_in*np.cos(delta_in)

        # Get indices of different regions of input data.
        j_mantle = np.where((r_in <= r_srf) & (r_in > r_cmb))[0]
        j_outer_core = np.where((r_in <= r_cmb) & (r_in > r_icb))[0]
        j_inner_core = np.where((r_in <= r_icb))[0]
        input_index_list = [j_mantle, j_outer_core, j_inner_core]

        x_section = r_section*np.sin((delta_section))
        y_section = r_section*np.cos((delta_section))
        
        # --
        # Get indices of different regions of input data.
        i_mantle = np.where((r_section <= r_srf) & (r_section > r_cmb))[0]
        i_outer_core = np.where((r_section <= r_cmb) & (r_section > r_icb))[0]
        i_inner_core = np.where((r_section <= r_icb))[0]
        output_index_list = [i_mantle, i_outer_core, i_inner_core]

        # Interpolate.
        s_section = np.zeros((n_pts_section, n_t))
        for i in range(n_t):

            for k in range(3):

                i_input = input_index_list[k]
                i_output = output_index_list[k]
            
                s_section[i_output, i] = griddata( (x_in[i_input], y_in[i_input]), s_in[i_input, i],
                                            (x_section[i_output], y_section[i_output]),
                                        method = 'linear')

        for i in range(n_t):

            for j in range(n_pts_section):
                
                if r_section[j] in r_in_unique:

                    j_interp = np.where(r_in == r_section[j])[0]

                    s_section[j, i] = np.interp(delta_of_section_rad, delta_in[j_interp], s_in[j_interp, i])

        np.save(path_section, s_section)

    return r_section, delta_section, s_section

def plot_section(section_type, grid_info, t, s, depth_or_delta_of_section, t_ref, dir_output, name_fig, depth_lims = [0.0, 6371.0], delta_lims = [0.0, 180.0], n_pts_section = 'default', r_spacing_apx = 200.0, mode_info = None, path_out = None, show_phase_vel = False, show_group_vel = False):

    # Currently only use one channel.
    channel = 0
    s = s[:, channel, :]

    # Check input arguments.
    assert section_type in ['delta', 'depth'] 
    
    # Get default number of points, if requested.
    if n_pts_section == 'default':

        if section_type == 'delta':

            n_pts_section = 19

        else:

            pass

    # Get coordinates of input data.
    r_in = grid_info['r']
    delta_in = grid_info['delta']
    n_t = len(t)

    if section_type == 'delta':
        
        x_lims = [0.0, 180.0]
        depth_of_section = depth_or_delta_of_section
        r_section, delta_section, s_section = get_delta_section(depth_of_section, delta_lims, n_pts_section, dir_output,  delta_in, r_in, s)

    else:

        x_lims = [0.0, 6371.0]
        delta_of_section = depth_or_delta_of_section
        r_section, delta_section, s_section = get_depth_section(delta_of_section, depth_lims, r_spacing_apx, dir_output, delta_in, r_in, s) 
        n_pts_section = len(r_section)

    traces = []
    for i in range(n_pts_section):
        
        rel_coords =  {'epi_dist_m' : delta_section[i]*r_srf*1.0E3,
                        'depth_m'   : (r_srf - r_section[i])*1.0E3 }
        header = {'rel_coords' : rel_coords, 'station' : '{:>05d}'.format(i),
                    'sampling_rate' : 1.0/(t[1] - t[0]),
                    'starttime' : UTCDateTime(t_ref)}
        trace = Trace(s_section[i, :], header = header)

        traces.append(trace)

    stream = Stream(traces)
    
    f_high_pass_Hz = 4.90*1.0E-3
    f_low_pass_Hz = 5.90*1.0E-3
    print('Filtering from {:.1f} to {:.1f} mHz'.format(f_high_pass_Hz*1.0E3, f_low_pass_Hz*1.0E3))
    #stream.filter(type = 'lowpass', freq = 0.005, corners = 4, zerophase = True)
    stream.filter(type = 'bandpass',
            freqmin = f_high_pass_Hz,
            freqmax = f_low_pass_Hz,
           corners = 2, zerophase = True)

    # Trim last  30 minutes.
    stream.trim(stream[0].stats.starttime, stream[0].stats.endtime - 45.0*60.0)
    t = stream[0].times()
    
    #i = 4
    #print(360.0*stream[i].stats['rel_coords']['epi_dist_m']/(2.0*np.pi*6371.0E3))
    #x = stream[i].data
    #X = np.fft.rfft(x)
    #f = np.fft.rfftfreq(len(x), d = 10.0)

    #fig = plt.figure()
    #ax = plt.gca()

    #ax.plot(1000.0*f, np.abs(X))

    #plt.show()

    #import sys
    #sys.exit()
    
    select_modes = True
    if select_modes:

        if mode_info is not None:

            #n_choose = 0
            #l_choose = 48
            #n_choose = 1
            #l_choose = 31
            
            #n_choose = [ 2,  2,  2,  2,  3,  3,  3]
            #l_choose = [22, 23, 24, 25, 26, 27, 28]

            #n_choose = [ 3,  3,  3,  3,  3,  2,  2,  2,  2]
            #l_choose = [21, 22, 23, 24, 25, 26, 27, 28, 29]

            #n_choose = [ 2,  3]
            #l_choose = [25, 25]

            n_choose = [ 2,  3,  2,  3,  2,  3]
            l_choose = [22, 22, 25, 25, 28, 28] 

            n_choose.reverse()
            l_choose.reverse()

            i_choose = []
            for i in range(len(n_choose)):
                
                i_choose.append(np.where((mode_info['n'] == n_choose[i]) & (mode_info['l'] == l_choose[i]))[0][0])

            for key in mode_info.keys():

                mode_info[key] = mode_info[key][i_choose]

    path_fig = os.path.join(dir_output, name_fig)
    plot_gather(stream, UTCDateTime(t_ref),
        normalisation = 'global_max',
        path_out = path_out,
        x_lims = x_lims,
        t_lims = [t[0], t[-1]],
        plot_type = '{:}_section'.format(section_type),
        offset_mode = 'proportional',
        mode_info = mode_info,
        line_style = 'line',
        show_phase_vel = show_phase_vel,
        show_group_vel = show_group_vel)

    return

def plot_snapshot(delta_profile, depth_profile, stream, time_index, path_out = None):

    # Convert to polar coordinates.
    circ_planet = 2.0*np.pi*r_srf
    r_profile = r_srf - depth_profile
    theta_profile = 2.0*np.pi*delta_profile/circ_planet
    
    r_grid, theta_grid = np.meshgrid(r_profile, theta_profile)

    x_grid = r_grid*np.sin(theta_grid)
    z_grid = r_grid*np.cos(theta_grid)

    x_grid_flat = x_grid.flatten()
    z_grid_flat = z_grid.flatten()

    r_grid_flat = r_grid.flatten()
    j_mantle = np.where((r_grid_flat <= r_srf) & (r_grid_flat > r_cmb))[0]
    j_outer_core = np.where((r_grid_flat <= r_cmb) & (r_grid_flat > r_icb))[0]
    j_inner_core = np.where((r_grid_flat <= r_icb))[0]
    input_index_list = [j_mantle, j_outer_core, j_inner_core]

    # Find the overall maximum amplitude.
    max_val = np.max(stream.max())

    # Get the output data grid. 
    n_delta = len(delta_profile)
    n_depth = len(depth_profile)
    a = np.zeros((n_delta, n_depth))

    p = 0
    for j in range(n_delta):

        for k in range(n_depth):

            station_id = '{:>05d}'.format(p)
            trace = stream.select(station = station_id)[0]

            a[j, k] = trace.data[time_index]

            p = p + 1

    a_grid_flat = a.flatten()

    # Re-grid.
    n_grid = 100
    span = np.linspace(-r_srf, r_srf, num = n_grid)
    X, Z = np.meshgrid(span, span)
    R = np.sqrt((X**2.0) + (Z**2.0))
    #
    X_flat = X.flatten()
    Z_flat = Z.flatten()
    R_flat = R.flatten()
    i_mask = np.where((R_flat > r_srf))[0]
    i_mantle = np.where((R_flat <= r_srf) & (R_flat > r_cmb))[0]
    i_outer_core = np.where((R_flat <= r_cmb) & (R_flat > r_icb))[0]
    i_inner_core = np.where((R_flat <= r_icb))[0]
    output_index_list = [i_mantle, i_outer_core, i_inner_core]

    A_flat = np.zeros(R_flat.shape)
    for k in range(3):

        i = output_index_list[k]
        j = input_index_list[k]

        #interp_func = interp2d(x_grid_flat[j], z_grid_flat[j], a_grid_flat[j])
        #A_flat[i] = interp_func(X_flat[i], Z_flat[i]) 

        A_flat[i] = griddata((x_grid_flat[j], z_grid_flat[j]), a_grid_flat[j], (X_flat[i], Z_flat[i]),
                    method = 'linear')
    
    A = A_flat.reshape(X.shape)

    x_corners = np.zeros(n_grid + 1)
    d_x = span[1] - span[0]
    x_corners[:-1] = span - (d_x/2.0)
    x_corners[-1] = span[-1] + d_x/2.0

    # Get time information.
    t = stream[0].times()[time_index]
    t_str = strfdelta(datetime.timedelta(seconds = t))

    # Define a colour scale.
    c_max = 0.1*max_val
    c_norm = colors.Normalize(vmin = -1.0*c_max, vmax = c_max)
    c_map = 'seismic'
    
    fig = plt.figure(figsize = (9.5, 9.0))
    ax = fig.add_axes([0.05, 0.05, 0.8, 0.9])

    h_pc = ax.pcolormesh(x_corners, x_corners, A, norm = c_norm, cmap = c_map)

    #h_sc = ax.scatter(x_grid, z_grid, c = a, norm = c_norm, cmap = c_map)
    h_sc = ax.scatter(x_grid, z_grid, c = 'k', s = 2)

    r_discons = [r_srf, r_cmb, r_icb]
    circle_pt_spacing = 100.0
    c_circ = 'k'
    for r_discon in r_discons:
        
        arc_length = 2.0*np.pi*r_discon
        n_pts = int(np.round(arc_length/circle_pt_spacing))
        p_circ = circle(r_discon, n_pts = n_pts)
        
        ax.plot(p_circ[0], p_circ[1], c = c_circ, zorder = 10)

    ax.set_aspect(1.0)

    #divider = make_axes_locatable(ax)
    #cax = divider.append_axes('right', size='1%', pad = 0.35)
    #cb = fig.colorbar(h_sc, cax=cax, orientation='vertical')
    #cb.ax.set_ylabel('Acceleration (nm s$^{-2}$)', fontsize = 12)

    cax = fig.add_axes([0.87, 0.05, 0.03, 0.5])
    #cax = divider.append_axes('right', size='1%', pad = 0.35)
    cb = fig.colorbar(h_pc, cax=cax, orientation='vertical')
    cb.ax.set_ylabel('Acceleration (nm s$^{-2}$)', fontsize = 12)
    
    buff = 100.0
    r_lim = (r_srf + buff)
    r_lims = [-r_lim, r_lim]
    ax.set_xlim(r_lims)
    ax.set_ylim(r_lims)
    #cax = fig.add_axes([0.78, 0.5, 0.03, 0.38])
    #cb = plt.colorbar(h_sc, cax = cax)

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    for ax_name in ['top', 'bottom', 'left', 'right']:
        ax.spines[ax_name].set_visible(False)

    ax.set_title(t_str, fontsize = 12)

    plt.show()

    return

def animation(grid_info, t, s, path_out = None, variable = 'acceleration'):
    
    channel = 0
    s = s[:, channel, :]
    
    s = s*1.0E9 # Convert from m to nm.

    r_in = grid_info['r']
    delta_in = grid_info['delta']
    n_t = len(t)

    x_in = r_in*np.sin(delta_in)
    y_in = r_in*np.cos(delta_in)

    j_mantle = np.where((r_in <= r_srf) & (r_in > r_cmb))[0]
    j_outer_core = np.where((r_in <= r_cmb) & (r_in > r_icb))[0]
    j_inner_core = np.where((r_in <= r_icb))[0]
    input_index_list = [j_mantle, j_outer_core, j_inner_core]

    # Find the overall maximum amplitude.
    max_val = np.max(s)
    max_val = 1.0E5 
    s = s/max_val
    scale = 10.0
    s = s*scale

    # Re-grid.
    n_x_grid = 100
    x_span = np.linspace(0.0, r_srf, num = n_x_grid)
    d_x = x_span[1] - x_span[0]
    x_corners = np.zeros(n_x_grid + 1)
    x_corners[:-1] = x_span - (d_x/2.0)
    x_corners[-1] = x_span[-1] + d_x/2.0
    #
    n_y_grid = 2*n_x_grid - 1
    y_span = np.linspace(-r_srf, r_srf, num = n_y_grid)
    d_y = y_span[1] - y_span[0]
    y_corners = np.zeros(n_y_grid + 1)
    y_corners[:-1] = y_span - (d_y/2.0)
    y_corners[-1] = y_span[-1] + d_y/2.0
    #
    X, Y = np.meshgrid(x_span, y_span, indexing = 'ij')
    R = np.sqrt((X**2.0) + (Y**2.0))
    #
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    R_flat = R.flatten()
    i_mask = np.where((R_flat > r_srf))[0]
    i_mantle = np.where((R_flat <= r_srf) & (R_flat > r_cmb))[0]
    i_outer_core = np.where((R_flat <= r_cmb) & (R_flat > r_icb))[0]
    i_inner_core = np.where((R_flat <= r_icb))[0]
    output_index_list = [i_mantle, i_outer_core, i_inner_core]
    #
    A = np.zeros((n_t, n_x_grid, n_y_grid))

    print('Interpolating traces onto grid.')
    for p in range(n_t): 
        
        A_flat = np.zeros(R_flat.shape) + np.nan
        for k in range(3):

            i = output_index_list[k]
            j = input_index_list[k]

            A_flat[i] = griddata(   (x_in[j], y_in[j]), s[j, p],
                                    (X_flat[i], Y_flat[i]),
                                    method = 'linear',
                                    fill_value = np.nan)

        A[p, :, :] = A_flat.reshape(X.shape)

    # Filter (if requested).
    #filter_info = {'type' : 'lowpass', 'f_low_pass_Hz' : 0.006}
    filter_info = {'type' : 'bandpass'}
    if filter_info['type'] == 'lowpass':

        d_t = t[1] - t[0]
        d_f = 1.0/d_t
        
        for j in range(n_x_grid):

            for k in range(n_y_grid):

                #print('{:>5d} {:>5d} {:>12.2f} {:>12.2f} {:>8.1e}'.format(j, k, x_span[j], y_span[k], A[0, j, k])) 
                
                if not np.isnan(A[0, j, k]):

                    A[:, j, k] = lowpass(A[:, j, k], f_low_pass_Hz, d_f, corners = 2,
                                    zerophase = True)

    elif filter_info['type'] == 'bandpass':

        d_t = t[1] - t[0]
        d_f = 1.0/d_t

        #f_range = 0.007
        #f_frac = 0.15
        #f_high_pass_Hz = f_frac*f_range
        #f_low_pass_Hz = (1.0 - f_frac)*f_range
    
        f_high_pass_Hz = 4.90*1.0E-3
        f_low_pass_Hz = 5.90*1.0E-3
        
        for j in range(n_x_grid):

            for k in range(n_y_grid):

                if not np.isnan(A[0, j, k]):

                    A[:, j, k] = bandpass(A[:, j, k], f_high_pass_Hz, f_low_pass_Hz,
                                    d_f, corners = 2, zerophase = True)

    # Define a colour scale.
    #c_max = 0.1*max_val
    c_max = 1.0 
    c_norm = colors.Normalize(vmin = -1.0*c_max, vmax = c_max)
    #c_map = 'seismic'
    c_map = copy.copy(mpl_cm.get_cmap('seismic'))
    c_map.set_bad('white')

    # Prepare axes.
    #fig = plt.figure(figsize = (9.5, 9.0))
    fig = plt.figure(figsize = (7.5, 9.0))
    ax = fig.add_axes([0.05, 0.05, 0.7, 0.9])

    h_pc = ax.pcolormesh(x_corners, y_corners, A[0, ...].T, norm = c_norm, cmap = c_map)
    
    show_grids = False
    if show_grids:

        h_sc = ax.scatter(X, Y, c = 'k', s = 2, alpha = 0.5)
        h_sc = ax.scatter(x_in, y_in, c = 'g', s = 4, alpha = 0.5)

    r_discons = [r_srf, r_cmb, r_icb]
    circle_pt_spacing = 100.0
    c_circ = 'k'
    for r_discon in r_discons:
        
        arc_length = 2.0*np.pi*r_discon
        n_pts = int(np.round(arc_length/circle_pt_spacing))
        p_circ = circle(r_discon, n_pts = n_pts)
        
        ax.plot(p_circ[0], p_circ[1], c = c_circ, zorder = 10)

    ax.set_aspect(1.0)

    cax = fig.add_axes([0.77, 0.05, 0.03, 0.5])
    cb = fig.colorbar(h_pc, cax=cax, orientation='vertical')

    variable_cb_dict = {'acceleration'  : 'Acceleration ({:.1e} nm s$^{{-2}}$)'.format(max_val/scale),
            'velocity'      : 'Velocity ({:.1e} nm s$^{{-1}}$)'.format(max_val/scale),
            'displacement'  : 'Displacement ({:.1e} nm)'.format(max_val/scale)}
    cb_label = variable_cb_dict[variable]
    cb.ax.set_ylabel(cb_label, fontsize = 12)
    
    buff = 100.0
    r_lim = (r_srf + buff)
    r_lims = [-r_lim, r_lim]
    #ax.set_xlim(r_lims)
    #ax.set_ylim(r_lims)
    ax.set_xlim([-buff, r_lim])
    ax.set_ylim(r_lims)

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    for ax_name in ['top', 'bottom', 'left', 'right']:
        ax.spines[ax_name].set_visible(False)

    t_str = strfdelta(datetime.timedelta(seconds = 0))
    title = ax.set_title(t_str, fontsize = 12)

    def animate(i):

        h_pc.set_array(A[i, :, :].T.flatten())

        t_str = strfdelta(datetime.timedelta(seconds = t[i]))
        title.set_text(t_str)

    anim = FuncAnimation(
            fig, animate, interval = 100, frames = n_t - 1)


    if path_out is None: 

        plt.draw()
        plt.show()
    
    else:
        
        print("Writing to {:}".format(path_out))
        anim.save(path_out, writer = 'ffmpeg')

    return

def main():
    
    # Grid settings.
    half_or_full_circle = 'half'
    grid_type = 'even'
    assert grid_type in ['spokes', 'even']

    # Parse input arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("path_mode_input", help = "File path (relative or absolute) to Ouroboros mode input file.")
    parser.add_argument("path_summation_input", help = "File path (relative or absolute) to Ouroboros summation input file.")
    parser.add_argument("event", type = int, help = "Event integer ID.")
    parser.add_argument("--delta", metavar = 'depth', nargs = 1, type = float, help = "Do an epicentral distance profile at the specified depth (km).")
    parser.add_argument("--depth", metavar = 'delta', nargs = 1, type = float, help = "Do a depth profile at the specified epicentral distance (degrees).")
    parser.add_argument("--time", nargs = 1, type = int, help = "Do a 2-D snapshot at specified time, or use -1 to make animation.")
    parser.add_argument("--t_lims", nargs = 2, type = float, help = "Specify time range in seconds.")
    parser.add_argument("--show_phase_vel", action = 'store_true', help = "For epicentral distance profiles, add phase velocity lines.")
    parser.add_argument("--show_group_vel", action = 'store_true', help = "For epicentral distance profiles, add group velocity lines.")
    #
    args = parser.parse_args()
    path_mode_input = args.path_mode_input
    path_summation_input = args.path_summation_input
    i = args.event
    depth_profile_arg = args.depth
    delta_profile_arg = args.delta
    time_snapshot_arg = args.time
    t_lims = args.t_lims
    show_phase_vel = args.show_phase_vel
    show_group_vel = args.show_group_vel

    # Read input files.
    run_info = read_input_file(path_mode_input)
    summation_info = read_summation_input_file(path_summation_input, 'ouroboros')
    run_info['dir_model'], run_info['dir_run'] = get_Mineos_out_dirs(run_info) 
    summation_info = get_Ouroboros_summation_out_dirs(run_info, summation_info,
                            name_summation_dir = 'summation_Ouroboros')

    # Load CMT.
    cmt = read_mineos_cmt(summation_info['path_cmt'])

    # Load grid information.
    dir_channels = os.path.dirname(summation_info['path_channels'])
    if grid_type == 'spokes':

        # Load x-profile information.
        path_x_profile = os.path.join(dir_channels, 'great_circle_{:>03d}_x_profile.txt'.format(i))
        x_profile_lon, x_profile_lat, x_profile = np.loadtxt(path_x_profile, usecols = (1, 2, 3)).T

        # Load z-profile information.
        path_z_profile = os.path.join(dir_channels, 'great_circle_{:>03d}_z_profile.txt'.format(i))
        z_profile = np.loadtxt(path_z_profile, usecols = (1))

        grid_info = {'type' : 'spokes', 'x_profile' : x_profile,
                        'z_profile' : z_profile}

    elif grid_type == 'even':
        
        path_grid = os.path.join(dir_channels, 'great_circle_{:>03d}_point_list.txt'.format(i))
        lon, lat, delta, r = read_even_grid(path_grid)
        grid_info = {'type' : grid_type, 'lon' : lon, 'lat' : lat,
                     'delta' : delta, 'r' : r}

    # Get time information.
    t = np.array(range(0, summation_info['n_samples']))*summation_info['d_t']
    i_t_min = np.argmax(t >= t_lims[0])
    i_t_max = np.argmax(t > t_lims[1])
    n_t = i_t_max - i_t_min
    t = t[i_t_min : i_t_max]

    # Make array of displacement.
    s = merge_numpy_arrays(summation_info['dir_output'],
            summation_info['path_channels'], summation_info['output_type'],
            i_t_min, i_t_max)

    # Load mode information.
    if show_phase_vel or show_group_vel:

        mode_info = load_eigenfreq(run_info, 'S')

        if summation_info['path_mode_list'] is not None:

            mode_info = filter_mode_list({'S' : mode_info}, summation_info['path_mode_list'])['S']

    else:

        mode_info = None

    assert (depth_profile_arg is not None) or (delta_profile_arg is not None) or (time_snapshot_arg is not None), 'Must specify one of --depth, --delta, or --time.'
    if depth_profile_arg is not None:

        assert delta_profile_arg is None, 'Can only specify one of --depth, --delta and --time.'
        assert time_snapshot_arg is None, 'Can only specify one of --depth, --delta and --time.'

        delta_of_profile = depth_profile_arg[0]
        plot_type = 'depth_section'

    elif delta_profile_arg is not None:
        
        assert depth_profile_arg is None, 'Can only specify one of --depth, --delta and --time.'
        assert time_snapshot_arg is None, 'Can only specify one of --depth, --delta and --time.'

        depth_of_profile = delta_profile_arg[0]
        plot_type = 'delta_section'

    elif time_snapshot_arg is not None:

        assert depth_profile_arg is None, 'Can only specify one of --depth, --delta and --time.'
        assert delta_profile_arg is None, 'Can only specify one of --depth, --delta and --time.'

        p = time_snapshot_arg[0]
        if p == -1:

            plot_type = 'animation'

        else:
            
            raise NotImplementedError
            assert t_lims is None, "Cannot specify --t_lims when plotting a time snapshot. Use --time."
            plot_type = 'time_snapshot'

    # Get name of variable for output file.
    var_name_dict = {'displacement' : 's', 'velocity' : 'v', 'acceleration' : 'a'}
    var_name = var_name_dict[summation_info['output_type']]

    # Get output file names and axes limits.
    if plot_type == 'delta_section':

        name_fig = 'delta_section_{:}_{:>04d}_{:>05.1f}.png'.format(var_name, i, depth_of_profile)
        path_fig = os.path.join(summation_info['dir_output'], name_fig)

    elif plot_type == 'depth_section':

        name_fig = 'depth_section_{:}_{:>5.1f}.png'.format(var_name, i, delta_of_profile)
        path_fig = os.path.join(summation_info['dir_output'], name_fig)

    elif plot_type == 'time_snapshot':

        name_fig = 'time_snapshot_{:}_{:>04d}_{:>04d}.png'.format(var_name, i, p)
        path_fig = os.path.join(summation_info['dir_output'], name_fig)

    elif plot_type == 'animation':

        name_fig = 'animation_{:}_{:>04d}.mp4'.format(var_name, i)
        path_fig = os.path.join(summation_info['dir_output'], name_fig)

    if plot_type == 'delta_section':
        
        plot_section('delta', grid_info, t, s, depth_of_profile,
                    cmt['datetime_ref'],
                    summation_info['dir_output'],
                    name_fig,
                    mode_info = mode_info,
                    show_group_vel = show_group_vel,  
                    show_phase_vel = show_phase_vel,
                    path_out = path_fig)

    elif plot_type == 'depth_section':

        plot_section('depth', grid_info, t, s, delta_of_profile,
                    cmt['datetime_ref'],
                    summation_info['dir_output'],
                    name_fig)

    elif plot_type == 'animation':

        animation(grid_info, t, s, path_out = path_fig,
                    variable = summation_info['output_type'])

    return

if __name__ == '__main__':

    main()
