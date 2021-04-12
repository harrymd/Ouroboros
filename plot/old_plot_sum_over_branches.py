import argparse
import datetime
import os
from string import Formatter

from matplotlib.animation import FuncAnimation
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from obspy.core import read as obspy_read
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.stream import Stream
#from scipy.interpolate import interp2d
from scipy.interpolate import griddata

from Ouroboros.common import (filter_mode_list, load_eigenfreq,
                            get_Mineos_out_dirs, get_Ouroboros_summation_out_dirs,
                            read_input_file, read_summation_input_file)
from Ouroboros.misc.cmt_io import read_mineos_cmt
from Ouroboros.plot.plot_gather import plot_gather

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

def plot_snapshot(delta_profile, depth_profile, stream, time_index, path_out = None):

    # Define radii.
    r_srf = 6371.0
    r_cmb = 3480.0
    r_icb = 1221.5

    # Convert to polar coordinates.
    r_srf = 6371.0
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

def animation(delta_profile, depth_profile, stream, path_out = None, variable = 'acceleration', t_lims = None):

    # Define radii.
    r_srf = 6371.0
    r_cmb = 3480.0
    r_icb = 1221.5

    # Convert to polar coordinates.
    r_srf = 6371.0
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
    # Prepare output array.
    if t_lims is None:
        
        n_t_max = len(stream[0].times())

    else:

        t = stream[0].times()
        i_t_min = np.argmax(t >= t_lims[0])
        i_t_max = np.argmax(t > t_lims[1])
        n_t = i_t_max - i_t_min + 1

    n_delta = len(delta_profile)
    n_depth = len(depth_profile)
    a = np.zeros((n_t, n_delta, n_depth)) 

    print('Reading traces.')
    p = 0
    for j in range(n_delta):

        for k in range(n_depth):

            station_id = '{:>05d}'.format(p)
            trace = stream.select(station = station_id)[0]
                
            a[:, j, k] = trace.data[i_t_min : i_t_max + 1]

            p = p + 1

    # Re-grid.
    n_grid = 200
    span = np.linspace(-r_srf, r_srf, num = n_grid)
    x_corners = np.zeros(n_grid + 1)
    d_x = span[1] - span[0]
    x_corners[:-1] = span - (d_x/2.0)
    x_corners[-1] = span[-1] + d_x/2.0
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
    A = np.zeros((n_t, n_grid, n_grid))

    print('Interpolating traces onto grid.')
    for p in range(n_t): 
        
        a_grid_flat = a[p, :, :].flatten()
        A_flat = np.zeros(R_flat.shape)
        for k in range(3):

            i = output_index_list[k]
            j = input_index_list[k]

            A_flat[i] = griddata(   (x_grid_flat[j], z_grid_flat[j]), a_grid_flat[j],
                                    (X_flat[i], Z_flat[i]),
                                    method = 'linear')
    
        A[p, :, :] = A_flat.reshape(X.shape)

    # Define a colour scale.
    c_max = 0.1*max_val
    c_norm = colors.Normalize(vmin = -1.0*c_max, vmax = c_max)
    c_map = 'seismic'

    # Prepare axes.
    fig = plt.figure(figsize = (9.5, 9.0))
    ax = fig.add_axes([0.05, 0.05, 0.8, 0.9])

    h_pc = ax.pcolormesh(x_corners, x_corners, A[0, ...], norm = c_norm, cmap = c_map)
    #h_sc = ax.scatter(x_grid, z_grid, c = 'k', s = 2)

    r_discons = [r_srf, r_cmb, r_icb]
    circle_pt_spacing = 100.0
    c_circ = 'k'
    for r_discon in r_discons:
        
        arc_length = 2.0*np.pi*r_discon
        n_pts = int(np.round(arc_length/circle_pt_spacing))
        p_circ = circle(r_discon, n_pts = n_pts)
        
        ax.plot(p_circ[0], p_circ[1], c = c_circ, zorder = 10)

    ax.set_aspect(1.0)

    cax = fig.add_axes([0.87, 0.05, 0.03, 0.5])
    cb = fig.colorbar(h_pc, cax=cax, orientation='vertical')

    variable_cb_dict = {'acceleration'  : 'Acceleration (nm s$^{-2}$)',
                        'velocity'      : 'Velocity (nm s$^{-1}$)',
                        'displacement'  : 'Displacement (nm)'}
    cb_label = variable_cb_dict[variable]
    cb.ax.set_ylabel(cb_label, fontsize = 12)
    
    buff = 100.0
    r_lim = (r_srf + buff)
    r_lims = [-r_lim, r_lim]
    ax.set_xlim(r_lims)
    ax.set_ylim(r_lims)

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    for ax_name in ['top', 'bottom', 'left', 'right']:
        ax.spines[ax_name].set_visible(False)

    t_str = strfdelta(datetime.timedelta(seconds = 0))
    title = ax.set_title(t_str, fontsize = 12)

    times = stream[0].times()

    def animate(i):

        h_pc.set_array(A[i, :, :].flatten())

        t_str = strfdelta(datetime.timedelta(seconds = times[i]))
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

    # Parse input arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("path_mode_input", help = "File path (relative or absolute) to Ouroboros mode input file.")
    parser.add_argument("path_summation_input", help = "File path (relative or absolute) to Ouroboros summation input file.")
    parser.add_argument("event", type = int, help = "Event integer ID.")
    parser.add_argument("--delta", metavar = ('idx_or_val', 'depth'), nargs = 2, type = str, help = "Do an epicentral distance profile at the specified depth. The depth can be specified by an index (e.g. --delta idx 5) or by a value in km (e.g. --delta val 3000.0). In the latter case the nearest index is found.")
    parser.add_argument("--depth", metavar = ('idx_or_val', 'delta'), nargs = 2, type = str, help = "Do a depth profile at the specified epicentral distance. The distance can be specified by an index (e.g. --depth idx 5) or by a value in degrees (e.g. --depth val 70.0). In the latter case the nearest index is found.")
    parser.add_argument("--time", nargs = 1, type = int, help = "Do a 2-D snapshot at specified time, or use -1 to make animation.")
    parser.add_argument("--t_lims", nargs = 2, type = float, help = "Specify time range in seconds.")
    #
    args = parser.parse_args()
    path_mode_input = args.path_mode_input
    path_summation_input = args.path_summation_input
    i = args.event
    depth_profile_arg = args.depth
    delta_profile_arg = args.delta
    time_snapshot_arg = args.time
    t_lims = args.t_lims

    # Read input files.
    run_info = read_input_file(path_mode_input)
    summation_info = read_summation_input_file(path_summation_input, 'ouroboros')
    run_info['dir_model'], run_info['dir_run'] = get_Mineos_out_dirs(run_info) 
    summation_info = get_Ouroboros_summation_out_dirs(run_info, summation_info,
                            name_summation_dir = 'summation_Ouroboros')

    # Load CMT.
    cmt = read_mineos_cmt(summation_info['path_cmt'])

    # Load profile information.
    dir_channels = os.path.dirname(summation_info['path_channels'])
    path_x_profile = os.path.join(dir_channels, 'great_circle_{:>03d}_x_profile.txt'.format(i))
    x_profile_lon, x_profile_lat, x_profile = np.loadtxt(path_x_profile, usecols = (1, 2, 3)).T

    # Load profile information.
    dir_channels = os.path.dirname(summation_info['path_channels'])
    path_z_profile = os.path.join(dir_channels, 'great_circle_{:>03d}_z_profile.txt'.format(i))
    z_profile = np.loadtxt(path_z_profile, usecols = (1))

    # Load mode information.
    mode_info = load_eigenfreq(run_info, 'S')
    mode_info = filter_mode_list({'S' : mode_info}, summation_info['path_mode_list'])['S']

    assert (depth_profile_arg is not None) or (delta_profile_arg is not None) or (time_snapshot_arg is not None), 'Must specify one of --depth, --delta, or --time.'
    if depth_profile_arg is not None:

        assert delta_profile_arg is None, 'Can only specify one of --depth, --delta and --time.'
        assert time_snapshot_arg is None, 'Can only specify one of --depth, --delta and --time.'
        assert depth_profile_arg[0] in ['idx', 'val'], 'First argument of --depth should be \'idx\' or \'val\'.'
        if depth_profile_arg[0] == 'idx':

            j = int(depth_profile_arg[1])

        else:
            
            x_profile_deg = 360.0*x_profile/(2.0*np.pi*6371.0)
            j = np.argmin(np.abs(float(depth_profile_arg[1]) - x_profile_deg)) 

        plot_type = 'depth_section'

    elif delta_profile_arg is not None:
        
        assert depth_profile_arg is None, 'Can only specify one of --depth, --delta and --time.'
        assert time_snapshot_arg is None, 'Can only specify one of --depth, --delta and --time.'
        assert delta_profile_arg[0] in ['idx', 'val'], 'First argument of --delta should be \'idx\' or \'val\'.'
        if delta_profile_arg[0] == 'idx':

            k = int(delta_profile_arg[1])

        else:

            k = np.argmin(np.abs(float(delta_profile_arg[1]) - z_profile)) 

        plot_type = 'delta_section'

    elif time_snapshot_arg is not None:

        assert depth_profile_arg is None, 'Can only specify one of --depth, --delta and --time.'
        assert delta_profile_arg is None, 'Can only specify one of --depth, --delta and --time.'

        p = time_snapshot_arg[0]
        if p == -1:

            plot_type = 'animation'

        else:
            
            assert t_lims is None, "Cannot specify --t_lims when plotting a time snapshot. Use --time."
            plot_type = 'time_snapshot'

    # Get name of variable for output file.
    var_name_dict = {'displacement' : 's', 'velocity' : 'v', 'acceleration' : 'a'}
    var_name = var_name_dict[summation_info['output_type']]

    if plot_type == 'delta_section':

        ivar = x_profile
        name_fig = 'delta_section_{:}_{:>04d}_{:>04d}.png'.format(var_name, i, k)
        path_fig = os.path.join(summation_info['dir_output'], name_fig)
        x_lims = [-180.0, 180.0]

    elif plot_type == 'depth_section':

        ivar = x_profile
        name_fig = 'depth_section_{:}_{:>04d}_{:>04d}.png'.format(var_name, i, j)
        path_fig = os.path.join(summation_info['dir_output'], name_fig)
        x_lims = [0.0, 6371.0]

    elif plot_type == 'time_snapshot':

        name_fig = 'time_snapshot_{:}_{:>04d}_{:>04d}.png'.format(var_name, i, p)
        path_fig = os.path.join(summation_info['dir_output'], name_fig)

    elif plot_type == 'animation':

        name_fig = 'animation_{:}_{:>04d}.mp4'.format(var_name, i)
        path_fig = os.path.join(summation_info['dir_output'], name_fig)

    # Load stream and apply metadata.
    path_stream = os.path.join(summation_info['dir_output'], 'stream_{:}.mseed'.format(var_name))
    print('Loading {:}'.format(path_stream))
    stream = obspy_read(path_stream)
    #stream = add_epi_dist(stream, summation_info['path_channels'])
    print('Calculating geometric information.')
    stream = add_geometric_info(stream, x_profile, z_profile)
    print('Done with geometric information.')

    if plot_type in ['delta_section', 'depth_section']:
    
        if plot_type == 'delta_section':

            traces = []
            for trace in stream:

                if trace.stats.rel_coords.z_index == k:

                    traces.append(trace)

        elif plot_type == 'depth_section':

            traces = []
            for trace in stream:

                if trace.stats.rel_coords.x_index == j:

                    traces.append(trace)

        stream = Stream(traces)
        
        plot_gather(stream, UTCDateTime(cmt['datetime_ref']),
                    #normalisation = 'individual_max',
                    #normalisation = 'median_rms',
                    normalisation = 'global_max',
                    offset_mode = 'proportional',
                    x_lims = x_lims,
                    plot_type = plot_type,
                    path_out = path_fig,
                    t_lims = t_lims,
                    mode_info = mode_info)

    elif plot_type == 'time_snapshot':

        plot_snapshot(x_profile, z_profile, stream, p, path_out = path_fig)

    elif plot_type == 'animation':

        animation(x_profile, z_profile, stream, path_out = path_fig,
                    variable = summation_info['output_type'],
                    t_lims = t_lims)

    return

if __name__ == '__main__':

    main()
