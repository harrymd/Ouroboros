import argparse
import os

import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib import colors as mpl_colors
import numpy as np
from obspy import read
from obspy import UTCDateTime

from Ouroboros.common import add_epi_dist_and_azim, get_Mineos_out_dirs, get_Mineos_summation_out_dirs, mkdir_if_not_exist, read_channel_file, read_Mineos_input_file, read_Mineos_summation_input_file
from Ouroboros.misc.cmt_io import read_mineos_cmt

def plot_gather(stream, t_ref, normalisation = 'individual_max', path_out = None, offset_mode = 'uniform', x_lims = [0.0, 180.0], t_lims = None, plot_type = 'delta_section', mode_info = None, line_style = 'line', show_phase_vel = False, show_group_vel = False):

    line_style = 'varwiggle'

    assert plot_type in ['delta_section', 'depth_section']

    # Count traces.
    n_traces = len(stream)

    # Sort by epicentral distance.
    if plot_type == 'delta_section':

        stream.traces.sort(key = lambda x: x.stats.rel_coords.epi_dist_m)

    elif plot_type == 'depth_section':

        stream.traces.sort(key = lambda x: -x.stats.rel_coords.depth_m)

    # De-trend.
    stream.detrend('demean')

    # Trim.
    if t_lims is not None:

        stream.trim(starttime   = t_ref + t_lims[0],
                    endtime     = t_ref + t_lims[1])

    # Apply normalisation.
    if normalisation == 'individual_max':

        stream.normalize(global_max = False)

    elif normalisation == 'median_rms':

        trace_rms = np.zeros(n_traces) 
        for i in range(n_traces):

            trace_rms[i] = np.sqrt(np.mean(stream[i].data**2.0))

        median_rms = np.median(trace_rms)
        for trace in stream:

            trace.normalize(norm = median_rms)

    elif normalisation == 'global_max':

        stream.normalize(global_max = True)

    else:

        raise ValueError
    
    # Get the overall start and end time, and duration.
    stream_start_times = [trace.stats.starttime for trace in stream]
    stream_end_times = [trace.stats.endtime for trace in stream]
    stream_start_time = min(stream_start_times)
    stream_end_time = max(stream_end_times)
    stream_t_length = stream_end_time - stream_start_time

    # Scaling parameters. 
    t_scale = 1.0/3600.0
    m_to_deg = (2.0*np.pi*6371.0*1.0E3/360.0)
    m_to_km = 1.0E3
    x_range = x_lims[1] - x_lims[0]
    x_per_trace = x_range/n_traces
    if normalisation == 'individual_max':

        x_scale = 4.0*x_per_trace

    elif normalisation == 'median_rms':

        x_scale = 1.0E-1*x_per_trace

    elif normalisation == 'global_max':

        x_scale = 8.0*x_per_trace

    # Plot settings.
    font_size_label = 12
    #label_x_coords = np.linspace(0.0, 180.0, num = (2*n_traces) + 1)[1::2]
    #label_x_coords = np.linspace(x_lims[0], x_lims[1], num = n_traces)

    if plot_type == 'delta_section':

        min_offset = np.min([tr.stats.rel_coords.epi_dist_m/m_to_deg for tr in stream])
        max_offset = np.max([tr.stats.rel_coords.epi_dist_m/m_to_deg for tr in stream])

    elif plot_type == 'depth_section':

        min_offset = np.min([6371.0 - tr.stats.rel_coords.depth_m/m_to_km for tr in stream])
        max_offset = np.max([6371.0 - tr.stats.rel_coords.depth_m/m_to_km for tr in stream])

    label_x_coords = np.linspace(min_offset, max_offset, num = n_traces)

    label_t_coords = np.zeros(n_traces) + stream_t_length*t_scale*1.1
    color_mode = 'constant'

    # Create axes.
    fig = plt.figure(figsize = (11.0, 8.5), constrained_layout = True)
    ax = plt.gca()

    # Set line kwargs.
    line_kwargs = {'lw' : 1, 'alpha' : 0.5}

    #offset_mode = 'proportional'
    #offset_mode = 'uniform'

    if offset_mode == 'uniform':
        
        t_unif = (stream_start_time - t_ref - 3600.0)*t_scale
    
    # Plot each trace.
    for i, trace in enumerate(stream):
        
        if offset_mode == 'proportional':

            if plot_type == 'delta_section':

                # Determine the offset (epicentral distance in degrees).
                offset = trace.stats.rel_coords.epi_dist_m/m_to_deg
                #print(offset)
                #offset = label_x_coords[i]

            elif plot_type == 'depth_section':

                offset = 6371.0 - trace.stats.rel_coords.depth_m/m_to_km

        elif offset_mode == 'uniform':
            
            if plot_type == 'delta_section':

                offset_true = trace.stats.rel_coords.epi_dist_m/m_to_deg
                offset = label_x_coords[i]

        # Get the start time of the trace relative to reference time.
        t_start_relative_to_ref = trace.stats.starttime - t_ref

        # Get the time and displacent arrays.
        t = (trace.times() + t_start_relative_to_ref)*t_scale
        x = trace.data*x_scale

        ## Scale displacement to counteract time decay.
        #x = x*np.exp((t/t_scale)/16000.0)
        #x[np.abs(x) > x_scale] = x_scale*np.sign(x[np.abs(x) > x_scale])

        # Set the line color (cyclic).
        if color_mode == 'cyclic':

            color = 'C{:d}'.format(i % 5)

        else:

            color = 'k'
           
        if line_style == 'line':

            # Plot the line with appropriate offset.
            ax.plot(x + offset, t, color = color, **line_kwargs) 

        elif line_style == 'varwiggle':

            #color_pos = 'r'
            #color_neg = 'b'
            color_pos = 'k'
            color_neg = 'k'
            alpha = 0.6

            ax.fill_betweenx(t, x + offset, x2 = offset, where = x >= 0.0, color = color_pos, alpha = alpha)
            ax.fill_betweenx(t, x + offset, x2 = offset, where = x <= 0.0, color = color_neg, alpha = alpha)
        
        #print(x[-1] + offset, label_x_coords[i])
        # Join the seismograph to its label with a straight line.
        ax.plot([x[-1] + offset, label_x_coords[i]],
                [t[-1], label_t_coords[i]],
                color = color,
                **line_kwargs)

        if offset_mode == 'uniform':

            ax.plot([x[0] + offset, offset_true], [t[0], t_unif],
                        color = color, **line_kwargs)

        # Label the trace.
        ax.annotate(trace.id, (label_x_coords[i], label_t_coords[i]), xycoords = 'data',
                ha = 'right',
                va = 'top',
                rotation = -90.0,
                fontsize = 8,
                color = color)

    if mode_info is not None:
        
        n_t_span = 100
        t_span = np.linspace(stream[0].times()[0],
                    stream[0].times()[-1], num = n_t_span)

        num_modes = len(mode_info['n'])

        # Create colour scale.
        c_map = plt.get_cmap('rainbow_r')
        if show_phase_vel:

            n_lines = 20
            c_norm = mpl_colors.Normalize(vmin = 0.0, vmax = (n_lines - 1.0))

        else:

            c_norm = mpl_colors.Normalize(vmin = 0.0, vmax = (num_modes - 1.0))

        for i in range(num_modes):

            if show_group_vel:

                v_g = mode_info['u'][i] # km/s
                v_g = v_g*1.0E3/m_to_deg

                color = c_map(c_norm(i))
                
                x_span = v_g*t_span
                section = np.floor(((x_span/180.0) + 1.0)/2.0).astype(np.int)
                section_list = np.unique(section)
                n_sections = len(section_list)

                for j in range(n_sections):

                    if j == 0:

                        #label = '$u$ $_{{{:d}}}S_{{{:d}}}$'.format(mode_info['n'][i], mode_info['l'][i])
                        label = '$_{{{:d}}}S_{{{:d}}}$'.format(mode_info['n'][i], mode_info['l'][i])

                    else:

                        label = None

                    #ax.plot([0.0, x_lims[1]], [0.0, (x_lims[1]/v_g)*t_scale], label = label)
                    k = np.where(section == section_list[j])[0]

                    offset = -360.0*j
                    
                    ax.plot(x_span[k] + offset, t_span[k]*t_scale, label = label, c = color)
                    ax.plot(-1.0*(x_span[k] + offset), t_span[k]*t_scale, c = color)

            if show_phase_vel:

                v_g = mode_info['u'][i] # km/s
                v_g = v_g*1.0E3/m_to_deg

                v_p = mode_info['c'][i] # km/s
                v_p = v_p*1.0E3/m_to_deg

                omega = mode_info['f'][i]*1.0E-3*2.0*np.pi # rad/s
                x_span_g = v_g*t_span

                #color = c_map(c_norm(i))


                initial_phase = -1.0*np.array(list(range(n_lines)))*2.0*np.pi
                for k in range(n_lines):

                    x_span = v_p*(t_span + initial_phase[k]/omega)

                    i_good = np.where(np.abs(x_span - x_span_g) < 20.0)[0]

                    section = np.floor(((x_span/180.0) + 1.0)/2.0).astype(np.int)
                    section_list = np.unique(section)
                    n_sections = len(section_list)

                    color = c_map(c_norm(k))

                    for j in range(n_sections):

                        if (j == 0) and (k == 0):

                            label = '$c$ $_{{{:d}}}S_{{{:d}}}$'.format(mode_info['n'][i], mode_info['l'][i])

                        else:

                            label = None

                        #ax.plot([0.0, x_lims[1]], [0.0, (x_lims[1]/v_p)*t_scale], label = label)
                        p = np.where(section == section_list[j])[0]
                        q = np.intersect1d(p, i_good)
                        #q = p

                        offset = -360.0*j
                        
                        ax.plot(x_span[q] + offset, t_span[q]*t_scale, label = label, c = color)
                        ax.plot(-1.0*(x_span[q] + offset), t_span[q]*t_scale, c = color)

        if show_group_vel:

            ax.legend()

    # Tidy up the axes.
    ax.xaxis.set_label_position('top')
    ax.xaxis.set_ticks_position('top')
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    if plot_type == 'delta_section':

        buff = 5.0

    elif plot_type == 'depth_section':

        buff = 500.0
        
    ax.set_xlim([x_lims[0] - buff, x_lims[1] + buff])
    if offset_mode == 'uniform':

        ax.set_ylim([t_unif, plt.ylim()[-1]])

    else:

        ax.set_ylim([(stream_start_time - t_ref)*t_scale, plt.ylim()[-1]])

    #ax.set_xticks(np.array(range(10))*20.0)
    if plot_type == 'delta_section':
        
        x_label = 'Epicentral distance (degrees)'
        #title = 'Depth: {:>6.1f} km'.format(stream[0].stats.rel_coords.depth_m*1.0E-3)
        title = 'Radius: {:>6.1f} km'.format(6371.0 - stream[0].stats.rel_coords.depth_m*1.0E-3)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(20.0))


    elif plot_type == 'depth_section':
        
        x_label = 'Radial coordinate (km)'
        title = 'Epicentral distance: {:>5.1f} degrees'.format(stream[0].stats.rel_coords.epi_dist_m/m_to_deg)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1000.0))

    plt.title(title)

    ax.set_xlabel(x_label, fontsize = font_size_label)
    ax.invert_yaxis()
    ax.set_ylabel('Time relative to centroid (hours)', fontsize = font_size_label, rotation = -90.0, labelpad = 20)

    # Save (if requested).
    if path_out is not None:

        print('Saving to {:}'.format(path_out))
        plt.savefig(path_out, dpi = 300)

    plt.show()

    return

def main():

    # Parse input arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("path_mode_input", help = "File path (relative or absolute) to Mineos mode input file.")
    parser.add_argument("path_summation_input", help = "File path (relative or absolute) to Mineos summation input file.")
    #parser.add_argument("--spectrum", action = 'store_true', help = "Plot Fourier spectrum (default: time series).")
    #parser.add_argument("--spec_and_trace", action = 'store_true', help = "Plot Fourier spectrum and time series in a single plot (default: time series only).")
    parser.add_argument("--use_mineos", action = 'store_true', help = 'Plot summation result from Mineos (default: Ouroboros).')
    parser.add_argument("--path_comparison", help = 'Path to a real data trace to be plotted for comparison. Should have units of nm/s, or provide the --comparison_scale flag with a number to multiply the comparison trace so that the units are nm/s.')
    parser.add_argument("--comparison_scale", type = float, default = 1.0)
    #
    input_args = parser.parse_args()
    path_mode_input = input_args.path_mode_input
    path_summation_input = input_args.path_summation_input
    use_mineos = input_args.use_mineos
    path_comparison = input_args.path_comparison
    comparison_scale = input_args.comparison_scale

    if not use_mineos:

        raise NotImplementedError

    else:

        # Read the mode input file.
        run_info = read_Mineos_input_file(path_mode_input)

        # Read the summation input file. 
        summation_info = read_Mineos_summation_input_file(path_summation_input)
        if summation_info['f_lims'] == 'same':

            summation_info['f_lims'] = run_info['f_lims']

        # Add information about directories.
        run_info['dir_model'], run_info['dir_run'] = get_Mineos_out_dirs(run_info) 
        summation_info = get_Mineos_summation_out_dirs(run_info, summation_info)

        dir_plot = os.path.join(run_info['dir_run'], 'plots')
        dir_sac = os.path.join(summation_info['dir_cmt'], 'sac')

        # Read moment tensor file.
        cmt_info = read_mineos_cmt(summation_info['path_cmt'])

        # Read miniSEED file. 
        path_mseed = os.path.join(dir_sac, 'stream.mseed')
        print('Reading {:}'.format(path_mseed))
        stream = read(path_mseed)

        # Read channel file.
        inv_dict = read_channel_file(summation_info['path_channels'])

        # Assign relative coordinates.
        stream = add_epi_dist_and_azim(inv_dict, cmt_info, stream)

    plot_gather(stream, UTCDateTime(cmt_info['datetime_ref']))

    return

if __name__ == '__main__':

    main()
