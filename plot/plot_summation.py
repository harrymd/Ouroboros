import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from obspy import read
from obspy.core.trace import Trace
from obspy.core.stream import Stream
from obspy.realtime.signal import scale

from Ouroboros.common import (  get_Mineos_out_dirs, get_Mineos_summation_out_dirs,
                                mkdir_if_not_exist,
                                read_Mineos_input_file, read_Mineos_summation_input_file,
                                read_Ouroboros_input_file, read_Ouroboros_summation_input_file)
from Ouroboros.misc.cmt_io import read_mineos_cmt
from Ouroboros.summation.run_summation import load_time_info

font_size_label = 12

def unpack_trace(trace, trace_comparison = None):

    t = trace.times()
    x = trace.data

    if trace_comparison is not None:

        t_offset = trace.stats.starttime - trace_comparison.stats.starttime

        t_c = trace_comparison.times() - t_offset
        x_c = trace_comparison.data

    else:

        t_c = None
        x_c = None

    return t, x, t_c, x_c

def plot_seismograph(trace, trace_comparison = None, path_out = None, ax = None, show = True, label = 'auto'):
    '''
    Trace input units assumed to be nm/s.
    '''

    if label == 'auto':

        label = auto_label(trace)

    t, x, t_c, x_c = unpack_trace(trace, trace_comparison = trace_comparison)

    t_units = 'hours'

    t_scale_dict = {'hours' : 1.0/3600.0}
    t_scale = t_scale_dict[t_units]

    if trace_comparison is None:

        line_kwargs = {'color' : 'black'}

    else:
        
        line_alpha = 0.5
        line_kwargs = {'color' : 'blue', 'alpha' : line_alpha}
        line_kwargs_comparison = {'color' : 'red', 'alpha' : line_alpha}
    
    if ax is None:

        fig = plt.figure(figsize = (8.5, 5.0), constrained_layout = True)
        ax  = plt.gca()

    ax.plot(t*t_scale, x, **line_kwargs, label = 'Synthetic')

    if trace_comparison is not None:

        t_offset = t_c

        ax.plot(t_c*t_scale, x_c, **line_kwargs_comparison, label = 'Observed')

        ax.legend(loc = 'lower right')

    ax.set_xlabel('Time ({:})'.format(t_units), fontsize = font_size_label)
    ax.set_ylabel('Velocity (nm s$^{-1}$)', fontsize = font_size_label)
    if label is not None:

        ax.text(0.9, 0.9, label, transform = ax.transAxes, ha = 'right', va = 'top')

    ax.axhline(linestyle = '-', alpha = 0.5, c = 'k')

    if path_out is not None:
        
        print('Saving figure to {:}'.format(path_out))
        plt.savefig(path_out, dpi = 300, bbox_inches = 'tight')

    if show:

        plt.show()

    return

def auto_label(trace):

    label = '{:}\n$t_{{0}}$ = {:}'.format(trace.id, trace.stats.starttime)

    return label

def do_fft(t, x):

    # Get number of samples and time spacing.
    n_t = len(x)
    d_t = np.median(np.diff(t))

    # Get frequencies and do Fourier transform.
    f = np.fft.rfftfreq(n_t, d = d_t)
    X = np.fft.rfft(x)
    # Convert to X per Hz.
    X = X*d_t

    return f, X

def plot_spectrum(trace, trace_comparison = None, path_out = None, ax_arr = None, show = True, label = 'auto'):
    '''
    Trace input units assumed to be nm/s.
    '''

    if label == 'auto':

        label = auto_label(trace)

    # Get time and displacement arrays.
    t, x, t_c, x_c = unpack_trace(trace, trace_comparison = trace_comparison)
    
    # Do Fourier transform.
    f, X = do_fft(t, x)
    abs_X = np.abs(X)
    abs_X_max = np.max(abs_X)

    # Do Fourier transform of comparison trace.
    if trace_comparison is not None:

        f_c, X_c = do_fft(t_c, x_c)
        abs_X_c = np.abs(X_c)
        abs_X_c_max = np.max(abs_X_c)

    else:

        abs_X_c_max = 0.0

    abs_X_all_max = np.max([abs_X_max, abs_X_c_max])

    if ax_arr is None:

        fig, ax_arr = plt.subplots(2, 1,
                        figsize = (8.5, 6.0),
                        sharex = True,
                        gridspec_kw = {'height_ratios': [1, 2]},
                        constrained_layout = True)

    else:

        ax_arr[0].get_shared_x_axes().join(*ax_arr)

    f_scale = 1.0E3
    f_lims = [0.0, 5.0]
    X_scale = 1.0E-3 # 1.0E-3 for 1/Hz to 1/mHz
    font_size_label = 12
    
    line_width = 1
    alpha = 0.5
    scatter_s = 5
    if trace_comparison is None:

        line_kwargs = {'color' : 'k', 'linewidth' : 1}
        fill_kwargs = {'color' : 'k', 'alpha' : 0.5}
        scatter_kwargs = {'color' : 'k', 's' : scatter_s}

    else:

        color_synth = 'b'
        line_kwargs = {'color' : color_synth, 'linewidth' : line_width}
        fill_kwargs = {'color' : color_synth, 'alpha' : alpha}
        scatter_kwargs = {'color' : color_synth, 's' : scatter_s}
        
        color_obs = 'r'
        line_c_kwargs = {'color' : color_obs, 'linewidth' : line_width}
        fill_c_kwargs = {'color' : color_obs, 'alpha' : alpha}
        scatter_c_kwargs = {'color' : color_obs, 's' : scatter_s}

    ax = ax_arr[0]
    
    from scipy.signal import find_peaks
    #i_low_amp = np.where(abs_X < 0.1*np.max(abs_X))[0]
    angle_X = np.angle(X)
    #angle_X[i_low_amp] = np.nan

    i_peak, _ = find_peaks(abs_X, prominence = 0.01*np.max(abs(X)))
    #peak_mask = np.ones(len(abs_X), np.bool)
    #peak_mask[i_peak] = 0
    #angle_X[peak_mask] = np.nan

    ax.plot(f*f_scale, angle_X, **line_kwargs)
    ax.scatter(f[i_peak]*f_scale, angle_X[i_peak], **scatter_kwargs)

    if trace_comparison is None:

        ax.fill_between(f*f_scale, angle_X, y2 = -np.pi, **fill_kwargs)

    else:

        #i_low_amp_c = np.where(abs_X_c < 0.1*np.max(abs_X_c))[0]
        angle_X_c = np.angle(X_c)
        #angle_X_c[i_low_amp_c] = np.nan

        i_peak_c, _ = find_peaks(abs_X_c, prominence = 0.01*np.max(abs(X_c)))
        
        ax.scatter(f_c[i_peak_c]*f_scale, angle_X_c[i_peak_c], **scatter_c_kwargs)

        ax.plot(f_c*f_scale, angle_X_c, **line_c_kwargs)

    ax.set_ylabel('Phase (radians)', fontsize = font_size_label)

    ax.set_ylim([-np.pi, np.pi])
    ax.set_yticks(np.array([-1.0, -0.5, 0.0, 0.5, 1.0])*np.pi)
    ax.set_yticklabels(['-$\pi$', '-$\pi/2$', '0', '$\pi$/2', '$\pi$'])

    ax = ax_arr[1]

    ax.plot(f*f_scale, abs_X*X_scale, **line_kwargs, label = 'Synthetic')
    ax.fill_between(f*f_scale, abs_X*X_scale, **fill_kwargs)
    ax.scatter(f[i_peak]*f_scale, abs_X[i_peak]*X_scale, **scatter_kwargs)

    if trace_comparison is not None:

        ax.plot(f_c*f_scale, abs_X_c*X_scale, **line_c_kwargs, label = 'Observed')
        ax.fill_between(f_c*f_scale, abs_X_c*X_scale, **fill_c_kwargs)
        ax.scatter(f_c[i_peak_c]*f_scale, abs_X_c[i_peak_c]*X_scale, **scatter_c_kwargs)
        ax.legend()

    ax.set_ylabel('Spectral amplitude (nm s$^{-1}$ mHz$^{-1}$)', fontsize = font_size_label)

    y_lim_buff = 0.05
    y_log = False
    if y_log:


        ax.set_yscale('log')

    else:

        y_lims = [0.0, (1.0 + y_lim_buff)*abs_X_all_max*X_scale]
        ax.set_ylim(y_lims)

    ax.set_xlim(f_lims)
    if label is not None:
        ax.text(0.1, 0.9, label, transform = ax.transAxes, ha = 'left')

    ax.set_xlabel('Frequency (mHz)', fontsize = font_size_label)

    if path_out is not None:
        
        print('Saving figure to {:}'.format(path_out))
        plt.savefig(path_out, dpi = 300, bbox_inches = 'tight')

    if show:

        plt.show()

    return

def plot_seismograph_and_spectrum(trace, path_out = None, show = True, labels = ['auto', None], trace_comparison = None):

    fig, ax_arr = plt.subplots(3, 1,
                    figsize = (10.0, 8.0),
                    gridspec_kw = {'height_ratios': [2, 1, 2]},
                    constrained_layout = True)

    ax = ax_arr[0]
    plot_seismograph(trace, ax = ax, show = False, label = labels[0], trace_comparison = trace_comparison)

    sub_ax_arr = ax_arr[1:]
    plot_spectrum(trace, ax_arr = sub_ax_arr, show = False, label = labels[1], trace_comparison = trace_comparison)

    if path_out is not None:
        
        print('Saving figure to {:}'.format(path_out))
        plt.savefig(path_out, dpi = 300, bbox_inches = 'tight')

    if show:

        plt.show()

    return fig, ax_arr

def align_traces(tr_1, tr_2, d_t):

    t_0 = min(tr_1.stats.starttime, tr_2.stats.starttime)
    t_1_actual = min(tr_1.stats.endtime, tr_2.stats.endtime)

    t_span_actual = t_1_actual - t_0
    n_t = int(np.ceil(t_span_actual/d_t)) + 1
    t_span = (n_t - 1)*d_t
    t_1 = t_0 + t_span

    t_span = np.linspace(0.0, t_span, num = n_t)

    tr_1.trim(t_0, t_1, pad = True, fill_value = 0.0) 
    tr_2.trim(t_0, t_1, pad = True, fill_value = 0.0) 

    sampling_rate = 1.0/d_t

    tr_1.interpolate(sampling_rate, method = 'linear')
    tr_2.interpolate(sampling_rate, method = 'linear')

    #stream = Stream([tr_1, tr_2])
    #stream.plot()

    return tr_1, tr_2

def main():

    # Parse input arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("path_mode_input", help = "File path (relative or absolute) to mode input file.")
    parser.add_argument("path_summation_input", help = "File path (relative or absolute) to summation input file.")
    parser.add_argument("station", help = "Station code (same as used in station list file).")
    parser.add_argument("channel", help = "Channel code (same as used in station list file).")
    parser.add_argument("--spectrum", action = 'store_true', help = "Plot Fourier spectrum (default: time series).")
    parser.add_argument("--spec_and_trace", action = 'store_true', help = "Plot Fourier spectrum and time series in a single plot (default: time series only).")
    parser.add_argument("--use_mineos", action = 'store_true', help = 'Plot summation result from Mineos (default: Ouroboros).')
    parser.add_argument("--use_mineos_modes_only", action = 'store_true', help = 'Plot summation results using Ouroboros summation code with Mineos mode output. Note: This option is used only for testing purposes. Not compatible with --use_mineos flag.')
    parser.add_argument("--path_comparison", help = 'Path to a real data trace to be plotted for comparison. Should have units of nm/s, or provide the --comparison_scale flag with a number to multiply the comparison trace so that the units are nm/s.')
    parser.add_argument("--comparison_scale", type = float, default = 1.0)

    input_args = parser.parse_args()
    path_mode_input = input_args.path_mode_input
    path_summation_input = input_args.path_summation_input
    station = input_args.station
    channel = input_args.channel
    spectrum = input_args.spectrum
    spectrum_and_seismograph = input_args.spec_and_trace
    assert not (spectrum and spectrum_and_seismograph), 'Flags --spectrum and --spec_and_trace cannot be used together.'
    use_mineos = input_args.use_mineos
    use_mineos_modes_only = input_args.use_mineos_modes_only
    assert not (use_mineos and use_mineos_modes_only), 'Only one of --use_mineos and --use_mineos_modes_only may be specified.'
    path_comparison = input_args.path_comparison
    comparison_scale = input_args.comparison_scale

    if use_mineos:

        # Read the mode input file.
        run_info = read_Mineos_input_file(path_mode_input)

        # Read the summation input file. 
        summation_info = read_Mineos_summation_input_file(path_summation_input)
        if summation_info['f_lims'] == 'same':

            summation_info['f_lims'] = run_info['f_lims']

        # Add information about directories.
        run_info['dir_model'], run_info['dir_run'] = get_Mineos_out_dirs(run_info) 
        summation_info = get_Mineos_summation_out_dirs(run_info, summation_info)

        #dir_plot = os.path.join(run_info['dir_run'], 'plots')
        #mkdir_if_not_exist(dir_plot)

        dir_sac = os.path.join(summation_info['dir_cmt'], 'sac')

        # Read moment tensor file.
        cmt_info = read_mineos_cmt(summation_info['path_cmt'])

        # Read miniSEED file.
        path_mseed = os.path.join(dir_sac, 'stream.mseed')
        stream = read(path_mseed)
        stream = stream.select(station = station, channel = channel)

    else:

        if use_mineos_modes_only:

            # Read Mineos input file.
            run_info = read_Mineos_input_file(path_mode_input)

        else:

            raise NotImplementedError



        # Read the summation input file.
        summation_info = read_Ouroboros_summation_input_file(path_summation_input)

        # Get information about output dirs.
        if use_mineos_modes_only:

            run_info['dir_model'], run_info['dir_run'] = get_Mineos_out_dirs(run_info) 
            summation_info = get_Mineos_summation_out_dirs(run_info, summation_info,
                                name_summation_dir = 'summation_Ouroboros')

        else:
            
            raise NotImplementedError

        summation_info['dir_output'] = summation_info['dir_cmt']
        
        ## Load trace.
        #name = '{:}_{:}'.format(station, channel)
        #name_trace = '{:}.npy'.format(name)
        #path_trace = os.path.join(summation_info['dir_output'], name_trace)
        #trace_data = np.load(path_trace)

        ## Load timing information.
        #name_timing = 'timing.txt'
        #path_timing = os.path.join(summation_info['dir_output'], name_timing)
        #num_t, d_t = load_time_info(path_timing)
        #
        #trace_header = {'delta' : d_t, 'station' : station, 'channel' : channel}
        #trace = Trace(data = trace_data, header = trace_header)
        #
        #stream = Stream([trace])
        name_stream = 'stream.mseed'
        path_stream = os.path.join(summation_info['dir_output'], name_stream) 
        print('Reading {:}'.format(path_stream))
        stream = read(path_stream)
        stream = stream.select(station = station, channel = channel)

    trace = stream[0]

    dir_plot = os.path.join(run_info['dir_run'], 'plots')
    mkdir_if_not_exist(dir_plot)

    if path_comparison is not None:

        stream_comparison = read(path_comparison)
        print(len(stream_comparison))
        if len(stream_comparison) > 1:

            stream_comparison = stream_comparison.select(station = station,
                                    channel = channel)

        trace_comparison = stream_comparison[0]
        trace_comparison.normalize(norm = 1.0/comparison_scale)

        trace, trace_comparison = align_traces(trace, trace_comparison,
                                    d_t = trace.stats.delta)

    else:

        trace_comparison = None

    if spectrum:
        
        path_out = os.path.join(dir_plot, 'spectrum_{:}.png'.format(trace.id))
        plot_spectrum(trace, trace_comparison = trace_comparison,
                path_out = path_out,
                label = 'auto')

    elif spectrum_and_seismograph:

        path_out = os.path.join(dir_plot, 'seismograph_and_spectrum_{:}.png'
                                            .format(trace.id))
        plot_seismograph_and_spectrum(trace, path_out = path_out, show = True, trace_comparison = trace_comparison)

    else:
        
        path_out = os.path.join(dir_plot, 'seismograph_{:}.png'.format(trace.id))
        plot_seismograph(trace, trace_comparison = trace_comparison,
                path_out = path_out,
                label = 'auto')
    
    return

if __name__ == '__main__':

    main()
