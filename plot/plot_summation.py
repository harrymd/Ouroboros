import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from obspy import read
from obspy.realtime.signal import scale

from Ouroboros.common import get_Mineos_out_dirs, get_Mineos_summation_out_dirs, mkdir_if_not_exist, read_Mineos_input_file, read_Mineos_summation_input_file
from Ouroboros.misc.cmt_io import read_mineos_cmt

font_size_label = 12

def unpack_trace(trace, trace_comparison = None):

    t = trace.times()
    x = trace.data

    if trace_comparison is not None:

        t_offset = trace.stats.starttime - trace_comparison.stats.starttime

        t_c = trace_comparison.times() + t_offset
        x_c = trace_comparison.data

    else:

        t_c = None
        x_c = None

    return t, x, t_c, x_c

def plot_seismograph(trace, trace_comparison = None, path_out = None, ax = None, show = True):
    '''
    Trace input units assumed to be nm/s.
    '''

    t, x, t_c, x_c = unpack_trace(trace, trace_comparison = trace_comparison)

    t_units = 'hours'

    t_scale_dict = {'hours' : 1.0/3600.0}
    t_scale = t_scale_dict[t_units]
    
    if ax is None:

        fig = plt.figure(figsize = (8.5, 5.0), constrained_layout = True)
        ax  = plt.gca()

    ax.plot(t*t_scale, x, label = 'Synthetic')

    if trace_comparison is not None:

        t_offset = t_c

        ax.plot(t_c*t_scale, x_c, label = 'Observed')

        ax.legend()

    ax.set_xlabel('Time ({:})'.format(t_units), fontsize = font_size_label)
    ax.set_ylabel('Velocity (nm s$^{-1}$)', fontsize = font_size_label)

    ax.axhline(linestyle = '-', alpha = 0.5)

    if path_out is not None:
        
        print('Saving figure to {:}'.format(path_out))
        plt.savefig(path_out, dpi = 300, bbox_inches = 'tight')

    if show:

        plt.show()

    return

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

def plot_spectrum(trace, trace_comparison = None, path_out = None, ax_arr = None, show = True):
    '''
    Trace input units assumed to be nm/s.
    '''

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

    if trace_comparison is None:

        line_kwargs = {'color' : 'k', 'linewidth' : 1}
        fill_kwargs = {'color' : 'k', 'alpha' : 0.5}

    else:

        line_kwargs = {'color' : 'b', 'linewidth' : 1}
        fill_kwargs = {'color' : 'b', 'alpha' : 0.5}
        
        line_c_kwargs = {'color' : 'r', 'linewidth' : 1}
        fill_c_kwargs = {'color' : 'r', 'alpha' : 0.5}

    ax = ax_arr[0]

    ax.plot(f*f_scale, np.angle(X), **line_kwargs)

    if trace_comparison is None:

        ax.fill_between(f*f_scale, np.angle(X), y2 = -np.pi, **fill_kwargs)

    else:

        ax.plot(f_c*f_scale, np.angle(X_c), **line_c_kwargs)

    ax.set_ylabel('Phase (radians)', fontsize = font_size_label)

    ax.set_ylim([-np.pi, np.pi])
    ax.set_yticks(np.array([-1.0, -0.5, 0.0, 0.5, 1.0])*np.pi)
    ax.set_yticklabels(['-$\pi$', '-$\pi/2$', '0', '$\pi$/2', '$\pi$'])

    ax = ax_arr[1]

    ax.plot(f*f_scale, abs_X*X_scale, **line_kwargs, label = 'Synthetic')
    ax.fill_between(f*f_scale, abs_X*X_scale, **fill_kwargs)

    if trace_comparison is not None:

        ax.plot(f_c*f_scale, abs_X_c*X_scale, **line_c_kwargs, label = 'Observed')
        ax.fill_between(f_c*f_scale, abs_X_c*X_scale, **fill_c_kwargs)
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
    ax.set_xlabel('Frequency (mHz)', fontsize = font_size_label)



    if path_out is not None:
        
        print('Saving figure to {:}'.format(path_out))
        plt.savefig(path_out, dpi = 300, bbox_inches = 'tight')

    if show:

        plt.show()

    return

def plot_seismograph_and_spectrum(trace, path_out = None, show = True):

    fig, ax_arr = plt.subplots(3, 1,
                    figsize = (10.0, 8.0),
                    gridspec_kw = {'height_ratios': [2, 1, 2]},
                    constrained_layout = True)

    ax = ax_arr[0]
    plot_seismograph(trace, ax = ax, show = False)

    sub_ax_arr = ax_arr[1:]
    plot_spectrum(trace, ax_arr = sub_ax_arr, show = False)

    if path_out is not None:
        
        print('Saving figure to {:}'.format(path_out))
        plt.savefig(path_out, dpi = 300, bbox_inches = 'tight')

    if show:

        plt.show()

    return fig, ax_arr

def main():

    # Parse input arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("path_mode_input", help = "File path (relative or absolute) to Mineos mode input file.")
    parser.add_argument("path_summation_input", help = "File path (relative or absolute) to Mineos summation input file.")
    parser.add_argument("station", help = "Station code (same as used in station list file).")
    parser.add_argument("channel", help = "Channel code (same as used in station list file).")
    parser.add_argument("--spectrum", action = 'store_true', help = "Plot Fourier spectrum (default: time series).")
    parser.add_argument("--use_mineos", action = 'store_true', help = 'Plot summation result from Mineos (default: Ouroboros).')
    parser.add_argument("--path_comparison", help = 'Path to a real data trace to be plotted for comparison. Should have units of nm/s, or provide the --comparison_scale flag with a number to multiply the comparison trace so that the units are nm/s.')
    parser.add_argument("--comparison_scale", type = float, default = 1.0)

    input_args = parser.parse_args()
    path_mode_input = input_args.path_mode_input
    path_summation_input = input_args.path_summation_input
    station = input_args.station
    channel = input_args.channel
    spectrum = input_args.spectrum
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

        # Get path to SAC file.
        src_time_year_day_str = cmt_info['datetime_ref'].strftime('%Y%j')
        src_time_hms_str = cmt_info['datetime_ref'].strftime('%H:%M:%S').replace('0', ' ')
        #syndat_out.2011070: 5:46:23.ANMO.LHZ.SAC
        #syndat_out.2011070:05:46:23..LHZ.SAC
        file_sac = 'syndat_out.{:}:{:}.{:}.{:}.SAC'.format(src_time_year_day_str,
                    src_time_hms_str, station, channel)
        path_sac = os.path.join(dir_sac, file_sac)

        # Read SAC file.
        stream = read(path_sac)

    if path_comparison is not None:

        stream_comparison = read(path_comparison)
        trace_comparison = stream_comparison[0]
        trace_comparison.normalize(norm = 1.0/comparison_scale)

    else:

        trace_comparison = None
    
    trace = stream[0]

    if spectrum:
        
        path_out = os.path.join(dir_plot, 'spectrum.png')
        plot_spectrum(trace, trace_comparison = trace_comparison,
                path_out = path_out)

    else:
        
        path_out = os.path.join(dir_plot, 'seismograph.png')
        plot_seismograph(trace, trace_comparison = trace_comparison,
                path_out = path_out)
    
    return

if __name__ == '__main__':

    main()
