'''
Plot synthetic seismograms created by normal-mode summation.
'''

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from obspy import read
from obspy.core.trace import Trace
from obspy.core.stream import Stream
from obspy.realtime.signal import scale
from scipy.signal import find_peaks

from Ouroboros.common import (  get_Mineos_out_dirs, get_Mineos_summation_out_dirs,
                                get_Ouroboros_out_dirs, get_Ouroboros_summation_out_dirs,
                                mkdir_if_not_exist,
                                read_Mineos_input_file, read_Mineos_summation_input_file,
                                read_Ouroboros_input_file, read_Ouroboros_summation_input_file)
from Ouroboros.misc.cmt_io import read_mineos_cmt
from Ouroboros.summation.run_summation import load_time_info

font_size_label = 12

def unpack_trace(trace, trace_comparison = None):
    '''
    Convert a trace to t- and x-arrays.
    '''

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

def plot_seismograph(trace, trace_comparison = None, path_out = None, ax = None, show = True, label = 'auto', show_legend = True, y_label = None, legend_keys = ['Synthetic', 'Observed']):
    '''
    Plot seismograph from trace.
    Trace input units assumed to be nm/s.
    '''

    if label == 'auto':

        label = auto_label(trace)

    # Get NumPy arrays.
    t, x, t_c, x_c = unpack_trace(trace, trace_comparison = trace_comparison)

    # Set t-axis units and scale t-values accordingly.
    t_units = 'hours'
    t_scale_dict = {'hours' : 1.0/3600.0}
    t_scale = t_scale_dict[t_units]

    # Set line colours.
    if trace_comparison is None:

        line_kwargs = {'color' : 'black'}

    else:
        
        line_alpha = 0.5
        line_kwargs = {'color' : 'blue', 'alpha' : line_alpha}
        line_kwargs_comparison = {'color' : 'red', 'alpha' : line_alpha}
    
    # Create axes if needed.
    if ax is None:

        fig = plt.figure(figsize = (8.5, 5.0), constrained_layout = True)
        ax  = plt.gca()

    # Plot the time series.
    ax.plot(t*t_scale, x, **line_kwargs, label = legend_keys[0])

    # Plot the comparison time series.
    if trace_comparison is not None:

        t_offset = t_c

        ax.plot(t_c*t_scale, x_c, **line_kwargs_comparison, label = legend_keys[1])

        if show_legend:

            ax.legend(loc = 'lower right')

    # Label axes and plot.
    ax.set_xlabel('Time ({:})'.format(t_units), fontsize = font_size_label)
    if y_label is not None:

        ax.set_ylabel(y_label, fontsize = font_size_label)

    if label is not None:

        ax.text(0.9, 0.9, label, transform = ax.transAxes, ha = 'right', va = 'top')

    # Draw line for zero displacement.
    ax.axhline(linestyle = '-', alpha = 0.5, c = 'k')
    
    if trace_comparison is not None:

        t_min = np.min(np.concatenate([t_c, t]))
        t_max = np.max(np.concatenate([t_c, t]))

    else:

        t_min = np.min(t)
        t_max = np.max(t)

    # Set axis limits.
    t_lims = np.array([t_min, t_max])
    ax.set_xlim(t_lims*t_scale)

    # Save.
    if path_out is not None:
        
        print('Saving figure to {:}'.format(path_out))
        plt.savefig(path_out, dpi = 300, bbox_inches = 'tight')

    # Show.
    if show:

        plt.show()

    return

def auto_label(trace):
    '''
    Make label based on trace start time and ID.
    '''

    label = '{:}\n$t_{{0}}$ = {:}'.format(trace.id, trace.stats.starttime)

    return label

def do_fft(t, x):
    '''
    Calculate Fourier transform and convert to spectral amplitude.
    '''

    # Get number of samples and time spacing.
    n_t = len(x)
    d_t = np.median(np.diff(t))

    # Get frequencies and do Fourier transform.
    f = np.fft.rfftfreq(n_t, d = d_t)
    X = np.fft.rfft(x)
    # Convert to X per Hz.
    X = X*d_t

    return f, X

def plot_spectrum(trace, trace_comparison = None, path_out = None, ax_arr = None, show = True, label = 'auto', label_coeff_info = None, y_label = None, legend_keys = ['Synthetic', 'Observed']):
    '''
    Plot a spectrum. The spectrum has two parts: amplitude and phase.
    Trace input units assumed to be nm/s.
    '''

    # Add label.
    if label == 'auto':

        label = auto_label(trace)

    # Get time and displacement arrays.
    t, x, t_c, x_c = unpack_trace(trace, trace_comparison = trace_comparison)
    
    # Do Fourier transform.
    # Get amplitude spectrum, maximum amplitude, phase, and location of peaks.
    f, X = do_fft(t, x)
    abs_X = np.abs(X)
    abs_X_max = np.max(abs_X)
    angle_X = np.angle(X)
    prominence_factor = 1.0E-3
    i_peak, _ = find_peaks(abs_X, prominence = prominence_factor*abs_X_max)

    # Do Fourier transform of comparison trace.
    if trace_comparison is not None:

        f_c, X_c = do_fft(t_c, x_c)
        abs_X_c = np.abs(X_c)
        abs_X_c_max = np.max(abs_X_c)
        angle_X_c = np.angle(X_c)
        i_peak_c, _ = find_peaks(abs_X_c, prominence = prominence_factor*abs_X_c_max)


    else:

        abs_X_c_max = 0.0
    
    # Get overall maximum amplitude.
    abs_X_all_max = np.max([abs_X_max, abs_X_c_max])

    # Create axes.
    if ax_arr is None:

        fig, ax_arr = plt.subplots(2, 1,
                        figsize = (8.5, 6.0),
                        sharex = True,
                        gridspec_kw = {'height_ratios': [1, 2]},
                        constrained_layout = True)

    # Enforce shared x axis if axes are provided.
    else:

        ax_arr[0].get_shared_x_axes().join(*ax_arr)

    # Set plot properties.
    f_scale = 1.0E3 # Hz to mHz.
    f_lims = [0.0, 5.0]
    #f_lims = [0.0, 2.0]
    X_scale = 1.0E-3 # 1.0E-3 for 1/Hz to 1/mHz
    font_size_label = 12
    line_width = 1
    alpha = 0.5
    scatter_s = 5
    if trace_comparison is None:

        line_kwargs = {'color' : 'k', 'linewidth' : 1}
        fill_kwargs = {'color' : 'k', 'alpha' : alpha}
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

    # Plot phase.
    ax = ax_arr[0]

    # Plot phase.
    ax.plot(f*f_scale, angle_X, **line_kwargs)
    ax.scatter(f[i_peak]*f_scale, angle_X[i_peak], **scatter_kwargs)

    # If no comparison trace, also fill phase curve.
    if trace_comparison is None:

        ax.fill_between(f*f_scale, angle_X, y2 = -np.pi, **fill_kwargs)

    # Plot phase of comparison trace.
    else:

        ax.scatter(f_c[i_peak_c]*f_scale, angle_X_c[i_peak_c], **scatter_c_kwargs)
        ax.plot(f_c*f_scale, angle_X_c, **line_c_kwargs)
    
    # Tidy up phase plot.
    ax.set_ylabel('Phase (radians)', fontsize = font_size_label)
    ax.set_ylim([-np.pi, np.pi])
    ax.set_yticks(np.array([-1.0, -0.5, 0.0, 0.5, 1.0])*np.pi)
    ax.set_yticklabels(['-$\pi$', '-$\pi/2$', '0', '$\pi$/2', '$\pi$'])

    # Plot amplitude spectrum.
    ax = ax_arr[1]
    #
    ax.plot(f*f_scale, abs_X*X_scale, **line_kwargs, label = legend_keys[0])
    ax.fill_between(f*f_scale, abs_X*X_scale, **fill_kwargs)
    ax.scatter(f[i_peak]*f_scale, abs_X[i_peak]*X_scale, **scatter_kwargs)

    # Plot amplitude spectrum of comparison trace.
    if trace_comparison is not None:

        ax.plot(f_c*f_scale, abs_X_c*X_scale, **line_c_kwargs, label = legend_keys[1])
        ax.fill_between(f_c*f_scale, abs_X_c*X_scale, **fill_c_kwargs)
        ax.scatter(f_c[i_peak_c]*f_scale, abs_X_c[i_peak_c]*X_scale, **scatter_c_kwargs)
        ax.legend()

    # Tidy up amplitude spectrum.
    if y_label is not None:

        #ax.set_ylabel('Spectral amplitude (nm s$^{-1}$ mHz$^{-1}$)', fontsize = font_size_label)
        ax.set_ylabel(y_label, fontsize = font_size_label)

    y_lim_buff = 0.05
    y_log = False
    if y_log:

        ax.set_yscale('log')

    else:

        y_lims = [0.0, (1.0 + y_lim_buff)*abs_X_all_max*X_scale]
        ax.set_ylim(y_lims)
        #ax.set_ylim([0.0, 2.0E3])

    # Label prominent peaks in the spectrum.
    if label_coeff_info is not None:
        
        # Get values of coefficients.
        coeffs = label_coeff_info['coeffs']
        mode_info = label_coeff_info['modes']

        # Find modes with coefficients greater than a certain threshold.
        amp_thresh_frac = 0.1
        key = 'A_r'
        abs_coeff = np.abs(coeffs[key])
        max_coeff = np.max(abs_coeff)
        cond_amp = np.array((abs_coeff > amp_thresh_frac*max_coeff), dtype = np.bool)

        # Find modes within frequency range.
        cond_freq = np.array(((mode_info['f'] > f_lims[0]) & (mode_info['f'] < f_lims[1])), dtype = np.bool)

        # Find modes satisfying both conditions.
        i_label =  np.where(cond_amp & cond_freq)[0]

        # Label each mode satisfying the conditions.
        for i in i_label:
            
            mode_str = '$_{{{:>d}}}{:}_{{{:>d}}}$'.format(mode_info['n'][i], mode_info['type'][i], mode_info['l'][i])
            ax.text(mode_info['f'][i], y_lims[1]*0.9, mode_str, rotation = 90.0, va = 'bottom', ha = 'center')
            ax.plot([mode_info['f'][i], mode_info['f'][i]], [0.0, y_lims[1]*0.9], color = 'k', lw = 1)

    # Set x-limits.
    ax.set_xlim(f_lims)

    # Label.
    if label is not None:

        ax.text(0.1, 0.8, label, transform = ax.transAxes, ha = 'left')

    ax.set_xlabel('Frequency (mHz)', fontsize = font_size_label)

    # Save (if requested).
    if path_out is not None:
        
        print('Saving figure to {:}'.format(path_out))
        plt.savefig(path_out, dpi = 300, bbox_inches = 'tight')

    # Show (if requested).
    if show:

        plt.show()

    return

def plot_seismograph_and_spectrum(trace, path_out = None, show = True, labels = ['auto', None], trace_comparison = None, label_coeff_info = None, y_labels = [None, None], legend_keys = ['Synthetic', 'Observed']):
    '''
    Make a combined plot showing a time series and a spectrum.
    '''

    # Create axes.
    fig, ax_arr = plt.subplots(3, 1,
                    figsize = (10.0, 8.0),
                    gridspec_kw = {'height_ratios': [2, 1, 2]},
                    constrained_layout = True)

    # Plot time series in first axes.
    ax = ax_arr[0]
    plot_seismograph(trace, ax = ax, show = False, label = labels[0],
            trace_comparison = trace_comparison, show_legend = False,
            y_label = y_labels[0])

    # Plot spectrum in lower two axes.
    sub_ax_arr = ax_arr[1:]
    plot_spectrum(trace, ax_arr = sub_ax_arr, show = False, label = labels[1],
            trace_comparison = trace_comparison,
            label_coeff_info = label_coeff_info, y_label = y_labels[1],
            legend_keys = legend_keys)

    # Save.
    if path_out is not None:
        
        print('Saving figure to {:}'.format(path_out))
        plt.savefig(path_out, dpi = 300, bbox_inches = 'tight')

    # Show.
    if show:

        plt.show()

    return fig, ax_arr

def align_traces(tr_1, tr_2, d_t):
    '''
    Align two traces, given their start times.
    Exact alignment of time sample points is important for comparison of
    spectral peaks and phase values.
    '''

    # Find start and end times.
    t_0 = min(tr_1.stats.starttime, tr_2.stats.starttime)
    t_1_actual = min(tr_1.stats.endtime, tr_2.stats.endtime)

    t_span_actual = t_1_actual - t_0
    n_t = int(np.ceil(t_span_actual/d_t)) + 1
    t_span = (n_t - 1)*d_t
    buff = 1.0*d_t
    t_1 = t_0 + t_span + buff

    # Get time array.
    t_span = np.linspace(0.0, t_span, num = n_t)

    # Interpolate the traces at the chosen time values.
    tr_1_data_interpolated = np.interp(t_span, tr_1.times(), tr_1.data)
    tr_2_data_interpolated = np.interp(t_span, tr_2.times(), tr_2.data)

    # This is a hack; for unknown reasons Mineos and Ouroboros differ
    # at the very start of their traces.
    #tr_1_data_interpolated[0:6] = 0.0
    #tr_2_data_interpolated[0:6] = 0.0
    
    # Create new traces containing the interpolated values.
    tr_1_copy = tr_1.copy()
    tr_1_copy.data = tr_1_data_interpolated
    tr_1_copy.stats.starttime = t_0
    tr_1 = tr_1_copy

    tr_2_copy = tr_2.copy()
    tr_2_copy.data = tr_2_data_interpolated
    tr_2_copy.stats.starttime = t_0
    tr_2 = tr_2_copy

    # Check the traces are the same length and have the same time bounds.
    assert tr_1.stats.npts == tr_2.stats.npts
    assert tr_1.stats.starttime == tr_2.stats.starttime
    assert tr_1.stats.endtime == tr_2.stats.endtime

    return tr_1, tr_2

def main():

    # Mappings between different variable names.
    output_type_to_char = {'displacement' : 's', 'velocity' : 'v',
                'acceleration' : 'a'}
    mineos_output_type_to_char = {0 : 's', 1 : 'v', 2 : 'a'}
    mineos_output_type_to_data_type = { 0 : 'displacement', 1 : 'velocity',
                                        2 : 'acceleration'}

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
    parser.add_argument("--label", action = 'store_true', help = 'Add labels to modes with excitation coefficients above a certain threshold.')
    parser.add_argument("--path_comparison", help = 'Path to a real data trace to be plotted for comparison. Should have units of nm/s, or provide the --comparison_scale flag with a number to multiply the comparison trace so that the units are nm/s.')
    parser.add_argument("--comparison_scale", type = float, default = 1.0)
    #
    input_args = parser.parse_args()
    #
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
    add_labels = input_args.label
    assert not (use_mineos and add_labels), 'Cannot add mode labels to Mineos plot (excitation coefficients are not available).'

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

        # Read moment tensor file.
        cmt_info = read_mineos_cmt(summation_info['path_cmt'])

        # Read miniSEED file.
        name_stream = 'stream_{:}.mseed'.format(mineos_output_type_to_char[
                            summation_info['data_type']])
        dir_sac = os.path.join(summation_info['dir_cmt'], 'sac')
        path_mseed = os.path.join(dir_sac, name_stream)
        stream = read(path_mseed)
        stream = stream.select(station = station, channel = channel)

        if add_labels:

            raise NotImplementedError

        else:

            label_coeff_info = None

    else:

        if use_mineos_modes_only:

            # Read Mineos input file.
            run_info = read_Mineos_input_file(path_mode_input)

        else:

            run_info = read_Ouroboros_input_file(path_mode_input)


        # Read the summation input file.
        summation_info = read_Ouroboros_summation_input_file(path_summation_input)

        # Get information about output dirs.
        if use_mineos_modes_only:

            run_info['dir_model'], run_info['dir_run'] = get_Mineos_out_dirs(run_info) 
            summation_info = get_Mineos_summation_out_dirs(run_info, summation_info,
                                name_summation_dir = 'summation_Ouroboros')

        else:
            
            run_info['dir_model'], run_info['dir_run'], _, _ = get_Ouroboros_out_dirs(run_info, 'none')

            summation_info = get_Ouroboros_summation_out_dirs(run_info, summation_info)

        summation_info['dir_output'] = summation_info['dir_cmt']
        
        # Load stream.
        name_stream = 'stream_{:}.mseed'.format(output_type_to_char[
                            summation_info['output_type']])
        path_stream = os.path.join(summation_info['dir_output'], name_stream) 
        print('Reading {:}'.format(path_stream))
        stream = read(path_stream)
        stream = stream.select(station = station, channel = channel)

        # Load coefficient information.
        if add_labels:

            path_coeffs = os.path.join(summation_info['dir_output'], 'coeffs.pkl')
            coeffs = np.load(path_coeffs, allow_pickle = True)
            label_coeffs = coeffs.loc[station]
            
            path_modes = os.path.join(summation_info['dir_output'], 'modes.pkl')
            modes = np.load(path_modes, allow_pickle = True)

            label_coeff_info = {'coeffs' : coeffs, 'modes' : modes}

        else:

            label_coeff_info = None
    
    # Determine data type and from this get axis labels.
    if use_mineos:

        data_type = mineos_output_type_to_data_type[summation_info['data_type']]

    else:

        data_type = summation_info['output_type']

    if data_type == 'acceleration':

        displacement_y_label = 'Acceleration (nm s$^{-2}$)'
        spectrum_y_label = 'Spectral amplitude (nm s$^{-2}$ mHz$^{-1}$)'

    elif data_type == 'velocity':

        displacement_y_label = 'Velocity (nm s$^{-1}$)'
        spectrum_y_label = 'Spectral amplitude (nm s$^{-1}$ mHz$^{-1}$)'

    elif data_type == 'displacement':

        displacement_y_label = 'Displacement (nm)'
        spectrum_y_label = 'Spectral amplitude (nm mHz$^{-1}$)'

    else:

        raise ValueError
    
    # Define legend keys (hard-coded for now).
    legend_keys = ['Synthetic', 'Observed']
    legend_keys = ['New code', 'Mineos']
    #legend_keys = ['New code', 'Data']

    trace = stream[0]

    dir_plot = os.path.join(run_info['dir_run'], 'plots')
    mkdir_if_not_exist(dir_plot)

    # Align traces, if there are two.
    if path_comparison is not None:
        
        # Load comparison trace.
        stream_comparison = read(path_comparison)
        if len(stream_comparison) > 1:

            stream_comparison = stream_comparison.select(station = station,
                                    channel = channel)
            #stream_comparison = stream_comparison.select(station = station)

        trace_comparison = stream_comparison[0]
        trace_comparison.normalize(norm = 1.0/comparison_scale)

        trace, trace_comparison = align_traces(trace, trace_comparison,
                                    d_t = trace.stats.delta)

    else:

        trace_comparison = None

    # Plot spectrum.
    if spectrum:
        
        path_out = os.path.join(dir_plot, 'spectrum_{:}.png'.format(trace.id))
        plot_spectrum(trace, trace_comparison = trace_comparison,
                path_out = path_out,
                label = 'auto',
                label_coeff_info = label_coeff_info,
                y_label = spectrum_y_label,
                legend_keys = legend_keys)

    # Plot spectrum and time series.
    elif spectrum_and_seismograph:

        path_out = os.path.join(dir_plot, 'seismograph_and_spectrum_{:}.png'
                                            .format(trace.id))
        plot_seismograph_and_spectrum(trace, path_out = path_out, show = True, trace_comparison = trace_comparison,
                label_coeff_info = label_coeff_info,
                y_labels = [displacement_y_label, spectrum_y_label],
                legend_keys = legend_keys)

    # Plot time series.
    else:
        
        path_out = os.path.join(dir_plot, 'seismograph_{:}.png'.format(trace.id))
        plot_seismograph(trace, trace_comparison = trace_comparison,
                path_out = path_out,
                label = 'auto',
                y_label = displacement_y_label,
                legend_keys = legend_keys)
    
    return

if __name__ == '__main__':

    main()
