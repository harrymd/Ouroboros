import argparse

import matplotlib.pyplot as plt
import numpy as np
from obspy import read

from Ouroboros.plot.plot_summation import do_fft

def align_stream(stream, d_t):

    t_0 = min([trace.stats.starttime for trace in stream])
    t_1_actual = max([trace.stats.endtime for trace in stream])

    t_span_actual = t_1_actual - t_0
    n_t = int(np.ceil(t_span_actual/d_t)) + 1
    t_len = (n_t - 1)*d_t
    t_1 = t_0 + t_len

    t_span = np.linspace(0.0, t_len, num = n_t)

    stream.trim(t_0, t_1, pad = True, fill_value = 0.0)
    
    sampling_rate = 1.0/d_t

    stream.interpolate(sampling_rate, method = 'linear')

    return stream 

def fft_stack(stream):

    n_traces = len(stream)
    first_iteration = True
    for i, trace in enumerate(stream):

        if first_iteration:

            first_iteration = False
            f, X_i = do_fft(trace.times(), trace.data)
            
            n_pts = len(X_i)
            abs_X_list = np.zeros((n_traces, n_pts))
            phase_list = np.zeros((n_traces, n_pts))
            
        else:

            _, X_i = do_fft(trace.times(), trace.data)

        abs_X_list[i, :] = np.abs(X_i)
        phase_list[i, :] = np.angle(X_i)
    

    abs_X_mean = np.mean(abs_X_list, axis = 0)
    abs_X_std  = np.std(abs_X_list, axis = 0)
    
    return f, abs_X_mean, abs_X_std

def plot_stack(f, abs_X, abs_X_std):

    font_size_label = 12

    f_scale = 1.0E3
    f = f*f_scale

    fig = plt.figure(figsize = (11.0, 8.5), constrained_layout = True)
    ax = plt.gca()

    line_kwargs = {'color' : 'black'}
    fill_kwargs = {'color' : 'black', 'alpha' : 0.5}

    ax.plot(f, abs_X, **line_kwargs)
    ax.fill_between(f, abs_X - abs_X_std, y2 = abs_X + abs_X_std, **fill_kwargs)

    ax.set_xlim([0.0, 5.0])
    ax.set_ylim([0.0, 2.0E8])

    ax.set_xlabel('Frequency (mHz)', fontsize = font_size_label)
    ax.set_ylabel('Spectral amplitude (nm s$^{-1}$ mHz$^{-1}$)', fontsize = font_size_label)

    plt.show()

    return

def main():

    # Parse input arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("path_mseed", help = "File path to MSEED file.")
    parser.add_argument("--scale", type = float, default = 1.0)
    #
    input_args = parser.parse_args()
    scale = input_args.scale
    
    print('Reading {:}'.format(input_args.path_mseed))
    stream = read(input_args.path_mseed)

    # Normalise stream.
    for trace in stream:

        trace.normalize(norm = 1.0/scale)
    
    stream = align_stream(stream, d_t = stream[0].stats.delta)

    f, abs_X, abs_X_std = fft_stack(stream)
    
    plot_stack(f, abs_X, abs_X_std)

    return

if __name__ == '__main__':

    main()
