import argparse

import matplotlib.pyplot as plt
import numpy as np
from obspy import read

from Ouroboros.plot.plot_summation import do_fft

def align_streams(str_A, str_B, d_t):

    t_0_A = min([trace.stats.starttime for trace in str_A])
    t_0_B = min([trace.stats.starttime for trace in str_B])
    t_0  = min([t_0_A, t_0_B])

    t_1_actual_A = max([trace.stats.endtime for trace in str_A])
    t_1_actual_B = max([trace.stats.endtime for trace in str_B])
    t_1_actual = max([t_1_actual_A, t_1_actual_B])

    t_span_actual = t_1_actual - t_0
    n_t = int(np.ceil(t_span_actual/d_t)) + 1
    t_len = (n_t - 1)*d_t
    t_1 = t_0 + t_len

    t_span = np.linspace(0.0, t_len, num = n_t)

    str_A.trim(t_0, t_1, pad = True, fill_value = 0.0)
    str_B.trim(t_0, t_1, pad = True, fill_value = 0.0)
    
    sampling_rate = 1.0/d_t

    str_A.interpolate(sampling_rate, method = 'linear')
    str_B.interpolate(sampling_rate, method = 'linear')

    return str_A, str_B

def freq_domain_misfit(str_A, str_B, f_lims = [-np.inf, np.inf]):

    n_traces = len(str_A)
    misfit = np.zeros(n_traces)
    
    first_iteration = True
    for i, tr_A in enumerate(str_A):

        station = tr_A.stats.station
        tr_B = str_B.select(station = station)[0]

        if first_iteration:

            first_iteration = False
            f, fft_A = do_fft(tr_A.times(), tr_A.data)
            
            i_mask = np.where((f > f_lims[1]) | (f < f_lims[0]))[0]

        else:

            _, fft_A = do_fft(tr_A.times(), tr_A.data)

        _, fft_B = do_fft(tr_B.times(), tr_B.data)
        
        # Apply mask.
        fft_A[i_mask] = 0.0
        fft_B[i_mask] = 0.0

        amp_fft_A = np.abs(fft_A)
        amp_fft_B = np.abs(fft_B)

        misfit[i] = np.sqrt(np.mean((amp_fft_A - amp_fft_B)**2.0))

    return misfit

def main():

    # Parse input arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("path_mseed_A", help = "File path to first MSEED file.")
    parser.add_argument("path_mseed_B", help = "File path to second MSEED file.")
    parser.add_argument("--scale_B", type = float, default = 1.0)
    parser.add_argument("--f_lims", nargs = 2, type = np.float, default = [-np.inf, np.inf])

    input_args = parser.parse_args()
    scale_B = input_args.scale_B
    f_lims = np.array(input_args.f_lims)*1.0E-3 # mHz to Hz.
    
    print('Reading {:}'.format(input_args.path_mseed_A))
    stream_A = read(input_args.path_mseed_A)

    print('Reading {:}'.format(input_args.path_mseed_B))
    stream_B = read(input_args.path_mseed_B)

    # Normalise stream B.
    for trace in stream_B:

        trace.normalize(norm = 1.0/scale_B)

    stream_A, stream_B = align_streams(stream_A, stream_B, d_t = 1.0/stream_A[0].stats.sampling_rate)

    # Calculate misfit.
    misfit = freq_domain_misfit(stream_A, stream_B, f_lims = f_lims)

    i_worst = np.argmax(misfit)
    i_best = np.argmin(misfit)

    print('Worst agreement: station {:4}, misfit: {:>9.3} nm/s /Hz'.format(stream_A[i_worst].stats.station, misfit[i_worst]))
    print('Best  agreement: station {:4}, misfit: {:>9.3} nm/s /Hz'.format(stream_A[i_best].stats.station, misfit[i_best]))

    return

if __name__ == '__main__':

    main()

