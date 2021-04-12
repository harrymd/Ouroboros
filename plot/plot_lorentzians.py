import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from   obspy import read
import pandas

from Ouroboros.plot.plot_summation import unpack_trace, do_fft

def lorentzian(omega, omega_mode, gamma_mode):
    '''
    Dahlen and Tromp eq. 10.65.
    '''

    denominator = 2.0*(gamma_mode + 1.0j*(omega - omega_mode))

    return 1.0/denominator

def main():

    # Parse input arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("dir_coeffs", help = "Path of directory containing coefficient file.")
    parser.add_argument("station", help = "Station code (same as used in station list file).")
    parser.add_argument("--path_comparison", help = "Path of stream file containing trace for comparison.")
    args = parser.parse_args()
    dir_coeffs = args.dir_coeffs
    station = args.station
    path_comparison = args.path_comparison

    if path_comparison is not None:

        # Read
        stream = read(path_comparison)
        stream = stream.select(station = station, channel = 'LHZ')
        assert len(stream) == 1
        trace = stream[0]

        # Get time and displacement arrays.
        t_tr, x_tr, _, _ = unpack_trace(trace)
        
        # Do Fourier transform.
        # Get amplitude spectrum, maximum amplitude, phase, and location of peaks.
        f_tr, X_tr = do_fft(t_tr, x_tr)
        f_tr_mHz = f_tr*1.0E3
        abs_X_tr = np.abs(X_tr)
        abs_X_tr_max = np.max(abs_X_tr)
        angle_X_tr = np.angle(X_tr)

    # Load mode data.
    path_modes = os.path.join(dir_coeffs, 'modes.pkl')
    print('Loading {:}'.format(path_modes))
    modes = pandas.read_pickle(path_modes)

    # Load coeff data.
    path_coeffs = os.path.join(dir_coeffs, 'coeffs.pkl')
    print('Loading {:}'.format(path_coeffs))
    coeffs = pandas.read_pickle(path_coeffs)
    #
    coeffs = coeffs.loc[station]
    mode_type_list = np.unique(modes['type'])

    f_centre = 5.4
    f_plot_half_width = 0.25
    f_lims_plot = np.array([f_centre - f_plot_half_width, f_centre + f_plot_half_width])
    f_range_plot = f_lims_plot[1] - f_lims_plot[0]
    f_lim_buff = 0.25
    f_lim_buffs = f_lim_buff*f_range_plot*np.array([-1.0, 1.0])
    f_lims = f_lims_plot + f_lim_buffs

    f_range = f_lims[1] - f_lims[0]
    d_f_apx = 0.001
    n_f = int(np.round(f_range/d_f_apx)) + 1
    d_f = f_range/(n_f - 1)
    f_span_mHz = np.linspace(f_lims[0], f_lims[1], num = n_f)
    f_span_rad_per_s = 1.0E-3*2.0*np.pi*f_span_mHz

    highlight_n_list = [2, 3]
    highlight_l_list = [25, 25]
    num_highlights = len(highlight_n_list)
    i_highlight = []
    for k in range(num_highlights):

        i = np.where((modes['n'] == highlight_n_list[k]) &
                     (modes['l'] == highlight_l_list[k]))[0][0]

        i_highlight.append(i)

    i_choose = np.where((modes['f'] < f_lims[1]) & (modes['f'] > f_lims[0]))[0]
    i_highlight = np.array(i_highlight, dtype = np.int)
    i_choose = np.union1d(i_choose, i_highlight)
    num_modes = len(i_choose)
    spectrum = np.zeros((num_modes, n_f), dtype = np.complex)
    peak_height = np.zeros(num_modes)

    for j, i in enumerate(i_choose):

        f_mHz = modes['f'][i]
        Q       = modes['Q'][i]
        A       = coeffs['A_r'][i]

        f_Hz = 1.0E-3*f_mHz
        f_rad_per_s = 2.0*np.pi*f_Hz
        gamma   = f_rad_per_s/(2.0*Q)

        spectrum[j, :] = A*lorentzian(  f_span_rad_per_s, f_rad_per_s, gamma)
        peak_height[j] = np.abs(A)/(2.0*gamma)

    fig = plt.figure(figsize = (11.0, 8.5), constrained_layout = True)
    ax  = plt.gca()

    scale = 1.0E9
    
    max_peak_height = np.max(peak_height)
    general_peak_height_thresh = 0.1*max_peak_height
    general_clip_thresh = 0.05*max_peak_height
    label_offset = 0.01*max_peak_height

    for j, i in enumerate(i_choose):

        if i in i_highlight:

            peak_height_thresh = 0.0
            clip_thresh = 0.4*peak_height[j]

        else:

            peak_height_thresh = general_peak_height_thresh
            clip_thresh = general_clip_thresh 
        
        if peak_height[j] > peak_height_thresh:
            
            abs_spectrum = np.abs(spectrum[j, :])
            k = np.where(abs_spectrum > clip_thresh)[0]

            line = ax.plot(f_span_mHz[k], abs_spectrum[k]*scale)[0] 

            if (modes['f'][i] > f_lims_plot[0]) & (modes['f'][i] < f_lims_plot[1]):

                mode_str = '$_{{{:>d}}}{:}_{{{:>d}}}$'.format(modes['n'][i], modes['type'][i], modes['l'][i])
                label = ax.text(modes['f'][i], (peak_height[j] + label_offset)*scale, mode_str,
                        rotation = 90.0,
                        va = 'bottom',
                        ha = 'center',
                        color = line.get_color(),
                        fontsize = 12)
                label.set_bbox(dict(facecolor='white', alpha=0.9, edgecolor='none'))

    spectrum_sum = np.sum(spectrum, axis = 0)
    abs_spectrum_sum = np.abs(spectrum_sum)

    ax.plot(f_span_mHz, abs_spectrum_sum*scale, c = 'k')

    max_peak_plot = np.max([max_peak_height, np.max(abs_spectrum_sum)])

    if path_comparison is not None:

        ax.plot(f_tr_mHz, abs_X_tr)

    ax.set_xlim(f_lims_plot)
    ax.set_ylim([0.0, 1.1*max_peak_plot*scale])

    font_size_label = 14
    ax.set_xlabel('Frequency (mHz)', fontsize = font_size_label)
    ax.set_ylabel('Spectral amplitude (nm s$^{-2}$)', fontsize = font_size_label)

    fig_name = 'lorentzians.png'
    print('Writing to {:}'.format(fig_name))
    plt.savefig(fig_name, dpi = 300)

    plt.show()

    return

if __name__ == '__main__':

    main()
