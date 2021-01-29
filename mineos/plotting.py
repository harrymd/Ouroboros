import matplotlib.pyplot as plt
import numpy as np

from process_mineos import read_sac_output

def plot_waveform(dir_sac, station):

    #stream = read_sac_output('/Users/hrmd_work/Documents/research/stoneley/output/mineos/prem_noocean/00015_00021_2/sac/', 'ANMO')
    stream = read_sac_output(dir_sac, station)

    trace = stream[0]
    z = trace.data
    t = trace.times() 
    h = t/(60.0*60.0)

    nearest_power_of_10 = np.round(np.log10(np.max(np.abs(z))))

    fig = plt.figure(figsize = (11.0, 5.5))
    ax = plt.gca()

    ax.plot(h, z/(10.0**nearest_power_of_10), lw = 1)

    font_size_label = 12
    ax.set_xlabel('Time (hours)', size = font_size_label)
    ax.set_ylabel('Acceleration (10$^{{{:>d}}}$ nm s$^{{-2}}$)'.format(int(nearest_power_of_10)), size = font_size_label)

    ax.axhline(0.0, alpha = 0.3, color = 'k')
    #ax.set_xlim([h[0], h[-1]])
    ax.set_xlim([0.0, 24.0])

    plt.tight_layout()

    plt.savefig('plots/waveform.png', dpi = 300)

    plt.show()
    
    #stream.plot()
    #print(stream)

    return

def plot_spectrum(dir_sac, station):

    #stream = read_sac_output('/Users/hrmd_work/Documents/research/stoneley/output/mineos/prem_noocean/00015_00021_2/sac/', 'ANMO')
    stream = read_sac_output(dir_sac, station)

    trace = stream[0]
    z = trace.data
    t = trace.times() 

    h = t/(60.0*60.0)
    h_lims = [0.0, 24.0]
    #h_lims = [12.0, 36.0]

    i_use = (h > h_lims[0]) & (h < h_lims[1])

    z = z[i_use]

    Z = np.fft.rfft(z)
    f = np.fft.rfftfreq(len(z), d = trace.stats.delta)


    Z = Z/trace.stats.delta # Convert to X per Hz.

    #nearest_power_of_10 = np.round(np.log10(np.max(np.abs(z))))

    mode_data = np.loadtxt('mode_data_tmp.txt')

    mode_n = mode_data[:, 0].astype(np.int)
    mode_l = mode_data[:, 1].astype(np.int)
    mode_f = mode_data[:, 3]

    i_sort_f = np.argsort(mode_f)
    mode_f = mode_f[i_sort_f]
    mode_l = mode_l[i_sort_f]
    mode_n = mode_n[i_sort_f]

    i_highlight = np.where((mode_f > 3.0) & (mode_f < 3.4) & (mode_n < 4))[0]

    fig = plt.figure(figsize = (11.0, 5.5))
    ax = plt.gca()

    #ax.plot(h, z/(10.0**nearest_power_of_10), lw = 1)

    ax.plot(1000.0*f, 10E-5*np.abs(Z))

    font_size_label = 14
    ax.set_xlabel('Frequency (mHz)', size = font_size_label)
    ax.set_ylabel('Spectral amplitude of acceleration (10$^{5}$ nm s$^{-2}$ Hz$^{-1}$)', size = font_size_label)

    #ax.axhline(0.0, alpha = 0.3, color = 'k')
    #ax.set_xlim([h[0], h[-1]])
    #ax.set_xlim([0.0, 5.5])
    ax.set_xlim([2.9, 3.5])
    ax.set_ylim([0.0, 3.0]) 

    for i in i_highlight:

        ax.axvline(mode_f[i], c = 'r', alpha = 0.5)
        ax.text(mode_f[i], 2.5 + 0.2*(i % 3), '{:d}S{:d}'.format(mode_n[i], mode_l[i]))

    plt.tight_layout()

    plt.savefig('plots/spectrum.png', dpi = 300)

    plt.show()
    
    #stream.plot()
    #print(stream)

    return

def plot_dispersion():

    data = np.loadtxt('mode_data_tmp.txt')

    n = data[:, 0].astype(np.int)
    l = data[:, 1].astype(np.int)
    f = data[:, 3]

    i_highlight = (f > 3.0) & (f < 3.4) & (n < 4)

    n_max = np.max(n)

    fig = plt.figure()
    ax  = plt.gca()

    ax.scatter(l, f, s = 3, c = 'k')

    ax.scatter(l[i_highlight], f[i_highlight], s = 10, c = 'r', zorder = 10)

    for n_ in range(n_max + 1):

        i = np.where(n == n_)[0]

        li = l[i]
        fi = f[i]

        ax.plot(li, fi, c = 'k')

    font_size_label = 14
    ax.set_xlabel('Angular order, $\ell$', fontsize = font_size_label)
    ax.set_ylabel('Frequency (mHz)', fontsize = font_size_label)

    ax.set_xlim([int(0), int(30)])
    ax.set_ylim([0.0, 4.0])

    plt.tight_layout()

    plt.savefig('plots/dispersion.png', dpi = 300)

    plt.show()

    return

def plot_spectrum_of_triangle_pulse():

    f = np.linspace(0.0, 5.5*1.0E-3, num = 100)
    omega = 2.0*np.pi*f
    
    tau = 70.0
    H = tau*((np.sinc(f*tau))**2.0)

    fig = plt.figure(figsize = (11.0, 5.5))
    ax = plt.gca()

    ax.plot(1000.0*f, H) 

    font_size_label = 14
    ax.set_xlabel('Frequency (mHz)', size = font_size_label)
    ax.set_ylabel('Spectral amplitude of source-time function (Hz$^{-1}$)', size = font_size_label)

    ax.set_xlim([0.0, 5.5])
    #ax.set_ylim([0.0, 3.0]) 

    plt.tight_layout()

    plt.savefig('plots/sourcetime_spectrum.png', dpi = 300)

    plt.show()

    return

def main():

    dir_sac = '/Users/hrmd_work/Documents/research/stoneley/output/mineos/prem_noocean/00015_00030_2/sac/'
    station = 'ANMO'

    #plot_waveform(dir_sac, station)
    plot_spectrum(dir_sac, station)

    #plot_dispersion()
    #plot_spectrum_of_triangle_pulse()

    return

if __name__ == '__main__':

    main()
