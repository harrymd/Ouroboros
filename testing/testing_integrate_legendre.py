import matplotlib.pyplot as plt
import numpy as np

from summation.run_summation import associated_Legendre_func_series_no_CS_phase 

def plot_one_example():
    
    l = 26

    n_theta = 1000
    theta_span_rad = np.linspace(0.0, np.pi, num = n_theta)
    theta_span_deg = np.rad2deg(theta_span_rad)
    cos_theta_span = np.cos(theta_span_rad)
    sin_theta_span = np.sin(theta_span_rad)
    
    Plm = np.zeros((n_theta, 3))
    for i in range(n_theta):

        Plm_series, Plm_prime_series = \
                    associated_Legendre_func_series_no_CS_phase(
                            2, l, cos_theta_span[i])

        Plm[i, :] = Plm_series[:, l]

    integral = np.zeros(3)
    for i in range(3):

        integral[i] = np.trapz(sin_theta_span*Plm[:, i], x = theta_span_rad)
    
    fig, ax_arr = plt.subplots(3, 2, figsize = (12.0, 12.0), sharex = True, constrained_layout = True)
    font_size_label = 13

    for i in range(3):
        
        ax = ax_arr[i, 0]
        ax.plot(theta_span_deg, Plm[:, i], label = '{:>1d}'.format(i))
        ax.set_ylabel('$P_{{{:>d},{:>d}}} (\cos \Theta)$'.format(l, i), fontsize = font_size_label)

        ax.axhline(linestyle = '-', color = 'k', alpha = 0.5)
        ax.axvline(90.0, linestyle = '-', color = 'k', alpha = 0.5)

    ax.set_xlabel('Epicentral distance, $\Theta$ (degrees)', fontsize = font_size_label)
    #ax.set_ylabel('Associated Legendre function $P_{{{:>d},m}} (\cos \Theta)$'.format(l), fontsize = font_size_label)
    
    ax.set_xlim([0.0, 180.0])

    for i in range(3):
        
        ax = ax_arr[i, 1]
        ax.plot(theta_span_deg, sin_theta_span*Plm[:, i], label = '{:>1d}'.format(i))
        ax.set_ylabel('$\sin \Theta \\times P_{{{:>d},{:>d}}} (\cos \Theta)$'.format(l, i), fontsize = font_size_label)
        ax.text(0.05, 0.9, 'I = {:>.1e}'.format(integral[i]), transform = ax.transAxes)

        ax.axhline(linestyle = '-', color = 'k', alpha = 0.5)
        ax.axvline(90.0, linestyle = '-', color = 'k', alpha = 0.5)
    
    plt.show()
    
    return

def orthogonality():
    
    l = 25

    n_theta = 1000
    theta_span_rad = np.linspace(0.0, np.pi, num = n_theta)
    theta_span_deg = np.rad2deg(theta_span_rad)
    cos_theta_span = np.cos(theta_span_rad)
    sin_theta_span = np.sin(theta_span_rad)
    
    Plm = np.zeros((n_theta, 3))
    for i in range(n_theta):

        Plm_series, Plm_prime_series = \
                    associated_Legendre_func_series_no_CS_phase(
                            2, l, cos_theta_span[i])

        Plm[i, :] = Plm_series[:, l]

    integral = np.zeros(3)
    for i in range(3):

        integral[i] = np.trapz(sin_theta_span*Plm[:, i], x = theta_span_rad)
    
    fig, ax_arr = plt.subplots(3, 1, figsize = (6.0, 12.0), sharex = True, constrained_layout = True)
    font_size_label = 13

    ax = ax_arr[0]

    ax.plot(theta_span_deg, sin_theta_span*Plm[:, 0]*Plm[:, 1])
    ax.set_ylabel('$\sin \Theta \\times P_{{{:>d},{:>d}}} \\times P_{{{:>d},{:>d}}}$'.format(l, 0, l, 1), fontsize = font_size_label)

    ax.axhline(linestyle = '-', color = 'k', alpha = 0.5)
    ax.axvline(90.0, linestyle = '-', color = 'k', alpha = 0.5)

    ax = ax_arr[1]

    ax.plot(theta_span_deg, sin_theta_span*Plm[:, 0]*Plm[:, 2])
    ax.set_ylabel('$\sin \Theta \\times P_{{{:>d},{:>d}}} \\times P_{{{:>d},{:>d}}}$'.format(l, 0, l, 2), fontsize = font_size_label)

    ax.axhline(linestyle = '-', color = 'k', alpha = 0.5)
    ax.axvline(90.0, linestyle = '-', color = 'k', alpha = 0.5)

    ax = ax_arr[2]

    ax.plot(theta_span_deg, sin_theta_span*Plm[:, 1]*Plm[:, 2])
    ax.set_ylabel('$\sin \Theta \\times P_{{{:>d},{:>d}}} \\times P_{{{:>d},{:>d}}}$'.format(l, 1, l, 2), fontsize = font_size_label)

    ax.axhline(linestyle = '-', color = 'k', alpha = 0.5)
    ax.axvline(90.0, linestyle = '-', color = 'k', alpha = 0.5)

    ax.set_xlabel('Epicentral distance, $\Theta$ (degrees)', fontsize = font_size_label)
    ax.set_xlim([0.0, 180.0])

    plt.show()
    
    return

def plot_one_example_squared():
    
    l = 26

    n_theta = 1000
    theta_span_rad = np.linspace(0.0, np.pi, num = n_theta)
    theta_span_deg = np.rad2deg(theta_span_rad)
    cos_theta_span = np.cos(theta_span_rad)
    sin_theta_span = np.sin(theta_span_rad)
    
    Plm = np.zeros((n_theta, 3))
    for i in range(n_theta):

        Plm_series, Plm_prime_series = \
                    associated_Legendre_func_series_no_CS_phase(
                            2, l, cos_theta_span[i])

        Plm[i, :] = Plm_series[:, l]

    integral = np.zeros(3)
    for i in range(3):

        integral[i] = np.trapz(sin_theta_span*(Plm[:, i]**2.0), x = theta_span_rad)
    
    fig, ax_arr = plt.subplots(3, 2, figsize = (12.0, 12.0), sharex = True, constrained_layout = True)
    font_size_label = 13

    for i in range(3):
        
        ax = ax_arr[i, 0]
        ax.plot(theta_span_deg, Plm[:, i]**2.0, label = '{:>1d}'.format(i))
        ax.set_ylabel('$P_{{{:>d},{:>d}}}^{{2}} (\cos \Theta)$'.format(l, i), fontsize = font_size_label)

        ax.axhline(linestyle = '-', color = 'k', alpha = 0.5)
        ax.axvline(90.0, linestyle = '-', color = 'k', alpha = 0.5)

    ax.set_xlabel('Epicentral distance, $\Theta$ (degrees)', fontsize = font_size_label)
    #ax.set_ylabel('Associated Legendre function $P_{{{:>d},m}} (\cos \Theta)$'.format(l), fontsize = font_size_label)
    
    ax.set_xlim([0.0, 180.0])

    for i in range(3):
        
        ax = ax_arr[i, 1]
        ax.plot(theta_span_deg, sin_theta_span*(Plm[:, i]**2.0), label = '{:>1d}'.format(i))
        ax.set_ylabel('$\sin \Theta \\times P_{{{:>d},{:>d}}}^{{2}} (\cos \Theta)$'.format(l, i), fontsize = font_size_label)
        ax.text(0.05, 0.9, 'I = {:>.1e}'.format(integral[i]), transform = ax.transAxes)

        ax.axhline(linestyle = '-', color = 'k', alpha = 0.5)
        ax.axvline(90.0, linestyle = '-', color = 'k', alpha = 0.5)
    
    plt.show()
    
    return

def plot_multi_example_squared():

    l_max = 30 

    n_theta = 1000
    theta_span_rad = np.linspace(0.0, np.pi, num = n_theta)
    theta_span_deg = np.rad2deg(theta_span_rad)
    cos_theta_span = np.cos(theta_span_rad)
    sin_theta_span = np.sin(theta_span_rad)
    
    Plm = np.zeros((n_theta, l_max + 1, 3))
    for i in range(n_theta):

        Plm_series, Plm_prime_series = \
                    associated_Legendre_func_series_no_CS_phase(
                            2, l_max, cos_theta_span[i])

        for l in range(l_max + 1):

            Plm[i, l, :] = Plm_series[:, l]

    integral = np.zeros((l_max + 1, 3))
    for l in range(l_max + 1):

        for i in range(3):

            integral[l, i] = np.trapz(sin_theta_span*(Plm[:, l, i]**2.0), x = theta_span_rad)

    #for l in range(l_max + 1):

    #    for i in range(3):

    #        print(l, i, integral[l, i])

    l_span = np.array(list(range(l_max + 1)), dtype = np.int)
    
    #f0 = (2*l_span)/(l_span*((2*l_span) + 1))
    #f0 = 2/((2*l_span) + 1)
    #print(integral[:, 0]/f0)

    #f1 = 2*l_span*(l_span + 1)/(2*l_span + 1)
    #print(integral[:, 1]/f1)

    #f2 = 2*l_span*l_span*l_span*(l_span + 1)/(2*l_span + 1)
    #f2 = 2*l_span*(l_span - 1)*(l_span + 1)
    f2 = 2*(l_span - 1)*(l_span + 0)*(l_span + 1)*(l_span + 2)/((2*l_span) + 1)
    print(integral[:, 2]/f2)

    fig, ax_arr = plt.subplots(3, 1, figsize = (6.0, 12.0), sharex = True, constrained_layout = True)

    font_size_label = 13

    for i in range(3):

        ax = ax_arr[i]

        ax.plot(l_span, integral[:, i])

        ax.set_ylabel('$\int_{{0}}^{{2\pi}} \sin\Theta \cdot P_{{\ell,{:>d}}}^{{2}} (\cos \Theta)$'.format(i), fontsize = font_size_label)
    

    ax.set_xlabel('Angular order, $\ell$', fontsize = font_size_label)

    plt.show()

    return

def main():

    #plot_one_example()
    #plot_one_example_squared()
    plot_multi_example_squared()
    #orthogonality()

    return 

if __name__ == '__main__':

    main()
