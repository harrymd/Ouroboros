import matplotlib.pyplot as plt
import numpy as np

from Ouroboros.summation.run_summation import associated_Legendre_func_series_no_CS_phase

def main():

    l_max = 42 
    m_max = 2
    cosTheta = 0.121335538633

    Plm_series, Plm_prime_series = \
                associated_Legendre_func_series_no_CS_phase(
                        2, l_max, cosTheta)

    cosTheta = np.cos(np.deg2rad(83.268))
    print(cosTheta)

    Plm_series_M, Plm_prime_series_M = \
                associated_Legendre_func_series_no_CS_phase(
                        2, l_max, cosTheta)
    
    l_span = list(range(l_max + 1))
    fig, ax_arr = plt.subplots(2, 1, sharex = True)

    ax = ax_arr[0]

    ax.plot(l_span, Plm_series[2, :])
    ax.plot(l_span, Plm_series_M[2, :])

    ax = ax_arr[1]

    ax.plot(l_span, Plm_prime_series[2, :])
    ax.plot(l_span, Plm_prime_series_M[2, :])

    plt.show()

    return

if __name__ == '__main__':

    main()
