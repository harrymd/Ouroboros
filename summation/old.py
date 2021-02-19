def test_associated_Legendre_poly():

    cosTheta = 0.5
    l_max = 3

    Plm_series, Plm_prime_series = associated_Legendre_func_series(l_max, l_max, cosTheta)
    Plm_series_no_cs, Plm_prime_series_no_cs = associated_Legendre_func_series_no_CS_phase(l_max, l_max, cosTheta)
    
    print('{:>3} {:>3} {:>10} {:>10}'.format('l', 'm', 'lpmv', 'lmpn'))
    for m in range(0, l_max + 1):

        for l in range(0, l_max + 1):

            Plm = associated_Legendre_func(m, l, cosTheta)

            print('{:>3d} {:>3d} {:>+10.5f} {:>+10.5f} {:>+10.5f}'.format(l, m, Plm, Plm_series[m, l], Plm_series_no_cs[m, l]))

    return
