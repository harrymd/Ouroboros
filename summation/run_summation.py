import argparse

import numpy as np
from scipy.special import  (lpmn as associated_Legendre_func_series,
                            lpmv as associated_Legendre_func)
import pandas

from Ouroboros.common import (  load_eigenfreq_Mineos, load_eigenfunc_Mineos, 
                                load_eigenfreq_Ouroboros, load_eigenfunc_Ouroboros,
                                read_channel_file,
                                read_Mineos_input_file,
                                read_Ouroboros_input_file, read_Ouroboros_summation_input_file)
from Ouroboros.misc.cmt_io import read_mineos_cmt

mode_type_to_int = {'R' : 0, 'S' : 1, 'T' : 2}

# Geometry. -------------------------------------------------------------------
def calculate_azimuth(lon_A, lat_A, lon_B, lat_B, io_in_degrees = True):
    '''
    Calculates initial bearing from point A for a great circle path to point B.
    Following convention in Dahlen and Tromp (1998, p. 364).
    http://mathforum.org/library/drmath/view/55417.html
    '''

    # Convert to radians if using degrees.
    if io_in_degrees:

        lon_A = np.deg2rad(lon_A)
        lat_A = np.deg2rad(lat_A)
        #
        lon_B = np.deg2rad(lon_B)
        lat_B = np.deg2rad(lat_B)
    
    # Calculate the azimuth, as an angle in radians measured from north.
    numerator = np.sin(lon_B - lon_A)*np.cos(lat_B)
    denominator = np.cos(lat_A)*np.sin(lat_B) - np.sin(lat_A)*np.cos(lat_B)*np.cos(lon_B - lon_A)
    azimuth = np.arctan2(numerator, denominator) # Radians from north.

    # Convert to a bearing measured from the south.
    azimuth = np.pi - azimuth

    # Convert back to degrees if using them.
    if io_in_degrees:

        azimuth = np.rad2deg(azimuth)

    return azimuth

def calculate_epicentral_distance(lon_A, lat_A, lon_B, lat_B, io_in_degrees = True):
    '''
    Dahlen and Tromp (1998) eq. 10.1.
    '''

    # Convert to radians if using degrees.
    if io_in_degrees:

        lon_A = np.deg2rad(lon_A)
        lat_A = np.deg2rad(lat_A)
        lon_B = np.deg2rad(lon_B)
        lat_B = np.deg2rad(lat_B)

    # Convert from latitude to colatitude (polar angle). 
    colat_A = (np.pi/2.0) - lat_A
    colat_B = (np.pi/2.0) - lat_B

    # Calculate the epicentral distance.
    cos_Theta = np.cos(colat_A)*np.cos(colat_B) + \
                np.sin(colat_A)*np.sin(colat_B)*np.cos(lon_A - lon_B)

    Theta = np.arccos(cos_Theta)

    # Convert back to degrees if using them.
    if io_in_degrees:

        Theta = np.rad2deg(Theta)

    return Theta, cos_Theta

#
def old_calculate_excitation_factor(l, cosTheta, Phi, cmt_info, eigfunc_source, verbose = True):
    '''
    Dahlen and Tromp (1998) eq. 10.53.
    '''
    
    if verbose:

        print('Calculating excitation factor.')

    # Associated Legendre function Plm.
    Pl0 = associated_Legendre_func(0, l, cosTheta)
    Pl1 = associated_Legendre_func(1, l, cosTheta)
    Pl2 = associated_Legendre_func(2, l, cosTheta)
    #
    Pl_list = [Pl0, Pl1, Pl2]
    
    # Azimuthal terms.
    cos0Phi = np.cos(0*Phi)
    cos1Phi = np.cos(1*Phi)
    cos2Phi = np.cos(2*Phi)
    cosPhi_list = [cos0Phi, cos1Phi, cos2Phi]
    #
    sin0Phi = np.sin(0*Phi)
    sin1Phi = np.sin(1*Phi)
    sin2Phi = np.sin(2*Phi)
    sinPhi_list = [sin0Phi, sin1Phi, sin2Phi]

    # Unpack dictionaries.
    r = cmt_info['r_centroid']
    Mrr = cmt_info['Mrr']
    Mtt = cmt_info['Mtt']
    Mpp = cmt_info['Mpp']
    Mrt = cmt_info['Mrt']
    Mrp = cmt_info['Mrp']
    Mtp = cmt_info['Mtp']
    #
    U = eigfunc_source['U']
    V = eigfunc_source['V']
    #
    dUdr = eigfunc_source['Up']
    dVdr = eigfunc_source['Vp']

    # Coefficients.
    # D&T eq. 10.54-10.59.
    k = np.sqrt(l*(l + 1.0))
    #
    A0 = Mrr*dUdr + ((Mtt + Mpp)*(U - 0.5*k*V)*(1.0/r))
    B0 = 0.0
    #
    C = dVdr - (V/r) + ((k*U)/r)
    A1 = Mrt*C/k
    B1 = Mrp*C/k
    #
    D = (0.5*V)/(k*r)
    A2 = D*(Mtt - Mpp)
    B2 = D*Mtp
    #
    A_list = [A0, A1, A2]
    B_list = [B0, B1, B2]
    
    #scale = 1.0E4
    #print(Mrr*dUdr*scale)
    #print(((Mtt + Mpp)*(U - 0.5*k*V)*(1.0/r))*scale)
    #print(A0*scale, B0*scale)
    #print(A1*scale, B2*scale)
    #print(A2*scale, B2*scale)
    ##print(C)

    #import sys
    #sys.exit()

    # Summation (D&T eq. 10.53).
    excitation = 0.0
    for m in range(3):
        
        term = Pl_list[m]*(A_list[m]*cosPhi_list[m] + B_list[m]*sinPhi_list[m])
        excitation = excitation + term 

    return excitation

def calculate_source_coefficients_spheroidal(l, cmt_info, eigfunc_source):
    '''
    Dahlen and Tromp (1998) eq. 10.53.
    '''
    
    ## Associated Legendre function Plm.
    ##Plm_list = Plm_series[:, l]
    #Pl0, Pl1, Pl2 = Plm_list
    #
    ## Azimuthal terms.
    #cos0Phi = np.cos(0*Phi)
    #cos1Phi = np.cos(1*Phi)
    #cos2Phi = np.cos(2*Phi)
    #cosPhi_list = [cos0Phi, cos1Phi, cos2Phi]
    ##
    #sin0Phi = np.sin(0*Phi)
    #sin1Phi = np.sin(1*Phi)
    #sin2Phi = np.sin(2*Phi)
    #sinPhi_list = [sin0Phi, sin1Phi, sin2Phi]

    # Unpack dictionaries.
    r = cmt_info['r_centroid']
    Mrr = cmt_info['Mrr']
    Mtt = cmt_info['Mtt']
    Mpp = cmt_info['Mpp']
    Mrt = cmt_info['Mrt']
    Mrp = cmt_info['Mrp']
    Mtp = cmt_info['Mtp']
    #
    U = eigfunc_source['U']
    V = eigfunc_source['V']
    #
    dUdr = eigfunc_source['Up']
    dVdr = eigfunc_source['Vp']

    # Coefficients.
    # D&T eq. 10.54-10.59.
    k = np.sqrt(l*(l + 1.0))
    #
    A0 = Mrr*dUdr + ((Mtt + Mpp)*(U - 0.5*k*V)*(1.0/r))
    B0 = 0.0
    #
    C = dVdr - (V/r) + ((k*U)/r)
    A1 = Mrt*C/k
    B1 = Mrp*C/k
    #
    D = V/(k*r)
    A2 = 0.5*D*(Mtt - Mpp)
    B2 = D*Mtp
    #
    #A_list = [A0, A1, A2]
    #B_list = [B0, B1, B2]
    #
    ##scale = 1.0E4
    ##print(Mrr*dUdr*scale)
    ##print(((Mtt + Mpp)*(U - 0.5*k*V)*(1.0/r))*scale)
    ##print(A0*scale, B0*scale)
    ##print(A1*scale, B2*scale)
    ##print(A2*scale, B2*scale)
    ###print(C)

    ##import sys
    ##sys.exit()

    ## Summation (D&T eq. 10.53).
    #excitation = 0.0
    #for m in range(3):
    #    
    #    term = Pl_list[m]*(A_list[m]*cosPhi_list[m] + B_list[m]*sinPhi_list[m])
    #    excitation = excitation + term 

    #return excitation

    # Store in dictionary.
    src_coeffs = {'A0' : A0, 'A1' : A1, 'A2' : A2, 'B0' : B0, 'B1' : B1, 'B2' : B2}

    return src_coeffs

def calculate_coeffs_spheroidal(source_coeffs, eigfunc_receiver, l, sinTheta, Phi, Plm_series, Plm_prime_series, sin_cos_Phi_list):
    '''
    Reference

    [1] Ouroboros/summation/notes.pdf
    '''

    # Unpack source coefficients.
    A0 = source_coeffs['A0']
    A1 = source_coeffs['A1']
    A2 = source_coeffs['A2']
    #
    B0 = source_coeffs['B0']
    B1 = source_coeffs['B1']
    B2 = source_coeffs['B2']

    # Unpack associated Legendre polynomial.
    Pl0, Pl1, Pl2 = Plm_series
    Pl0_p, Pl1_p, Pl2_p = Plm_prime_series

    # Unpack receiver eigenfunction.
    Ur = eigfunc_receiver['U']
    Vr = eigfunc_receiver['V']

    # Unpack functions of azimuth.
    cosPhi, sinPhi, cos2Phi, sin2Phi = sin_cos_Phi_list

    cosPhi = np.cos(Phi)
    sinPhi = np.sin(Phi)
    cos2Phi = np.cos(2.0*Phi)
    sin2Phi = np.sin(2.0*Phi)

    # Calculate k.
    k = np.sqrt(l*(l + 1.0))

    # Calculate the coefficients.
    A = dict()

    # Radial component, [1] equation (1).
    A['r'] = Ur*(       Pl0*(A0*1.0     + B0*0.0)
                    +   Pl1*(A1*cosPhi  + B1*sinPhi)
                    +   Pl2*(A2*cos2Phi + B2*sin2Phi))

    # Theta component, [1] equation (2).
    A['Theta'] = (-1.0/k)*Vr*sinTheta*(
                        Pl0_p*(A0*1.0     + B0*0.0)
                    +   Pl1_p*(A1*cosPhi  + B1*sinPhi)
                    +   Pl2_p*(A2*cos2Phi + B2*sin2Phi))

    # Phi component, [1] equation (3).
    A['Phi'] = (1.0/(sinTheta*k))*Vr*(
                            Pl1*(B1*sinPhi  - A1*cosPhi)
                    +   2.0*Pl2*(B2*sin2Phi - A2*cos2Phi))
                            

    return A

#
def associated_Legendre_func_series_no_CS_phase(m_max, l_max, cosTheta):
    '''
    Get the associated Legendre function Plm from m = 0 to m_max and
    l = 0 to l_max.
    Note that Condon-Shortley phase factor of (-1)**m is included in the
    Scipy implementation [1] but not in Dahlen and Tromp (see eq. B.71), so we
    remove this factor.

    [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.lpmv.html
    '''

    Plm_series, Plm_prime_series = associated_Legendre_func_series(m_max, l_max, cosTheta)

    for m in range(0, m_max + 1):
        
        if (m % 2) != 0:

            Plm_series[m, :] = -1.0*Plm_series[m, :]
            Plm_prime_series[m, :] = -1.0*Plm_prime_series[m, :]

    return Plm_series, Plm_prime_series

# -----------------------------------------------------------------------------
def load_mode_info(run_info, summation_info, use_mineos = False):

    # Load mode information.
    mode_info = dict()
    for mode_type in summation_info['mode_types']:

        # For toroidal modes, only use modes contained in the outer solid shell.
        if mode_type == 'T':

            i_toroidal = 0

        else:
            
            i_toroidal = None

        # Load file.
        if use_mineos:

            n, l, f, q = load_eigenfreq_Mineos(run_info, mode_type, return_q = True)
        
        else:

            n, l, f = load_eigenfreq_Ouroboros(
                            run_info, mode_type, i_toroidal = i_toroidal)

        # Apply frequency filter.
        i_freq = np.where(  (f > summation_info['f_lims'][0]) &
                            (f < summation_info['f_lims'][1]))[0]
        n = n[i_freq]
        l = l[i_freq]
        f = f[i_freq]
        q = q[i_freq]
        
        # Store in dictionary.
        mode_info[mode_type] = dict()
        mode_info[mode_type]['n'] = n
        mode_info[mode_type]['l'] = l 
        mode_info[mode_type]['f'] = f 
        mode_info[mode_type]['q'] = q 

    return mode_info

def load_eigenfunc(run_info, mode_type, n, l, z_source, z_receiver = 0.0, use_mineos = False):

    # Load eigenfunction information for this mode.
    if mode_type in ['R', 'S']:

        # Load spheroidal mode.
        if mode_type == 'S':

            if use_mineos:

                # Load Mineos eigenfunction.
                r, U, Up, V, Vp, P, Pp = \
                    load_eigenfunc_Mineos(run_info, mode_type, n, l)

                # Convert to km.
                r = r*1.0E-3

            else:

                print('Loading eigenfunction for spheroidal modes not\
                implemented yet for Ouroboros')
                raise NotImplementedError

        # Load toroidal mode.
        elif mode_type == 'R':

            if use_mineos:

                # Load Mineos eigenfunction.
                r, U, Up = \
                    load_eigenfunc_Mineos(run_info, mode_type, n, l)

                # Convert to km.
                r = r*1.0E-3

            else:

                print('Loading eigenfunction for radial modes not implemented\
                yet for Ouroboros.')
                raise NotImplementedError

            # No tangential component for radial modes.
            V = np.zeros(U.shape)
            Vp = np.zeros(Up.shape)

        # Store required values in dictionary.
        eigenfunc_dict = {
            'r'  : r,
            'U'  : U,
            'V'  : V,
            'Up' : Up,
            'Vp' : Vp}

    else:

        print('Toroidal modes not implemented yet.')
        raise NotImplementedError

    # Convert r to depth.
    r_planet = eigenfunc_dict['r'][0]
    z = r_planet - eigenfunc_dict['r']
    assert z[-1] > z[0], 'Depth must be increasing for np.interp'

    # Find the eigenfunctions and gradients at the depth of the source
    # and receiver.
    eigfunc_source = dict()
    eigfunc_receiver = dict()
    if mode_type in ['R', 'S']:

        keys = ['U', 'V', 'Up', 'Vp']

    else:

        raise NotImplementedError

    for key in keys:

        eigfunc_source[key] = np.interp(  z_source,   z, eigenfunc_dict[key])
        eigfunc_receiver[key] = np.interp(z_receiver, z, eigenfunc_dict[key])

    return eigfunc_source, eigfunc_receiver, r_planet

def main():

    # Parse input arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("path_mode_input", help = "File path (relative or absolute) to Ouroboros mode input file.")
    parser.add_argument("path_summation_input", help = "File path (relative or absolute) to Ouroboros summation input file.")
    parser.add_argument("--use_mineos", action = 'store_true', help = "Use Mineos path_mode_input file, eigenfrequencies and eigenfunctions. Note 1: The path_summation_input file should still be in Ouroboros format. Note 2: This option is for testing. For access to the built-in Mineos synthetics, see Ouroboros/mineos/summation.py.")
    input_args = parser.parse_args()
    path_mode_input = input_args.path_mode_input
    path_summation_input = input_args.path_summation_input
    use_mineos = input_args.use_mineos

    # Read the mode input file.
    if use_mineos:
        
        # Read Mineos input file.
        run_info = read_Mineos_input_file(path_mode_input)

    else:
        
        # Read Ouroboros input files.
        run_info = read_Ouroboros_input_file(path_mode_input)

    # Read the summation input file.
    summation_info = read_Ouroboros_summation_input_file(path_summation_input)
    assert all([(mode_type in run_info['mode_types']) for mode_type in summation_info['mode_types']]), \
    'The summation input file specifies mode types which are not found in the mode input file.'

    # Load CMT information.
    cmt = read_mineos_cmt(summation_info['path_cmt'])
    print('Event: {:>10}, lon. : {:>+8.2f}, lat.  {:>+8.2f}, depth {:>9.2f} km'.format(cmt['ev_id'],
            cmt['lon_centroid'], cmt['lat_centroid'], cmt['depth_centroid']))

    # Load channel information.
    channel_dict = read_channel_file(summation_info['path_channels'])
    
    # Load mode information.
    mode_info = load_mode_info(run_info, summation_info, use_mineos = use_mineos)
    num_modes_dict = dict()
    for mode_type in summation_info['mode_types']:

        num_modes_dict[mode_type] = len(mode_info[mode_type]['f'])

    num_modes_total = sum([num_modes_dict[mode_type] for mode_type in num_modes_dict])

    # Calculate maximum l-value. 
    l_max = np.max([np.max(mode_info[mode_type]['l']) for mode_type in mode_info])
    print('Maximum l-value: {:>5d}'.format(l_max))

    # Create output arrays.
    num_station = len(channel_dict)
    type_list    = np.zeros((num_station, num_modes_total), dtype = np.int)
    n_list       = np.zeros((num_station, num_modes_total), dtype = np.int)
    l_list       = np.zeros((num_station, num_modes_total), dtype = np.int)
    f_list       = np.zeros((num_station, num_modes_total), dtype = np.float)
    q_list       = np.zeros((num_station, num_modes_total), dtype = np.float)
    A_r_list     = np.zeros((num_station, num_modes_total), dtype = np.float) 
    A_Theta_list = np.zeros((num_station, num_modes_total), dtype = np.float)
    A_Phi_list   = np.zeros((num_station, num_modes_total), dtype = np.float)

    # Do summation.
    for j, station in enumerate(channel_dict):
        
        print('Station: {:>8}, lon. : {:>+8.2f},  lat. {:>+8.2f}'.format(station,
                channel_dict[station]['coords']['longitude'],
                channel_dict[station]['coords']['latitude']))
        
        # Calculate epicentral distance and azimuth.
        Theta_deg, cosTheta = calculate_epicentral_distance(
                            cmt['lon_centroid'], cmt['lat_centroid'],
                            channel_dict[station]['coords']['longitude'],
                            channel_dict[station]['coords']['latitude'],
                            io_in_degrees = True)
        Phi_deg = calculate_azimuth(cmt['lon_centroid'], cmt['lat_centroid'],
                                    channel_dict[station]['coords']['longitude'],
                                    channel_dict[station]['coords']['latitude'],
                                    io_in_degrees = True)
        # Convert to radians.
        Theta = np.deg2rad(Theta_deg)
        Phi = np.deg2rad(Phi_deg)

        # Calculate geometric quantities.
        sinTheta = np.sin(Theta)
        #
        cosPhi = np.cos(Phi)
        sinPhi = np.sin(Phi)
        cos2Phi = np.cos(2.0*Phi)
        sin2Phi = np.sin(2.0*Phi)
        sin_cos_Phi_list = [cosPhi, sinPhi, cos2Phi, sin2Phi]

        print('              Epi. dist.: {:>8.2f}, azim. {:>+8.2f}'.format(Theta_deg, Phi_deg))

        # Pre-calculate the associated Legendre functions.
        Plm_series, Plm_prime_series = \
                associated_Legendre_func_series_no_CS_phase(
                        2, l_max, cosTheta)

        for mode_type in summation_info['mode_types']:

            num_modes = num_modes_dict[mode_type]
            print('Mode type: {:>3}, mode count: {:>5d}'.format(mode_type, num_modes))

            for i in range(num_modes):
                
                # Unpack.
                n = mode_info[mode_type]['n'][i]
                l = mode_info[mode_type]['l'][i]
                f = mode_info[mode_type]['f'][i]
                q = mode_info[mode_type]['f'][i]

                # Load the eigenfunction information interpolated
                # at the source and receiver locations.
                # Also get the planet radius.
                eigfunc_source, eigfunc_receiver, r_planet = \
                    load_eigenfunc(run_info, mode_type,
                        n,
                        l,
                        cmt['depth_centroid'],
                        z_receiver = 0.0,
                        use_mineos = use_mineos)

                if i == 0:
                
                    # Calculate radial coordinate of event.
                    cmt['r_centroid'] = r_planet - cmt['depth_centroid']

                # Calculate the coefficients.
                if mode_type in ['R', 'S']:

                    # Excitation coefficients determined by source location.
                    source_coeffs = \
                        calculate_source_coefficients_spheroidal(
                            l, cmt, eigfunc_source)

                    # Overall coefficients including receiver location.
                    coeffs = calculate_coeffs_spheroidal(
                                source_coeffs, eigfunc_receiver, l,
                                sinTheta, Phi, Plm_series[:, l],
                                Plm_prime_series[:, l], sin_cos_Phi_list)

                else:

                    raise NotImplementedError

                # Store output.
                type_list[j, i]     = mode_type_to_int[mode_type]
                n_list[j, i]        = n
                l_list[j, i]        = l 
                f_list[j, i]        = f
                q_list[j, i]        = q
                A_r_list[j, i]       = coeffs['r']
                A_Theta_list[j, i]  = coeffs['Theta']
                A_Phi_list[j, i]    = coeffs['Phi']

    # Store the data in a data frame.
    print(type_list.shape)
    #index = pandas.MultiIndex.from_product([type_list, n_list, l_list, f_list, q_list, A_r_list, A_Theta_list, A_Phi_list])
    data_list = [type_list, n_list, l_list, f_list, q_list, A_r_list, A_Theta_list, A_Phi_list]
    name_list = ['type', 'n', 'l', 'f', 'q', 'A_r', 'A_Theta', 'A_Phi']
    num_data_types = len(name_list)

    data_frame_list = []
    #for data, name in zip(data_list, name_list):

    #    data_frame_list.append(pandas.DataFrame({name : np.squeeze(data)}))

    for i in range(num_station):
        
        data_dict = {x : y[i, :] for x, y in zip(name_list, data_list)}
        data_frame_list.append(pandas.DataFrame(data_dict))
    
    data_frame = pandas.concat(data_frame_list, keys = list(range(num_station)))

    # Save to pickle.
    path_data_frame = 'coeffs.pkl'
    print('Saving coefficients to {:}'.format(path_data_frame))
    data_frame.to_pickle(path_data_frame)

    print(data_frame)

        #for channel in channel_dict[station]['channels']:

        #    print('Channel: {:>5}'.format(channel))

        #    for mode_type in summation_info['mode_types']:
    
    #print(run_info)
    #print(summation_info)
    #summation_wrapper(path_mode_input, path_summation_input)

    return

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

if __name__ == '__main__':

    main()
    #test_associated_Legendre_poly()
