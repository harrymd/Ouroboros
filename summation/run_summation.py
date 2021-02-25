import argparse
import datetime
import os

import numpy as np
from obspy.core.stream import Stream
from obspy.core.trace import Trace
from scipy.special import  lpmn as associated_Legendre_func_series
#from scipy.special import lpmv as associated_Legendre_func
import pandas

#from Ouroboros.constants import G
from Ouroboros.constants import G_mineos as G
from Ouroboros.common import (  get_Mineos_out_dirs, get_Mineos_summation_out_dirs, 
                                load_eigenfreq_Mineos, load_eigenfunc_Mineos, 
                                load_eigenfreq_Ouroboros, load_eigenfunc_Ouroboros,
                                load_model,
                                mkdir_if_not_exist,
                                read_channel_file,
                                read_Mineos_input_file,
                                read_Ouroboros_input_file, read_Ouroboros_summation_input_file)
from Ouroboros.misc.cmt_io import read_mineos_cmt

mode_type_to_int = {'R' : 0, 'S' : 1, 'T' : 2}
mode_int_to_type = {0 : 'R', 1 : 'S', 2 : 'T'}

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

def calculate_epi_dist_azi_mineos(lon_0, lat_0, lon_1, lat_1):
    '''
    Copied from mineos/green.f:

          data pi,rad,tstart/3.14159265359,57.29578,0./

    .
    .
    .

      th = 90.0-atan(0.99329534*tan(slat/rad))*rad                              
      ph=slon 

    .
    .
    .

      t0=th/rad                                                                 
      p0=ph/rad                                                                 
      c0=cos(t0)                                                                
      s0=sin(t0)    

    .
    .
    .

    c.....convert station geographic latitude to geocentric                         
          t1 = (90.0-atan(0.99329534*tan(lat_site(ns(1))/rad))*rad)/rad             
          p1 = lon_site(ns(1))                                                      
          if(p1.lt.0.0) p1 = 360.0+p1                                               
          p1 = p1/rad                                                               
                                                                                    
          if(icomp.eq.2) goto 500                                                   
          len=0                                                                     
          if(icomp.eq.1) goto 35                                                    
    c                                                                               
    c....do some trigonometry                                                       
    c      epicentral distance: co, si                                              
    c      azimuth of source:   caz,saz                                             
    c                                                                               
          c1=cos(t1)                                                                
          s1=sin(t1)                                                                
          dp=p1-p0                                                                  
          co=c0*c1+s0*s1*cos(dp)                                                    
          si=dsqrt(1.d0-co*co)                                                      
                                                                                    
          del = datan2(si,co)                                                       
          del = del/pi*180.                                                         
          write(*,'(a,f10.3)')' green: Epicentral Distance : ',del                  
                                                                                    
          sz=s1*sin(dp)/si                                                          
          cz=(c0*co-c1)/(s0*si)                                                     
          saz=-s0*sin(dp)/si                                                        
          caz=(c0-co*c1)/(si*s1)                                                    
                                                                                    
          azim = datan2(saz,caz)                                                    
          azim = azim/pi*180.                                                       
          write(*,'(a,f10.3)')' green: Azimuth of Source   : ',azim 
    '''
    
    # Constant for converting from degrees to radians.
    pi = 3.14159265359
    rad = 57.29578

    # Correct latitude for Earth flattening and convert to radians.
    t0 = (90.0 - np.arctan(0.99329534*np.tan(lat_0/rad))*rad)/rad             
    t1 = (90.0 - np.arctan(0.99329534*np.tan(lat_1/rad))*rad)/rad             

    # Wrap longitude and convert to radians. 
    p0 = lon_0                                                      
    if p0 < 0.0:
        p0 = p0 + 360.0
    p0 = p0/rad                                                               
    #
    p1 = lon_1                                                      
    if p1 < 0.0:
        p1 = p1 + 360.0
    p1 = p1/rad                                                               
                                                                                
    # Get sines and cosines. 
    c0 = np.cos(t0)
    s0 = np.sin(t0)
    #
    c1 = np.cos(t1)                                                                
    s1 = np.sin(t1)                                                                
    #
    dp = p1 - p0                                                                  
    co = c0*c1 + s0*s1*np.cos(dp)                                                    
    si = np.sqrt(1.0 - co*co)                                                      
                                                                                
    # Get epicentral distance.
    delta = np.arctan2(si,co)                                                       
    delta = delta/pi*180.0                                                         
                                                                                
    # Get azimuth.
    sz = s1*np.sin(dp)/si                                                          
    cz = (c0*co - c1)/(s0*si)                                                     
    saz = -s0*np.sin(dp)/si                                                        
    caz = (c0 - co*c1)/(si*s1)                                                    
                                                                              
    azim = np.arctan2(saz, caz)                                                    
    azim = azim/pi*180.0                                                       

    return delta, azim

def polar_coords_to_unit_vector(theta, phi):
    '''
    See diagram in [1] for definitions of angles.
    theta   Azimuth (radians).
    phi     Polar angle (radian). 

    Reference:
    [1] https://mathworld.wolfram.com/SphericalCoordinates.html
    '''

    # [1], equation 4-6 with r set to 1.
    x = np.cos(theta)*np.sin(phi)
    y = np.sin(theta)*np.sin(phi)
    z = np.cos(phi)

    return x, y, z

#
def get_surface_gravity(run_info):
    
    # Load the planetary model.
    model = load_model(run_info['path_model'])
    
    # Integrate spherical shells to get mass.
    mass_element = 4.0*np.pi*model['rho']*(model['r']**2.0)
    mass = np.trapz(mass_element, x = model['r'])
    
    # Newton's formula for gravity.
    r_srf = model['r'][-1]
    g_srf = G*mass/(r_srf**2.0)

    return g_srf

#
def moment_sinc(omega, t_half):

    x = omega*t_half
    sinc = np.sin(x)/x

    return sinc

def moment_triangle(omega, t_half):
    '''
    Fourier transform of triangular pulse with half-width t_half and unit
    area.
    '''

    #x = omega*t_half

    #f = (2.0/(x**2.0))*(1.0 - np.cos(x))

    x = 0.5*0.5*t_half*omega
    f = (np.sin(x)/x)**2.0

    return f 

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

def scale_moment_tensor(cmt_info, omega, pulse_type):

    # Dahlen and Tromp (1998), eq. 5.92.
    comp_list = ['rr', 'tt', 'pp', 'rt', 'rp', 'tp']
    M = {comp : cmt_info['M{:}'.format(comp)] for comp in comp_list}
    
    # Multiply by the frequency-domain value of the unit source-time function
    # as described in Dahlen and Tromp (1998), p. 377, including factor
    # of sqrt(2).
    # Also divide by 10^7 to convert from dyn-cm to N-m.
    if pulse_type == 'rectangle':

        m_omega = moment_sinc(omega, cmt_info['half_duration'])

    elif pulse_type == 'triangle':

        m_omega = moment_triangle(omega, cmt_info['half_duration'])

    # Not sure if factor of sqrt(2) belongs here.
    #M_scaling = np.sqrt(2.0)*cmt_info['scale_factor']*m_omega*1.0E-7
    M_scaling = cmt_info['scale_factor']*m_omega*1.0E-7

    for comp in comp_list:

        M[comp] = M[comp]*M_scaling

    return M

def calculate_source_coefficients_spheroidal(l, omega, cmt_info, eigfunc_source, pulse_type):
    '''
    Dahlen and Tromp (1998) eq. 10.53.
    '''

    # Get radial coordinate of source.
    r = cmt_info['r_centroid']*1.0E3 # km to m
    
    # Apply appropriate frequency scaling to moment tensor.
    M = scale_moment_tensor(cmt_info, omega, pulse_type)
    
    ## Get unit moment tensor (Dahlen and Tromp, 1998, p. 167).
    ## Dahlen and Tromp (1998) eq. 5.91
    #M0 = (1.0/np.sqrt(2.0))*np.sqrt(Mrr**2.0 + Mtt**2.0 + Mpp**2.0
    #                                + 2.0*(Mrt**2.0) + 2.0*(Mrp**2.0) + 2.0*(Mtp**2.0))

    # Unpack eigenfunction information.
    U = eigfunc_source['U']
    V = eigfunc_source['V']
    #
    dUdr = eigfunc_source['Up']
    dVdr = eigfunc_source['Vp']
    
    # Coefficients.
    # D&T eq. 10.54-10.59.
    k = np.sqrt(l*(l + 1.0))
    #
    A0 = M['rr']*dUdr + ((M['tt'] + M['pp'])*(U - 0.5*k*V)*(1.0/r))
    B0 = 0.0
    #
    C = dVdr - (V/r) + ((k*U)/r)
    A1 = M['rt']*C/k
    B1 = M['rp']*C/k
    #
    D = V/(k*r)
    A2 = 0.5*D*(M['tt'] - M['pp'])
    B2 = D*M['tp']

    # Store in dictionary.
    src_coeffs = {'A0' : A0, 'A1' : A1, 'A2' : A2, 'B0' : B0, 'B1' : B1, 'B2' : B2}

    return src_coeffs

def calculate_coeffs_spheroidal(source_coeffs, eigfunc_receiver, l, sinTheta, Plm_series, Plm_prime_series, sin_cos_Phi_list):
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

    #print(Pl0, Pl1, Pl2)
    #print('\n')
    #print(Pl2)

    # Unpack receiver eigenfunction.
    Ur = eigfunc_receiver['U']
    Vr = eigfunc_receiver['V']

    # Unpack functions of azimuth.
    cosPhi, sinPhi, cos2Phi, sin2Phi = sin_cos_Phi_list

    # Calculate k and L.
    k = np.sqrt(l*(l + 1.0))
    L = (2.0*l + 1.0)/(4.0*np.pi)

    # Calculate the coefficients.
    A = dict()

    # Radial component, [1] equation (1).
    A['r'] = L*Ur*(     Pl0*(A0*1.0     + B0*0.0)
                    +   Pl1*(A1*cosPhi  + B1*sinPhi)
                    +   Pl2*(A2*cos2Phi + B2*sin2Phi))

    # Theta component, [1] equation (2).
    A['Theta'] = L*(-1.0/k)*Vr*sinTheta*(
                        Pl0_p*(A0*1.0     + B0*0.0)
                    +   Pl1_p*(A1*cosPhi  + B1*sinPhi)
                    +   Pl2_p*(A2*cos2Phi + B2*sin2Phi))

    # Phi component, [1] equation (3).
    A['Phi'] = L*(1.0/(sinTheta*k))*Vr*(
                            Pl1*(B1*sinPhi  - A1*cosPhi)
                    +   2.0*Pl2*(B2*sin2Phi - A2*cos2Phi))
                            

    return A

def calculate_source_coefficients_radial(omega, cmt_info, eigfunc_source, pulse_type):
    '''
    Dahlen and Tromp (1998) eq. 10.53.
    '''

    # Get radial coordinate of source.
    r = cmt_info['r_centroid']*1.0E3 # km to m
    
    # Apply appropriate frequency scaling to moment tensor.
    M = scale_moment_tensor(cmt_info, omega, pulse_type)
    
    # Unpack eigenfunction information.
    U = eigfunc_source['U']
    dUdr = eigfunc_source['Up']
    
    # Coefficients.
    # D&T eq. 10.54-10.59.
    #
    A0 = M['rr']*dUdr + (M['tt'] + M['pp'])*U/r

    # Store in dictionary.
    src_coeffs = {'A0' : A0}

    return src_coeffs

def calculate_coeffs_radial(source_coeffs, eigfunc_receiver):
    '''
    Reference

    [1] Ouroboros/summation/notes.pdf
    '''

    # Unpack source coefficients.
    A0 = source_coeffs['A0']

    # Unpack receiver eigenfunction.
    Ur = eigfunc_receiver['U']

    # Calculate the coefficients.
    A = dict()

    # Radial component, [1] equation (1).
    A['r'] = Ur*A0/(4.0*np.pi)

    # Radial modes have no tranverse motion.
    A['Theta'] = np.zeros(A['r'].shape)
    A['Phi']   = np.zeros(A['r'].shape)

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

    #for m in range(0, m_max + 1):
    #    
    #    if (m % 2) != 0:

    #        Plm_series[:, m] = -1.0*Plm_series[:, m]
    #        Plm_prime_series[:, m] = -1.0*Plm_prime_series[:, m]

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

            n, l, f, Q = load_eigenfreq_Mineos(run_info, mode_type, return_Q = True)
        
        else:

            n, l, f = load_eigenfreq_Ouroboros(
                            run_info, mode_type, i_toroidal = i_toroidal)

        # Apply frequency filter.
        i_freq = np.where(  (f > summation_info['f_lims'][0]) &
                            (f < summation_info['f_lims'][1]))[0]
        n = n[i_freq]
        l = l[i_freq]
        f = f[i_freq]
        Q = Q[i_freq]
        
        # Store in dictionary.
        mode_info[mode_type] = dict()
        mode_info[mode_type]['n'] = n
        mode_info[mode_type]['l'] = l 
        mode_info[mode_type]['f'] = f 
        mode_info[mode_type]['Q'] = Q 

    return mode_info

def load_eigenfunc(run_info, mode_type, n, l, f_rad_per_s, z_source, z_receiver = 0.0, use_mineos = False, response_correction_params = None):

    norm_args = {'norm_func' : 'DT', 'units' : 'SI', 'omega' : f_rad_per_s}

    # Load eigenfunction information for this mode.
    if mode_type == 'S':
        
        if use_mineos:

            # Load Mineos eigenfunction.
            r, U, Up, V, Vp, P, Pp = \
                load_eigenfunc_Mineos(run_info, mode_type, n, l,
                    **norm_args)

            # Store required values in dictionary.
            eigenfunc_dict = {
                'r'  : r,
                'U'  : U,
                'V'  : V,
                'Up' : Up,
                'Vp' : Vp} 

        else:

            print('Loading eigenfunction for spheroidal modes not\
            implemented yet for Ouroboros')
            raise NotImplementedError

    # Load toroidal mode.
    elif mode_type == 'R':

        if use_mineos:

            # Load Mineos eigenfunction.
            r, U, Up = \
                load_eigenfunc_Mineos(run_info, mode_type, n, l,
                        **norm_args)

            # Store in dictionary.
            eigenfunc_dict = {'r' : r, 'U' : U, 'Up' : Up}

        else:

            print('Loading eigenfunction for radial modes not implemented\
            yet for Ouroboros.')
            raise NotImplementedError

    else:
        
        print('Mode summation only implemented for R and S modes.')
        raise NotImplementedError

    if response_correction_params is not None:

        if mode_type == 'S':

            eigenfunc_dict['P'] = P

        elif mode_type == 'R':
            
            # Radial modes do have a perturbation to the potential
            # but only internally, so here we just set P = 0.
            eigenfunc_dict['P'] = np.zeros(U.shape)

        else:

            raise ValueError('Response correction can only be applied to R and S modes.')

    # Convert r to depth.
    r_planet = eigenfunc_dict['r'][0]
    z = r_planet - eigenfunc_dict['r']
    assert z[-1] > z[0], 'Depth must be increasing for np.interp'

    # Find the eigenfunctions and gradients at the depth of the source
    # and receiver.
    eigfunc_source = dict()
    eigfunc_receiver = dict()
    if mode_type == 'S':

        if response_correction_params is not None:
        
            keys = ['U', 'V', 'Up', 'Vp', 'P']

        else:

            keys = ['U', 'V', 'Up', 'Vp']

    elif mode_type == 'R':

        if response_correction_params is not None:

            keys = ['U', 'Up', 'P']

        else:

            keys = ['U', 'Up']

    else:
        
        print('Not implented yet for T modes.')
        raise NotImplementedError

    for key in keys:

        eigfunc_source[key] = np.interp(  z_source,   z, eigenfunc_dict[key])
        eigfunc_receiver[key] = np.interp(z_receiver, z, eigenfunc_dict[key])

    # Apply seismometer response correction (if requested).
    if response_correction_params is not None:
        
        assert mode_type in ['R', 'S']
        # Unpack response correction parameters.
        g = response_correction_params['g']
        #
        U = eigfunc_receiver['U']
        P = eigfunc_receiver['P']

        U_free, U_pot, V_tilt, V_pot = seismometer_response_correction(
                                        l, f_rad_per_s, r_planet, g,
                                        U, P)

        eigfunc_receiver['U'] = U + U_free + U_pot
        if mode_type == 'S':

            V = eigfunc_receiver['V']
            eigfunc_receiver['V'] = V + V_tilt + V_pot
    
    return eigfunc_source, eigfunc_receiver, r_planet

def seismometer_response_correction(l, f_rad_per_s, r_planet, g, U, P):

    # Dahlen and Tromp (1998) eq. 10.70-10.72.
    k = np.sqrt(l*(l + 1))
    c = r_planet*(f_rad_per_s**2.0)
    U_free = (2.0*g*U)/c
    U_pot = (l + 1.0)*P/c
    V_tilt = -k*g*U/c
    V_pot = -k*P/c

    #print('{:>5d} {:>10.3f} {:>10.3e} {:10.3e} {:>10.3e} {:>10.3e}'.format(l, f_rad_per_s/(2.0*np.pi*1.0E-3), U_free, U_pot, V_tilt, V_pot))

    return U_free, U_pot, V_tilt, V_pot

def get_coeffs_wrapper(run_info, summation_info, use_mineos = False, overwrite = False, response_correction_params = None):

    # Get name of output file.
    name_coeffs_data_frame = 'coeffs.pkl'
    name_stations_data_frame = 'stations.pkl'
    name_modes_data_frame = 'modes.pkl'
    #
    path_coeffs_data_frame = os.path.join(summation_info['dir_output'], name_coeffs_data_frame)
    path_stations_data_frame = os.path.join(summation_info['dir_output'], name_stations_data_frame)
    path_modes_data_frame = os.path.join(summation_info['dir_output'], name_modes_data_frame)
    #
    paths = [path_coeffs_data_frame, path_stations_data_frame, path_modes_data_frame]
    #
    paths_exist = [os.path.exists(path) for path in paths]

    if all(paths_exist)and (not overwrite):

        print('Coefficient output files already exist.')

        print('Loading {:}'.format(path_coeffs_data_frame))
        coeffs = pandas.read_pickle(path_coeffs_data_frame)

        print('Loading {:}'.format(path_stations_data_frame))
        stations = pandas.read_pickle(path_stations_data_frame)

        print('Loading {:}'.format(path_modes_data_frame))
        modes = pandas.read_pickle(path_modes_data_frame)

        return coeffs, stations, modes

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
    type_list    = np.zeros(num_modes_total, dtype = np.int)
    n_list       = np.zeros(num_modes_total, dtype = np.int)
    l_list       = np.zeros(num_modes_total, dtype = np.int)
    f_list       = np.zeros(num_modes_total, dtype = np.float)
    Q_list       = np.zeros(num_modes_total, dtype = np.float)
    A_r_list     = np.zeros((num_station, num_modes_total), dtype = np.float) 
    A_Theta_list = np.zeros((num_station, num_modes_total), dtype = np.float)
    A_Phi_list   = np.zeros((num_station, num_modes_total), dtype = np.float)
    Theta_list   = np.zeros(num_station, dtype = np.float)
    Phi_list     = np.zeros(num_station, dtype = np.float)

    # Do summation.
    epi_dist_azi_method = 'mineos'
    #epi_dist_azi_method = 'spherical'
    station_list = []
    for j, station in enumerate(channel_dict):

        station_list.append(station)
        
        print('Station: {:>8}, lon. : {:>+8.2f},  lat. {:>+8.2f}'.format(station,
                channel_dict[station]['coords']['longitude'],
                channel_dict[station]['coords']['latitude']))

        # Use Mineos approach to calculating epicentral distance and azimuth.
        # Includes a correction to latitude for Earth's flattening.
        if epi_dist_azi_method == 'mineos':

            #Theta_deg, Azi_deg = calculate_epi_dist_azi_mineos(
            #                cmt['lon_centroid'], cmt['lat_centroid'],
            #                channel_dict[station]['coords']['longitude'],
            #                channel_dict[station]['coords']['latitude'])
            Theta_deg, Azi_deg = calculate_epi_dist_azi_mineos(
                            channel_dict[station]['coords']['longitude'],
                            channel_dict[station]['coords']['latitude'],
                            cmt['lon_centroid'], cmt['lat_centroid'])

            Phi_deg = 180.0 - Azi_deg
            #Phi_deg = calculate_azimuth(cmt['lon_centroid'], cmt['lat_centroid'],
            #                            channel_dict[station]['coords']['longitude'],
            #                            channel_dict[station]['coords']['latitude'],
            #                            io_in_degrees = True)

            cosTheta = np.cos(np.deg2rad(Theta_deg))
        
        # Calculate epicentral distance and azimuth using spherical formulae.
        elif epi_dist_azi_method == 'spherical':
        
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

        else:

            raise ValueError

        #Theta_deg = 83.268
        #cosTheta = np.cos(np.deg2rad(Theta_deg))
        # Convert to radians.
        Theta = np.deg2rad(Theta_deg)
        Phi = np.deg2rad(Phi_deg)
        
        Theta_list[j] = Theta
        Phi_list[j] = Phi

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
        if l_max < 2:
            l_max_legendre = 2
        else:
            l_max_legendre = l_max
        print('Theta {:>.3f}'.format(Theta_deg))
        print('CosTheta: {:>18.12f}'.format(cosTheta))
        Plm_series, Plm_prime_series = \
                associated_Legendre_func_series_no_CS_phase(
                        2, l_max_legendre, cosTheta)
        
        i_offset = 0
        for mode_type in summation_info['mode_types']:

            num_modes = num_modes_dict[mode_type]
            print('Mode type: {:>3}, mode count: {:>5d}'.format(mode_type, num_modes))

            for i in range(num_modes):
                
                # Unpack.
                n = mode_info[mode_type]['n'][i]
                l = mode_info[mode_type]['l'][i]
                f = mode_info[mode_type]['f'][i]
                f_rad_per_s = f*1.0E-3*2.0*np.pi
                Q = mode_info[mode_type]['Q'][i]

                if i == (num_modes - 1):

                    str_end = '\n'

                else:

                    str_end = '\r'
                    
                print('Mode: {:>5d} of {:>5d}, n = {:>5d}, l = {:>5d}, f = {:>7.3f} mHz'.format(
                        i + 1, num_modes, n, l, f), end = str_end)
                
                # Load the eigenfunction information interpolated
                # at the source and receiver locations.
                # Also get the planet radius.
                eigfunc_source, eigfunc_receiver, r_planet = \
                    load_eigenfunc(run_info, mode_type,
                        n,
                        l,
                        f_rad_per_s,
                        cmt['depth_centroid']*1.0E3, # km to m.
                        z_receiver = 0.0,
                        use_mineos = use_mineos,
                        response_correction_params = response_correction_params)

                if (i == 0) & (i_offset == 0):
                
                    # Calculate radial coordinate of event (km).
                    cmt['r_centroid'] = r_planet*1.0E-3 - cmt['depth_centroid']

                # Calculate the coefficients.
                if mode_type == 'S':

                    # Excitation coefficients determined by source location.
                    source_coeffs = \
                        calculate_source_coefficients_spheroidal(
                            l, f_rad_per_s, cmt, eigfunc_source,
                            summation_info['pulse_type'])

                    # Overall coefficients including receiver location.
                    coeffs = calculate_coeffs_spheroidal(
                                source_coeffs, eigfunc_receiver, l,
                                sinTheta, Plm_series[:, l],
                                Plm_prime_series[:, l], sin_cos_Phi_list)

                elif mode_type == 'R':

                    # Excitation coefficients determined by source location.
                    source_coeffs = \
                        calculate_source_coefficients_radial(
                            f_rad_per_s, cmt, eigfunc_source,
                            summation_info['pulse_type'])

                    # Overall coefficients including receiver location.
                    coeffs = calculate_coeffs_radial(
                                source_coeffs, eigfunc_receiver)

                    ## Overall coefficients including receiver location.
                    #coeffs = calculate_coeffs_radial(
                    #            source_coeffs, eigfunc_receiver,
                    #            sinTheta, Plm_series[l, :],
                    #            Plm_prime_series[l, :], sin_cos_Phi_list)

                else:

                    raise NotImplementedError

                # Store output.
                if j == 0:
                    
                    type_list[i + i_offset]     = mode_type_to_int[mode_type]
                    n_list[i + i_offset]        = n
                    l_list[i + i_offset]        = l 
                    f_list[i + i_offset]        = f
                    Q_list[i + i_offset]        = Q 

                A_r_list[j, i + i_offset]       = coeffs['r']
                A_Theta_list[j, i + i_offset]  = coeffs['Theta']
                A_Phi_list[j, i + i_offset]    = coeffs['Phi']

            i_offset = i_offset + i + 1

    # Store station data.
    station_data_frame = pandas.DataFrame(
            {'Phi' : Phi_list,
             'Theta' : Theta_list},
            index = station_list)

    # Store mode data.
    mode_data_frame = pandas.DataFrame(
            {'type' : [mode_int_to_type[x] for x in type_list],
            'n'     : n_list,
            'l'     : l_list,
            'f'     : f_list,
            'Q'     : Q_list},
            columns = ['type', 'n', 'l', 'f', 'Q'])

    # Store coefficient data.
    data_list = [A_r_list, A_Theta_list, A_Phi_list]
    name_list = ['A_r', 'A_Theta', 'A_Phi']
    #
    data_frame_list = []
    for i in range(num_station):
        
        data_dict = {x : y[i, :] for x, y in zip(name_list, data_list)}
        data_frame_list.append(pandas.DataFrame(data_dict, columns = name_list))
    # 
    coeff_data_frame = pandas.concat(data_frame_list, keys = station_list) #keys = list(range(num_station)))

    # Save output.
    print('Saving coefficients to {:}'.format(path_coeffs_data_frame))
    coeff_data_frame.to_pickle(path_coeffs_data_frame)
    #
    print('Saving station information to {:}'.format(path_stations_data_frame))
    station_data_frame.to_pickle(path_stations_data_frame)
    #
    print('Saving mode information to {:}'.format(path_modes_data_frame))
    mode_data_frame.to_pickle(path_modes_data_frame)

    return coeff_data_frame, station_data_frame, mode_data_frame 

def make_time_array(num_t, d_t):

    # Create time span and output arrays.
    t_max = (num_t - 1)*d_t
    t = np.linspace(0.0, t_max, num = num_t)

    return t

def load_time_info(path_timing):

    with open(path_timing, 'r') as in_id:

        line = in_id.readline().split()
        num_t = int(line[0])
        d_t   = float(line[1])

    return num_t, d_t

def sum_coeffs(stations, modes, coeffs, num_t, d_t, dir_out, path_cmt, output_type = 'acceleration', attenuation = 'full', overwrite = False):

    # Get output path.
    path_timing = os.path.join(dir_out, 'timing.txt')
    var_name_dict = {'displacement' : 's', 'acceleration' : 'a'}
    path_out = os.path.join(dir_out, '{:}_r_Theta_Phi.npy'.format(var_name_dict[output_type]))
    if os.path.exists(path_timing) and os.path.exists(path_out) and (not overwrite):

        print('Summation file already exists, skipping calculation and loading: {:}'.format(path_out))
        s_r_Theta_Phi = np.load(path_out)
        print('Loading {:}'.format(path_timing))
        num_t, d_t = load_time_info(path_timing)
        t = make_time_array(num_t, d_t)

        return t, s_r_Theta_Phi

    # Load CMT information.
    cmt = read_mineos_cmt(path_cmt)

    # Get time values.
    t = make_time_array(num_t, d_t)

    # Get station list.
    station_list = list(stations.index)
    num_stations = len(station_list)

    # s is displacement in r, Theta, Phi components.
    s     = np.zeros((3, num_stations, num_t))
    key_list = ['A_r', 'A_Theta', 'A_Phi']

    if (attenuation == 'approx') and (output_type == 'acceleration'):

        print('Cannot use \'approx\' attenuation with \'acceleration\' output. Use \'full\' attenuation instead (the results are equivalent).')
        raise ValueError

    # Decide whether to neglect attenuation.
    if np.any(modes['Q'] == 0.0):

        if attenuation in ['full', 'approx']:

            print("Found modes with Q = 0, ignoring attenuation.")
            attenuation = 'none'

    for i in range(num_stations):

        station = station_list[i]
        print('Summating for station: {:>5}'.format(station))
        coeffs_station = coeffs.loc[station]

        if i == 0:

            n_modes = len(coeffs_station)

        for j in range(n_modes):

            # Angular frequency, rad per s.
            omega = modes['f'][j]*1.0E-3*2.0*np.pi
            #print('\n')
            #print(modes['f'][j], omega)

            if output_type == 'displacement':

                if attenuation == 'none':
                    
                    k0 = 1.0/(omega**2.0)
                    cos_wt = np.cos(omega*t)    

                    for k in range(3):

                        key = key_list[k]
                        A = coeffs_station[key][j]

                        s[k, i, :] = s[k, i, :] + k0*A*(1.0 - cos_wt)

                elif attenuation in ['approx', 'full']:

                    # Gamma (decay rate), 1/s.
                    # Dahlen and Tromp (1998), eq. 9.53.
                    gamma = omega/(2.0*modes['Q'][j])

                    # Sinusoids and decay.
                    cos_wt = np.cos(omega*t)
                    exp_wt = np.exp(-1.0*gamma*t)

                    # Low-attenuation approximation.
                    if attenuation == 'approx':

                        # Evaluate D&T equation 10.61.
                        for k in range(3):

                            key = key_list[k]
                            A = coeffs_station[key][j]

                            s[k, i, :] = s[k, i, :] + (1.0/omega**2.0)*A*(1.0 - cos_wt*exp_wt)

                    # Full form of attenuation.
                    elif attenuation == 'full':

                        # Evaluate D&T eq. 10.51
                        #
                        # Constants relating to omega and gamma.
                        c0 = (omega**2.0 + gamma**2.0)
                        c1 = (omega**2.0 - gamma**2.0) 
                        c2 = (2.0*omega*gamma)
                        #
                        k0 = 1.0/c0
                        k1 = c1/c0
                        k2 = c2/c0
                        #
                        # Sinusoids and decay.
                        sin_wt = np.sin(omega*t)
                        #cos_wt = np.cos(omega*t)
                        #exp_wt = np.exp(-1.0*gamma*t)
                        #
                        for k in range(3):

                            key = key_list[k]
                            A = coeffs_station[key][j]

                            s[k, i, :] = s[k, i, :] + k0*A*(k1*(1.0 - cos_wt*exp_wt) - k2*sin_wt*exp_wt)

                else:

                    raise ValueError

            elif output_type == 'acceleration':

                # Evaluate D&T eq. 10.63.
                #cos_wt = np.cos(omega*t)

                # This modification seems to be necessary to agree with
                # phase of Mineos output.
                # Based on a section of mineos/syndat.f starting with comment
                # "make correction for halfduratio, if tconst > 0"
                x = 0.5*cmt['half_duration']*omega
                cos_wt = np.cos(omega*t - x)

                if attenuation == 'none':

                    for k in range(3):

                        key = key_list[k]
                        A = coeffs_station[key][j]

                        s[k, i, :] = s[k, i, :] + A*cos_wt

                elif attenuation == 'full':

                    # Gamma (decay rate), 1/s.
                    # Dahlen and Tromp (1998), eq. 9.53.
                    gamma = omega/(2.0*modes['Q'][j])

                    exp_wt = np.exp(-1.0*gamma*t)
                    #
                    for k in range(3):

                        key = key_list[k]
                        A = coeffs_station[key][j]

                        s[k, i, :] = s[k, i, :] + A*cos_wt*exp_wt

                else:

                    raise ValueError

            else:

                raise ValueError

    # Save.
    print('Writing {:}'.format(path_out))
    np.save(path_out, s)
    print ('Writing {:}'.format(path_timing))
    with open(path_timing, 'w') as out_id:

        out_id.write('{:>10d} {:>18.12f}'.format(num_t, d_t))

    return t, s

def rotate_r_Theta_Phi_to_e_n_z(station_info, s_r_Theta_Phi):

    # Get station list.
    station_list = list(station_info.index)
    num_stations = len(station_list)

    # Create output array.
    s_e_n_z = np.zeros(s_r_Theta_Phi.shape)
    
    # z-component is simply r-component.
    s_e_n_z[2, :, :] = s_r_Theta_Phi[0, :, :]

    # Loop over stations.
    for i in range(num_stations):

        # Get Theta (anticlockwise angle from south to direction of Theta
        # component).
        station = station_list[i]
        Theta = station_info.loc[station]['Phi']
        # Get Chi (anticlockwise angle from east to direction of Theta
        # component).
        Chi = Theta - (np.pi/2.0)
        cosChi = np.cos(Chi)
        sinChi = np.sin(Chi)
        
        # The east and north components can be found from the Theta and Phi
        # components using trigonometry.
        # East component.
        s_e_n_z[0, i, :] = cosChi*s_r_Theta_Phi[1, i, :] - sinChi*s_r_Theta_Phi[2, i, :]
        # North component.
        s_e_n_z[1, i, :] = sinChi*s_r_Theta_Phi[1, i, :] + cosChi*s_r_Theta_Phi[2, i, :]

    return s_e_n_z

def rotate_e_n_z_to_channels(station_info, s_e_n_z, path_channels, dir_out, overwrite = False):

    # Load channel information.
    channel_dict = read_channel_file(path_channels)

    # Get station list.
    station_list = list(station_info.index)
    num_stations = len(station_list)

    # Get output path list.
    path_out_dict = dict()
    for i in range(num_stations):

        station = station_list[i]
        path_out_dict[station] = dict()

        channels = channel_dict[station]['channels']
        for channel in channels:

            name_out = '{:}_{:}.npy'.format(station, channel)
            path_out = os.path.join(dir_out, name_out)
            path_out_dict[station][channel] = path_out

    # Check output files already exist.
    out_files_exist = []
    if (not overwrite):

        for station in path_out_dict:

            for channel in path_out_dict[station]:

                out_files_exist.append(os.path.exists((path_out_dict[station][channel])))

        if all(out_files_exist):

            print('All channel output files exist, skipping calculation.')
            return

    # Loop over stations.
    for i in range(num_stations):

        station = station_list[i]
        channels = channel_dict[station]['channels']
        
        for channel in channels:

            # Get the unit vector.
            horiz_angle = np.deg2rad(channels[channel]['horiz_angle'])
            vert_angle = np.deg2rad(channels[channel]['vert_angle'])
            e, n, z = polar_coords_to_unit_vector(horiz_angle, vert_angle)

            # Resolve the displacement in the direction of the component. 
            s = s_e_n_z[0, i, :]*e + s_e_n_z[1, i, :]*n + s_e_n_z[2, i, :]*z

            # Save.
            name_out = '{:}_{:}.npy'.format(station, channel)
            path_out = os.path.join(dir_out, name_out)
            print('Writing to {:}'.format(path_out))
            np.save(path_out, s)

    return

def convert_to_mseed(dir_out, path_channels, station_info, path_cmt, differentiate = False):

    # Load timing information.
    name_timing = 'timing.txt'
    path_timing = os.path.join(dir_out, name_timing)
    num_t, d_t = load_time_info(path_timing)

    # Load CMT info.
    cmt = read_mineos_cmt(path_cmt)

    # Load channel information.
    channel_dict = read_channel_file(path_channels)

    # Get station list.
    station_list = list(station_info.index)
    num_stations = len(station_list)

    stream = Stream()
    for i in range(num_stations):

        station = station_list[i]

        channels = channel_dict[station]['channels']
        for channel in channels:

            name_trace = '{:}_{:}.npy'.format(station, channel)
            path_trace = os.path.join(dir_out, name_trace)

            trace_data = np.load(path_trace)
            
            trace_header = {'delta' : d_t, 'station' : station, 'channel' : channel,
                            'starttime' : cmt['datetime_ref']}# + datetime.timedelta(seconds=cmt['half_duration']/2.0)}
#                            'starttime' : cmt_info['}
            trace = Trace(data = trace_data, header = trace_header)
            trace.normalize(norm = 1.0E-9) # Convert from m to nm.

            stream = stream + trace
    
    # Differentiate (go from displacement (m) to velocity (m/s)).
    if differentiate:

        stream.differentiate()

    name_stream = 'stream.mseed'
    path_stream = os.path.join(dir_out, name_stream) 
    print('Writing {:}'.format(path_stream))
    stream.write(path_stream)

    return stream

def main():

    # Parse input arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("path_mode_input", help = "File path (relative or absolute) to Ouroboros mode input file.")
    parser.add_argument("path_summation_input", help = "File path (relative or absolute) to Ouroboros summation input file.")
    parser.add_argument("--use_mineos", action = 'store_true', help = "Use Mineos path_mode_input file, eigenfrequencies and eigenfunctions. Note 1: The path_summation_input file should still be in Ouroboros format. Note 2: This option is for testing. For access to the built-in Mineos synthetics, see Ouroboros/mineos/summation.py.")
    parser.add_argument("--overwrite", action = 'store_true', help = "Use this flag to overwrite existing output files (default: calculations will be skipped if output files detected.")
    input_args = parser.parse_args()
    path_mode_input = input_args.path_mode_input
    path_summation_input = input_args.path_summation_input
    use_mineos = input_args.use_mineos
    overwrite = input_args.overwrite

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

    # Get information about output dirs.
    if use_mineos:

        run_info['dir_model'], run_info['dir_run'] = get_Mineos_out_dirs(run_info) 
        summation_info = get_Mineos_summation_out_dirs(run_info, summation_info,
                            name_summation_dir = 'summation_Ouroboros')

        for key in ['dir_summation', 'dir_channels', 'dir_cmt']:

            mkdir_if_not_exist(summation_info[key])

        summation_info['dir_output'] = summation_info['dir_cmt']

    else:

        raise NotImplementedError

    # If necessary, calculate the surface gravity.
    if summation_info['correct_response']:

        g = get_surface_gravity(run_info)
        response_correction_params = dict()
        response_correction_params['g'] = g


    else:

        response_correction_params = None

    # Calculate the coefficients.
    coeffs, stations, modes = get_coeffs_wrapper(run_info, summation_info,
                use_mineos = use_mineos, overwrite = overwrite,
                response_correction_params = response_correction_params)

    # Do the summation to get r, Theta and Phi components.
    t, s_r_Theta_Phi = sum_coeffs(stations, modes, coeffs, summation_info['n_samples'],
                    summation_info['d_t'], summation_info['dir_output'],
                    summation_info['path_cmt'],
                    attenuation = summation_info['attenuation'],
                    overwrite = overwrite)

    # Rotate into east, north and vertical components.
    s_e_n_z = rotate_r_Theta_Phi_to_e_n_z(stations, s_r_Theta_Phi)

    # Rotate into specified channels.
    rotate_e_n_z_to_channels(stations, s_e_n_z, summation_info['path_channels'],
                summation_info['dir_output'], overwrite = overwrite)

    # Convert to a convenient MSEED stream file.
    convert_to_mseed(summation_info['dir_output'], summation_info['path_channels'], stations, summation_info['path_cmt'])
    
    return

if __name__ == '__main__':

    main()
