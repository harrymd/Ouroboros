import warnings

import numpy as np
# Gravitational constant (based on Wikipedia, SI units).
G = 6.674e-11  
## G value in Mineos (converted to SI).
#G = 6.6723e-11

g_switch_warning_message = 'g_switch 1 and 2 have not been implemented. Kernels will be approximated with non-gravitating expressions (but using eigenfunctions based on g_switch).'

# Utilities. ------------------------------------------------------------------
def get_indices_of_discontinuities(r):
    
    i_discon = []
    n_r = len(r)
    for i in range(n_r - 1):

        if r[i] == r[i + 1]:
        
            i_discon.append(i)
            
    return i_discon
        
def radial_derivative(r, x):
    '''
    Numerical differentiation of quantity with respect to radius.
    '''
    
    n_r = len(r)
    i_d = get_indices_of_discontinuities(r)
    i_d.append(n_r)
    n_d = len(i_d)

    dxdr = np.zeros(n_r)
    
    i1 = 0
    for i2 in i_d:
        
        # Note: 'Distances must be scalars' error if NumPy version too low.
        dxdr[i1 : i2 + 1] = np.gradient(x[i1 : i2 + 1], r[i1 : i2 + 1])
        i1 = i2 + 1

    return dxdr

# Wrappers. -------------------------------------------------------------------
def kernel_spheroidal(var, omega, r, U, V, l, g_switch, i_fluid = None, rho = None, alpha = None, beta = None, g = None, P = None):
    '''
    Calculates a sensitivity kernel of a spheroidal mode.
    
    Input:
    
    dir_eig
    n
    l
    var 'ka'    kappa
        'mu'    mu
        'rho'   rho
        'alpha' alpha
        'beta'  beta
        'beta'  rhop
    
    Output:
    '''
    
    if var == 'ka':
        
        K = kernel_spheroidal_ka(r, U, V, l, omega)
        
    elif var == 'mu':
        
        K = kernel_spheroidal_mu(r, U, V, l, omega, i_fluid = i_fluid)
        
    elif var == 'rho':
        
        K = kernel_spheroidal_rho(r, U, V, l, omega, g_switch, rho = rho, g = g, P = P)

    elif var == 'alpha':

        K_ka    = kernel_spheroidal_ka(r, U, V, l, omega)

        K = kernel_alpha(rho, alpha, K_ka)

    elif var == 'beta':

        K_ka    = kernel_spheroidal_ka(r, U, V, l, omega)
        K_mu    = kernel_spheroidal_mu(r, U, V, l, omega, i_fluid = i_fluid)

        K = kernel_beta(rho, beta, K_ka, K_mu)

    elif var == 'rhop':
        
        K_ka    = kernel_spheroidal_ka(r, U, V, l, omega)
        K_mu    = kernel_spheroidal_mu(r, U, V, l, omega, i_fluid = i_fluid)
        K_rho   = kernel_spheroidal_rho(r, U, V, l, omega, g_switch, rho = rho, g = g, P = P)

        K = kernel_rhop(alpha, beta, K_ka, K_mu, K_rho)

    elif var == 'abr':

        K_ka    = kernel_spheroidal_ka(r, U, V, l, omega)
        K_mu    = kernel_spheroidal_mu(r, U, V, l, omega, i_fluid = i_fluid)
        K_rho   = kernel_spheroidal_rho(r, U, V, l, omega, g_switch, rho = rho, g = g, P = P)

        K_alpha = kernel_alpha(rho, alpha, K_ka)
        K_beta  = kernel_beta(rho, beta, K_ka, K_mu)
        K_rhop  = kernel_rhop(alpha, beta, K_ka, K_mu, K_rho)

        return K_ka, K_mu, K_rho, K_alpha, K_beta, K_rhop

    else:

        raise ValueError
        
    return K

def get_kernels_spheroidal(omega_rad_per_s, r, U, V, l, vp, vs, g_switch, g = None, P = None, rho = None, i_fluid = None):

    if g_switch == 0:

        assert (g is None) and (P is None)

    elif g_switch == 1:

        assert (g is not None) and (P is None)

    elif g_switch == 2:

        assert (g is not None) and (P is not None)

    # Calculate the kernel.
    # f Frequency (Hz)
    # omega Angular frequency (radians per s).
    #omega = (2.0*np.pi*f)
    K_ka, K_mu, K_rho, K_alpha, K_beta, K_rhop = \
        kernel_spheroidal(
            'abr', omega_rad_per_s, r, U, V, l, g_switch,
            i_fluid = i_fluid,
            g       = g,
            rho     = rho,
            alpha   = vp,
            beta    = vs,
            P       = P)

    ##scale = 1.0E-6*(omega**2.0)/(2.0*np.pi)
    ##scale = 1.0
    #scale = omega**2.0
    #K_ka    = K_ka*scale
    #K_mu    = K_mu*scale
    #K_rho   = K_rho*scale
    #K_alpha = K_alpha*scale
    #K_beta  = K_beta*scale
    #K_rhop  = K_rhop*scale

    return K_ka, K_mu, K_rho, K_alpha, K_beta, K_rhop

def kernel_toroidal(var, omega, r, W, l, rho = None, beta = None):

    if var == 'ka':

        raise ValueError('Toroidal modes are not sensitive to ka.')

    elif var == 'mu':

        K = kernel_toroidal_mu(r, W, l, omega)

    elif var == 'rho':

        K = kernel_toroidal_rho(r, W, l, omega)

    elif var == 'alpha':

        raise ValueError('Toroidal modes are not sensitive to alpha.')

    elif var == 'beta':

        K_mu = kernel_toroidal_mu(r, W, l, omega)
        K_ka = np.zeros(K_mu.shape)
        K = kernel_beta(rho, beta, K_ka, K_mu)

    elif var == 'rhop':

        K_mu    = kernel_toroidal_mu(r, W, l, omega)
        K_ka    = np.zeros(K_mu.shape)
        K_rho   = kernel_toroidal_rho(r, W, l, omega)
        alpha   = np.zeros(beta.shape)

        K = kernel_rhop(alpha, beta, K_ka, K_mu, K_rho)

    elif var == 'br':

        K_mu    = kernel_toroidal_mu(r, W, l, omega)
        K_rho   = kernel_toroidal_rho(r, W, l, omega)
        K_ka    = np.zeros(K_mu.shape)
        alpha   = np.zeros(beta.shape)

        K_alpha = kernel_alpha(rho, alpha, K_ka)
        K_beta  = kernel_beta(rho, beta, K_ka, K_mu)
        K_rhop  = kernel_rhop(alpha, beta, K_ka, K_mu, K_rho)

        return K_mu, K_rho, K_beta, K_rhop

    else:

        raise ValueError

    return K

def get_kernels_toroidal(f, r, W, l, rho, vs):

    # Calculate the kernel.
    # f Frequency (Hz)
    # omega Angular frequency (radians per s).
    omega = (2.0*np.pi*f)
    K_mu, K_rho, K_beta, K_rhop = \
        kernel_toroidal(
            'br', omega, r, W, l,
            rho     = rho,
            beta    = vs)

    scale = 1.0E-6*(omega**2.0)/(2.0*np.pi)
    K_mu    = K_mu*scale
    K_rho   = K_rho*scale
    K_beta  = K_beta*scale
    K_rhop  = K_rhop*scale

    return K_mu, K_rho, K_beta, K_rhop

# Individual kernels for spheroidal modes. ------------------------------------
def common_normalisation(K, omega):

    # Move the (2*omega) term to right hand side.
    K = K/(2.0*omega)

    # Give sensitivity in Hz not rad per s.
    K = K/(2.0*np.pi)

    return K

def kernel_spheroidal_ka(r, U, V, l, omega):
    '''
    Sensitivity of spheroidal mode eigenfrequency with respect to kappa.
    '''
    
    k2      = l*(l + 1.0)
    k       = np.sqrt(k2)

    # Note factor of k difference between V in Mineos and RadialPNM.
    V       = k*V
    
    dUdr    = radial_derivative(r, U)    
    K_ka    = ((r*dUdr + 2.0*U - k*V)**2.0)

    K_ka = common_normalisation(K_ka, omega)

    # Convert from 
    # Hz per Pa per m
    # to
    # mHz per GPa per km
    # ? should be 1.0E12
    K_ka = K_ka*1.0E6

    return K_ka

def kernel_spheroidal_mu(r, U, V, l, omega, i_fluid = None):
    
    k2      = l*(l + 1.0)
    k       = np.sqrt(k2)

    # Note factor of k difference between V in Mineos and RadialPNM.
    V       = k*V
    
    dUdr    = radial_derivative(r, U)
    dVdr    = radial_derivative(r, V)
    
    a = ((2.0*r*dUdr - 2.0*U + k*V)**2.0)/3.0
    b = (r*dVdr - V + k*U)**2.0
    c = (k2 - 2.0)*(V**2.0)
    
    K_mu = (a + b + c)

    K_mu = common_normalisation(K_mu, omega)

    # Convert from 
    # Hz per Pa per m
    # to
    # mHz per GPa per km
    # ? should be 1.0E12
    K_mu = K_mu*1.0E6

    if i_fluid is not None:
        
        K_mu[i_fluid] = 0.0
    
    return K_mu

def kernel_spheroidal_rho(r, U, V, l, omega, g_switch, rho = None, g = None, P = None):
    '''
    Eq. 9.15
    Spheroidal modes (W = 0)
    If rho, g and P are None, the kernel is calculated without the effect of
    gravity.
    '''

    # Calculate k and scale V by K.
    k2      = l*(l + 1.0)
    k       = np.sqrt(k2)

    # Note factor of k difference between V in Mineos and RadialPNM.
    V       = k*V
    
    # Calculate first term.
    a = -1.0*((omega*r)**2.0)*(U**2.0 + V**2.0)
    
    # Case 1: No gravity or perturbation.
    #if g_switch == 0:
    if True:

        if g_switch in [1, 2]:

            warnings.warn(g_switch_warning_message)
            g = None
            P = None
        
        assert (g is None) and (P is None)

        K_rho = a

    ## Case 2. Cowling approximation. Initial gravitational acceleration,
    ## but no perturbation.
    #elif g_switch == 1:
    #    
    #    assert (g is not None) and (P is None)

    #    d = -2.0*g*r*U*(2.0*U - k*V)

    #    K_rho = a + d 

    ## Case 3. Gravity and perturbation.
    #elif g_switch == 2:

    #    assert (g is not None) and (P is not None)

    #    dPdr    = radial_derivative(r, P)
    #    b =  8.0*np.pi*G*rho*((r*U)**2.0)
    #    c = 2.0*(r**2.0)*(U*dPdr + ((k*V*P)/r))
    #    # Apply appropriate limit as r -> 0
    #    c[0] = 0.0
    #    d = -2.0*g*r*U*(2.0*U - k*V)
    #    e = -8.0*np.pi*G*(r**2.0)*kernel_spheroidal_rho_integral(
    #                                r, rho, U, V, k)

    #    K_rho = a + b + c + d + e

    K_rho = common_normalisation(K_rho, omega)

    return K_rho

def old_kernel_spheroidal_rho(r, U, V, l, omega, rho = None, g = None, P = None):
    '''
    Eq. 9.15
    Spheroidal modes (W = 0)
    If rho, g and P are None, the kernel is calculated without the effect of
    gravity.
    '''

    # Calculate k and scale V by K.
    k2      = l*(l + 1.0)
    k       = np.sqrt(k2)

    # Note factor of k difference between V in Mineos and RadialPNM.
    V       = k*V
    
    # Calculate first term.
    a = -1.0*((omega*r)**2.0)*(U**2.0 + V**2.0)
    
    # Case 1: No gravity or perturbation.
    if (g is None):
        
        #print('Case 1')
        assert P is None
        assert rho is None
        
        K_rho = a

        b = 0
        c = 0
        d = 0
        e = 0
        
    # Cases 2 and 3.
    else:
        
        b =  8.0*np.pi*G*rho*((r*U)**2.0)
        d = -2.0*g*r*U*(2.0*U - k*V)
        e = -8.0*np.pi*G*(r**2.0)*kernel_spheroidal_rho_integral(
                                    r, rho, U, V, k)

        # Case 2: Gravity, but no perturbation.
        if (P is None): 
            
            #print('Case 2')
            K_rho = a + b + d + e
        
        # Case 3: Gravity and perturbation.
        else:
            
            #print('Case 3')
            dPdr    = radial_derivative(r, P)
            
            c = 2.0*(r**2.0)*(U*dPdr + ((k*V*P)/r))

            # Apply appropriate limit as r -> 0
            c[0] = 0.0
            
            K_rho = a + b + c + d + e
    
    K_rho = K_rho/(2.0*omega)

    return K_rho

# Individual kernels for toroidal modes. --------------------------------------
def kernel_toroidal_mu(r, W, l, omega):

    k2      = l*(l + 1.0)
    k       = np.sqrt(k2)

    # Note factor of k difference between V in Mineos and RadialPNM.
    W       = k*W
    
    dWdr    = radial_derivative(r, W)

    a = (r*dWdr - W)**2.0
    b = (k2 - 2.0)*(W**2.0)
    
    K_mu = (a + b)

    # Move the (2*omega) term to right hand side.
    K_mu = K_mu/(2.0*omega)

    # Give sensitivity in Hz not rad per s.
    K_mu = K_mu/(2.0*np.pi)
    
    return K_mu

def kernel_toroidal_rho(r, W, l, omega):

    # Calculate k and scale V by K.
    k2      = l*(l + 1.0)
    k       = np.sqrt(k2)

    # Note factor of k difference between V in Mineos and RadialPNM.
    W       = k*W
    
    # Calculate first term.
    a = -1.0*((omega*r)**2.0)*(W**2.0)

    K_rho = a
    
    # Move the (2*omega) term to right hand side.
    K_rho = K_rho/(2.0*omega)

    # Give sensitivity in Hz not rad per s.
    K_rho = K_rho/(2.0*np.pi)

    return K_rho

# Derived kernels. ------------------------------------------------------------
def kernel_alpha(rho, alpha, K_ka):
    '''
    Dahlen and Tromp (1998), eq. 9.21.
    '''
    
    K_alpha = 2.0*rho*alpha*K_ka
    
    return K_alpha

def kernel_beta(rho, beta, K_ka, K_mu):
    '''
    Dahlen and Tromp (1998), eq. 9.22.
    '''
    
    K_beta = 2.0*rho*beta*(K_mu - (4.0/3.0)*K_ka)

    return K_beta

def kernel_rhop(alpha, beta, K_ka, K_mu, K_rho):
    '''
    Dahlen and Tromp (1998), eq. 9.23.
    '''

    K_rhop = (      (alpha**2.0 -(4.0/3.0)*(beta**2.0))*K_ka
                +   (beta**2.0)*K_mu
                +   K_rho)

    #import matplotlib.pyplot as plt
    #
    #fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex = True)

    #ax = ax1
    #ax.plot(alpha)
    #ax.plot(beta)

    #ax = ax2

    #ax.plot(K_ka, label = 'ka')
    #ax.plot(K_mu, label = 'mu')
    #ax.plot(K_rho, label = 'rho')

    #ax.legend()

    #ax = ax3

    #ax.plot((alpha**2.0 -(4.0/3.0)*(beta**2.0))*K_ka, label = '1')
    #ax.plot((beta**2.0)*K_mu, label = '2')
    #ax.plot(K_rho, label = '3')

    #ax.plot(K_rhop, label = 'sum')
    #ax.legend()

    #plt.show()
                
    return K_rhop

# Gravity terms. --------------------------------------------------------------
def kernel_spheroidal_rho_integral(r, rho, U, V, k):

    # Define function to be integrated. Take appropriate limit as r -> 0.
    f       = rho*U*(2.0*U - k*V)/r
    f[0]    = 0.0
    
    # At each depth, calculate integral from that depth to the surface.
    n_r = len(r)
    E   = np.zeros(n_r)
    #
    for i in range(n_r):
    
        E[i] = np.trapz(f[i:], x = r[i:])
        
    return E

def potential(r, U, V, l, rho):
    '''
    Equation 8.55.
    '''

    # Note factor of k difference in V between RadialPNM and Mineos.
    k2      = l*(l + 1.0)
    k       = np.sqrt(k2)
    V       = k*V
    
    nk  = len(r)
    P   = np.zeros(nk)
    for i in range(nk):
        
        P[i] = P_of_r(i, r, U, V, l, rho)

    return P

def P_of_r(i, r, U, V, l, rho):
    '''
    Equation 8.55
    '''
    
    nk = len(r)
    
    if (i == 0):

        I = potential_upper_integral(r[i], r, U, V, l, rho, contains_r0 = True)

    elif (i == (nk - 1)):

        I = potential_lower_integral(r[i], r, U, V, l, rho, contains_r0 = True)

    else:

        #r_low = r[:(i + 1)]
        #r_upp = r[i:]

        Il = potential_lower_integral(
            r[i], r[:(i + 1)], U[:(i + 1)], V[:(i + 1)], l, rho[:(i + 1)], contains_r0 = True)
        Iu = potential_upper_integral(
            r[i], r[i:], U[i:], V[i:], l, rho[i:])

        #r_low = r[:i]
        #r_upp = r[i - 1:]

        #Il = potential_lower_integral(
        #    r[i], r[:i], U[:i], V[:i], l, rho[:i], contains_r0 = True)
        #Iu = potential_upper_integral(
        #    r[i], r[i-1:], U[i-1:], V[i-1:], l, rho[i-1:])

        I  = Il + Iu
    #
    pref = (-4.0*np.pi*G)/((2.0*l) + 1)

    P = pref*I
    
    return P

def potential_lower_integral(ri, r, U, V, l, rho, contains_r0 = False):
    '''
    First term in brackets on RHS of eq. 8.55.
    Note the factor of r^(-l - 1) has been moved inside the integral, because (a/b)^x is more accurate than (a^x)*(b^-x) if a/b ~ 1 and x is large.
    '''
    
    k2  = l*(l + 1.0)
    k   = np.sqrt(k2)

    f   = rho*(l*U + k*V)*((r/ri)**(l + 1.0))
    
    # Apply limiting value at centre of the Earth.
    if contains_r0:
        
        f[0] = 0.0
    
    I = np.trapz(f, x = r)

    return I

def potential_upper_integral(ri, r, U, V, l, rho, contains_r0 = False):
    '''
    Second term in brackets on RHS of eq. 8.55.
    Note the factor of r^l has been moved inside the integral, because (a/b)^x is more accurate than (a^x)*(b^-x) if a/b ~ 1 and x is large.
    '''
    
    k2  = l*(l + 1.0)
    k   = np.sqrt(k2)
    f   = rho*(-1.0*(l + 1.0)*U + k*V)*((ri/r)**l)
    
    # Apply limiting value at centre of the Earth.
    if contains_r0:
        
        f[0] = 0.0

    I = np.trapz(f, x = r)
    
    return I
    
def gravitational_acceleration(r, rho):
    '''
    Calculates accceleration due to gravity as a function of radius within a planet with a radial density distribution.
    Radius and rho must be increasing, starting at the centre of the Earth at r = 0.
    '''
    
    n_r = len(r)
    g   = np.zeros(n_r)
    
    f = rho*(r**2.0)
    for i in range(n_r):
        
        pref = 1.0/(r[i]**2.0)
        g[i] = pref*np.trapz(f[:i + 1], x = r[:i + 1])
    
    # Take appropriate limit as r -> 0.
    g[0] = 0.0

    # Scale.
    g = (4.0*np.pi*G)*g

    return g

def get_gravity_info(r_model, rho_model, r, U, V, l, rho):

    # Calculate the gravitational acceleration.
    # (SI units m/s2)
    g_model      = gravitational_acceleration(r_model, rho_model)

    # Interpolate the gravitational acceleration at the nodal points.
    g = np.interp(r, r_model, g_model)

    # Calculate the gravitational potential (SI units).
    P = potential(r, U, V, l, rho)

    return g_model, g, P
