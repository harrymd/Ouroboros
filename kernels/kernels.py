import numpy as np

from Ouroboros.constants import G

def gravitational_acceleration(r, rho, Gc = G):
    '''
    Calculates accceleration due to gravity as a function of radius within a
    planet with a radial density distribution.
    Radius and rho must be increasing, starting at the centre of the
    Earth at r = 0.
    Uses Newton's formula g = GM/(r^2) where M is the mass contained by
    a given radius.
    '''
    
    n_r = len(r)
    g   = np.zeros(n_r)
    
    # Define mass of spherical shell as function of radii (missing factor
    # of 4*pi is multiplied later).
    f = rho*(r**2.0)

    # For each radial point...
    for i in range(n_r):
        
        pref = 1.0/(r[i]**2.0)

        # Integrate to get the total mass, then divide by r^2.
        g[i] = pref*np.trapz(f[:i + 1], x = r[:i + 1])
    
    # Take appropriate limit as r -> 0.
    g[0] = 0.0

    # Scale with appropriate constants.
    g = (4.0*np.pi*Gc)*g

    return g

def kernel_spheroidal_ka(r, U, V, dUdr, l, omega):
    '''
    D&T 1998 eq. 9.13.
    Sensitivity of spheroidal mode eigenfrequency with respect to kappa with
    mu and rho fixed.
    '''
   
    # Get asymptotic wavenumber.
    k2      = l*(l + 1.0)
    k       = np.sqrt(k2)

    # Evaluate expression. 
    K_ka    = ((r*dUdr + 2.0*U - k*V)**2.0)/(2.0*omega)

    return K_ka

def kernel_spheroidal_mu(r, U, V, dUdr, dVdr, l, omega, i_fluid = None, return_terms = False):
    '''
    D&T 1998 eq. 9.14.
    Sensitivity of spheroidal mode eigenfrequency with respect to mu with
    kappa and rho fixed.
    '''
    
    # Get asymptotic wavenumber.
    k2      = l*(l + 1.0)
    k       = np.sqrt(k2)
    
    # Evaluate expression.
    a = ((2.0*r*dUdr - 2.0*U + k*V)**2.0)/3.0
    b = (r*dVdr - V + k*U)**2.0
    c = (k2 - 2.0)*(V**2.0)
    # 
    K_mu = (a + b + c)/(2.0*omega)

    # Correct treatment of fluid regions.
    if i_fluid is not None:
        
        K_mu[i_fluid] = 0.0

    # For de-bugging it can be helpful to return the individual terms.
    if return_terms:
        
        a[i_fluid] = 0.0
        b[i_fluid] = 0.0
        c[i_fluid] = 0.0

        a = a/(2.0*omega)
        b = b/(2.0*omega)
        c = c/(2.0*omega)

        return a, b, c

    else:
    
        return K_mu

def get_kernels_S(r, U, V, dUdr, dVdr, l, omega, i_fluid = None):
    '''
    Wrapper to calculate all of the spheroidal-mode kernels.
    '''

    K_ka = kernel_spheroidal_ka(r, U, V, dUdr,          l, omega)
    K_mu = kernel_spheroidal_mu(r, U, V, dUdr, dVdr,    l, omega, i_fluid = i_fluid)

    return K_ka, K_mu

def get_kernels_T(r, W, dWdr, l, omega):
    '''
    Wrapper to calculate all of the toroidal-mode kernels.
    This function has not been written yet.
    '''

    K_mu = np.zeros(W.shape)
    K_rho = np.zeros(W.shape)

    return K_mu, K_rho
