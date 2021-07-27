'''
Scripts to build finite element matrices.
Different scripts are required for radial/spheroidal/toroidal, solid/fluid and noG/G/GP combinations.
'''

import numpy as np
    
from Ouroboros.constants import G
# Convert gravitational constant to units of cm^3 g^(-1) s^(-2) * e-6.
G = G*1.0E9 
from Ouroboros.modes import lib

# tol_FEM   Used for checking if value is effectively 0.
tol_FEM = 1e-16

# Radial modes. ---------------------------------------------------------------
def radial_solid_GPmixed(model,invV,invV_P,order,order_P,Dr,Dr_P,rho,radius):
    '''
    Create A and B matrices.
    For the radial modes in a solid region.
    Include gravity and Eulerian perturbation.

    For definitions of variables, see modes/compute_modes.py
    '''

    # implement finite element method on solid speroidal modes with gravity field and Euler perturbation
    Np = order+1
    Np_P = order_P+1

    x = model.x
    k = model.k
    x_P = model.xP
    Ki = len(x[0])
    dimension = Ki*order+1
    dimension_P = Ki*order_P+1
    
    A11=np.zeros((dimension,dimension))
    A13=np.zeros((dimension,dimension_P))
    A31=np.zeros((dimension_P,dimension))
    A33=np.zeros((dimension_P,dimension_P))

    B11=np.zeros((dimension,dimension))
    ABuP=np.zeros((dimension,dimension_P))
    B33=np.zeros((dimension_P,dimension_P))
    
    M = invV.T@invV
    if(Np < Np_P):
        MuP = invV.T@invV_P[:Np,:]
    else:
        MuP = invV[:Np_P,:].T@invV_P
    MP = invV_P.T@invV_P
    
    for i in range(Ki):
        Ji = model.J[0,i] #This is a number
        rxi = model.rx[0,i] #This is a number
        mu_i = model.mu[i] #This is a number
        rho_i = model.rho[i]
        ka_i = model.ka[i]
        ri = np.diag(x[:,i]) #This is a matrix
        ri_P = np.diag(x_P[:,i]) #This is a matrix
        g = np.diag(lib.gravfield_lst(x[:,i],rho,radius))
        
        Uprime_phi = ri@M@Dr*rxi
        U_phi_prime = rxi*Dr.T@M@ri
        Uprime_phi_prime = rxi**2*Dr.T@ri@M@ri@Dr
        
        i_order = i*order
        i_order_P = i*order_P
        Aelmt11 = 2*(ka_i-2/3*mu_i)*Uprime_phi + (4*ka_i+4/3*mu_i+mu_i*k**2)*M +\
                    (ka_i+4/3*mu_i)*Uprime_phi_prime + 2*(ka_i-2/3*mu_i)*U_phi_prime +\
                    4*np.pi*G*np.power(rho_i,2)*ri@M@ri - 4*rho_i*M@ri@g
        # manage unit: *1e15/1e15
        A11[i_order:i_order+Np,i_order:i_order+Np] = A11[i_order:i_order+Np,i_order:i_order+Np] +\
                Ji*(Aelmt11+Aelmt11.T)/2
        Aelmt13 = rho_i*ri@ri@MuP@Dr_P*rxi
        # manage unit: *1e15/1e15
        A13[i_order:i_order+Np,i_order_P:i_order_P+Np_P] = A13[i_order:i_order+Np,i_order_P:i_order_P+Np_P] +\
                Ji*Aelmt13
                
        Aelmt31 = rho_i*rxi*Dr_P.T@MuP.T@ri@ri;
        # manage unit: *1e15/1e15
        A31[i_order_P:i_order_P+Np_P,i_order:i_order+Np] = A31[i_order_P:i_order_P+Np_P,i_order:i_order+Np] +\
                Ji*Aelmt31
        
        Aelmt33 = 1/(4*np.pi*G)*(rxi*Dr_P.T@ri_P@MP@ri_P@Dr_P*rxi+k**2*MP)
        # manage unit: *1e15/1e15
        A33[i_order_P:i_order_P+Np_P,i_order_P:i_order_P+Np_P] = A33[i_order_P:i_order_P+Np_P,i_order_P:i_order_P+Np_P] +\
                Ji*(Aelmt33+Aelmt33.T)/2
        
        Belmt11 = rho_i*ri@M@ri
        # manage unit: *1e21/1e15
        B11[i_order:i_order+Np,i_order:i_order+Np] = B11[i_order:i_order+Np,i_order:i_order+Np] +\
                Ji*(Belmt11+Belmt11.T)/2*1e6 #average to improve precision
    
    #calculate average of A13 to avoid small error caused by round off
    A13_average = (A13+A31.T)/2
    # Combine A and B
    A = np.vstack((np.hstack((A11,A13_average)),\
                   np.hstack((A13_average.T,A33))))
    B = np.vstack((np.hstack((B11,ABuP)),\
                   np.hstack((ABuP.T,B33))))
    block_len = [dimension,dimension_P]
    
    return A,B,block_len

def radial_solid_G(model,invV,order,Dr,rho,radius):
    '''
    Create A and B matrices.
    For the radial modes in a solid region.
    Include gravity but no Eulerian perturbation.

    For definitions of variables, see modes/compute_modes.py
    '''

    # implement finite element method on solid radial modes with gravity field
    Np = order+1
    
    x = model.x
    k = model.k
    Ki = len(x[0])
    dimension = Ki*order+1
    
    A11=np.zeros((dimension,dimension))
    B11=np.zeros((dimension,dimension))
    
    M = invV.T@invV
    
    for i in range(Ki):
        Ji = model.J[0,i] #This is a number
        rxi = model.rx[0,i] #This is a number
        mu_i = model.mu[i] #This is a number
        rho_i = model.rho[i]
        ka_i = model.ka[i]
        ri = np.diag(x[:,i]) #2*2 matrix
        g = np.diag(lib.gravfield_lst(x[:,i],rho,radius))
        
        Uprime_phi = ri@M@Dr*rxi
        U_phi_prime = rxi*Dr.T@M@ri
        Uprime_phi_prime = rxi**2*Dr.T@ri@M@ri@Dr
        
        i_order = i*order
        Aelmt11 = 2*(ka_i-2/3*mu_i)*Uprime_phi + (4*ka_i+4/3*mu_i+mu_i*k**2)*M +\
                    (ka_i+4/3*mu_i)*Uprime_phi_prime + 2*(ka_i-2/3*mu_i)*U_phi_prime +\
                    4*np.pi*G*np.power(rho_i,2)*ri@M@ri - 4*rho_i*M@ri@g
        # manage unit: *1e15/1e15
        A11[i_order:i_order+Np,i_order:i_order+Np] = A11[i_order:i_order+Np,i_order:i_order+Np] +\
                Ji*(Aelmt11+Aelmt11.T)/2
                
        Belmt11 = rho_i*ri@M@ri
        # manage unit: *1e21/1e15
        B11[i_order:i_order+Np,i_order:i_order+Np] = B11[i_order:i_order+Np,i_order:i_order+Np] +\
                Ji*(Belmt11+Belmt11.T)/2*1e6 #average to improve precision
    
    block_len = [dimension]
    
    return A11,B11,block_len

def radial_solid_noG(model,invV,order,Dr):
    '''
    Create A and B matrices.
    For the radial modes in a solid region.
    Does not include gravity. 

    For definitions of variables, see modes/compute_modes.py
    '''

    # implement finite element method on solid radial modes
    Np = order+1

    x = model.x
    k = model.k
    Ki = len(x[0])
    dimension = Ki*order+1
    
    A11=np.zeros((dimension,dimension))
    B11=np.zeros((dimension,dimension))
    
    M = invV.T@invV
    
    for i in range(Ki):
        Ji = model.J[0,i] #This is a number
        rxi = model.rx[0,i] #This is a number
        mu_i = model.mu[i] #This is a number
        ka_i = model.ka[i]
        rho_i = model.rho[i]
        ri = np.diag(x[:,i]) #2*2 matrix
        
        Uprime_phi = ri@M@Dr*rxi
        U_phi_prime = rxi*Dr.T@M@ri
        Uprime_phi_prime = rxi**2*Dr.T@ri@M@ri@Dr
        
        i_order = i*order
        Aelmt11 = 2*(ka_i-2/3*mu_i)*Uprime_phi + (4*ka_i+4/3*mu_i+mu_i*k**2)*M +\
                    (ka_i+4/3*mu_i)*Uprime_phi_prime + 2*(ka_i-2/3*mu_i)*U_phi_prime
        # manage unit: *1e15/1e15
        A11[i_order:i_order+Np,i_order:i_order+Np] = A11[i_order:i_order+Np,i_order:i_order+Np] +\
                Ji*(Aelmt11+Aelmt11.T)/2
                
        Belmt11 = rho_i*ri@M@ri
        # manage unit: *1e21/1e15
        B11[i_order:i_order+Np,i_order:i_order+Np] = B11[i_order:i_order+Np,i_order:i_order+Np] +\
                Ji*(Belmt11+Belmt11.T)/2*1e6#average to improve precision
    
    block_len = [dimension]
    
    return A11,B11,block_len

def radial_fluid_GP_mixedPV(model,invV,invV_p,invV_P,order,order_p,order_P,Dr,Dr_p,Dr_P,rho,radius):
    '''
    Create A and B matrices.
    For the radial modes in a fluid region.
    Include gravity and Eulerian perturbation.

    For definitions of variables, see modes/compute_modes.py
    '''

    # implement finite element method on fluid radial modes with Euler pertubation and gravitional field
    Np = order+1
    Np_p = order_p+1
    Np_P = order_P+1
    
    x = model.x
    x_p=model.xp
    x_P = model.xP
    k = model.k
    ka = model.ka
    Ki = len(x[0])
    dimension = Ki*order+1
    dimension_p = Ki*order_p+1
    dimension_P = Ki*order_P+1
    
    A11 = np.zeros((dimension,dimension))
    A13 = np.zeros((dimension,dimension_P))
    A14 = np.zeros((dimension,dimension_p))
    A31 = np.zeros((dimension_P,dimension))
    A33 = np.zeros((dimension_P,dimension_P))
    A41 = np.zeros((dimension_p,dimension))
    A44 = np.zeros((dimension_p,dimension_p))
    ABup = np.zeros((dimension,dimension_p))
    ABuP = np.zeros((dimension,dimension_P))
    ABPp = np.zeros((dimension_P,dimension_p))
    B11 = np.zeros((dimension,dimension))
    B33 = np.zeros((dimension_P,dimension_P))
    B44 = np.zeros((dimension_p,dimension_p))
    
    M = invV.T@invV
    MP = invV_P.T@invV_P
    if Np<Np_P:
        MuP = invV.T@invV_P[:Np,:]
    else:
        MuP = invV[:Np_P,:].T@invV_P
    if Np<Np_p:
        Mup = invV.T@invV_p[:Np,:]
    else:
        Mup = invV[:Np_p,:].T@invV_p
    Mp = invV_p.T@invV_p
    
    for i in range(Ki):
        Ji = model.J[0,i] #This is a number
        rxi = model.rx[0,i] #This is a number
        rho_i = model.rho[i]
        rho_p_i = model.rho_p[i]
        ka_i = ka[i]
        ri = np.diag(x[:,i]) #2*2 matrix if order = 1
        ri_p=np.diag(x_p[:,i])
        ri_P=np.diag(x_P[:,i])
        g = np.diag(lib.gravfield_lst(x[:,i],rho,radius))
        g_xp = np.diag(lib.gravfield_lst(x_p[:,i],rho,radius))
        
        i_order = i*order
        i_order_p = i*order_p
        i_order_P = i*order_P
        
        Aelmt11 = -rho_p_i*g@ri@M@ri - rho_i**2/ka_i*g@ri@M@ri@g
        # manage unit: *1e15/1e15
        A11[i_order:i_order+Np,i_order:i_order+Np] = A11[i_order:i_order+Np,i_order:i_order+Np] +\
                Ji*(Aelmt11+Aelmt11.T)/2
        
        Aelmt13 = rho_i*ri@ri@MuP@Dr_P*rxi
        # manage unit: *1e15/1e15
        A13[i_order:i_order+Np,i_order_P:i_order_P+Np_P] = A13[i_order:i_order+Np,i_order_P:i_order_P+Np_P] +\
                Ji*Aelmt13
        
        Aelmt14 = ri@ri@Mup@Dr_p*rxi + rho_i/ka_i*g@ri@Mup@ri_p
        # manage unit: *1e12/1e15
        A14[i_order:i_order+Np,i_order_p:i_order_p+Np_p] = A14[i_order:i_order+Np,i_order_p:i_order_p+Np_p] +\
                Ji*Aelmt14*1e-3
        
        Aelmt31 = rho_i*rxi*Dr_P.T@MuP.T@ri@ri
        # manage unit: *1e15/1e15
        A31[i_order_P:i_order_P+Np_P,i_order:i_order+Np] = A31[i_order_P:i_order_P+Np_P,i_order:i_order+Np] +\
                Ji*Aelmt31
        
        Aelmt33 = 1/(4*np.pi*G)*(rxi*Dr_P.T@ri_P@MP@ri_P@Dr_P*rxi + k**2*MP)
        # manage unit: *1e15/1e15
        A33[i_order_P:i_order_P+Np_P,i_order_P:i_order_P+Np_P] = A33[i_order_P:i_order_P+Np_P,i_order_P:i_order_P+Np_P] +\
                Ji*(Aelmt33+Aelmt33.T)/2 #Symmetric operation to eliminate little error
        
        Aelmt41 = rxi*Dr_p.T@Mup.T@ri@ri + rho_i/ka_i*ri_p@g_xp@Mup.T@ri
        # manage unit: *1e12/1e15
        A41[i_order_p:i_order_p+Np_p,i_order:i_order+Np] = A41[i_order_p:i_order_p+Np_p,i_order:i_order+Np] +\
                Ji*Aelmt41*1e-3
        
        Aelmt44 = -ri_p@Mp@ri_p/ka_i
        # manage unit: *1e9/1e15
        A44[i_order_p:i_order_p+Np_p,i_order_p:i_order_p+Np_p] = A44[i_order_p:i_order_p+Np_p,i_order_p:i_order_p+Np_p] +\
                Ji*(Aelmt44+Aelmt44.T)/2*1e-6
        
        Belmt11 = rho_i*ri@M@ri
        # manage unit: *1e21/1e15
        B11[i_order:i_order+Np,i_order:i_order+Np] = B11[i_order:i_order+Np,i_order:i_order+Np] +\
                Ji*(Belmt11+Belmt11.T)/2*1e6 #Symmetric operation to eliminate little error
    
    A13_average = (A13+A31.T)/2 #Symmetric operation to eliminate little error
    A14_average = (A14+A41.T)/2
    A = np.vstack((np.hstack((A11, A13_average, A14_average)),\
                    np.hstack((A13_average.T, A33, ABPp)),\
                    np.hstack((A14_average.T, ABPp.T, A44))))
    
    B = np.vstack((np.hstack((B11, ABuP, ABup)),\
                    np.hstack((ABuP.T, B33, ABPp)),\
                    np.hstack((ABup.T, ABPp.T, B44))))
    block_len = [dimension,dimension_P,dimension_p]
    
    return A,B,block_len

def radial_fluid_G_mixedV(model,invV,invV_p,order,order_p,Dr,Dr_p,rho,radius):
    '''
    Create A and B matrices.
    For the radial modes in a fluid region.
    Include gravity but no Eulerian perturbation.

    For definitions of variables, see modes/compute_modes.py
    '''

    # implement finite element method on fluid radial modes with gravity field
    Np = order+1
    Np_p = order_p+1

    x = model.x
    x_p=model.xp
    ka = model.ka
    Ki = len(x[0])
    dimension = Ki*order+1
    dimension_p = Ki*order_p+1
    
    A11 = np.zeros((dimension,dimension))
    A13 = np.zeros((dimension,dimension_p))
    A31 = np.zeros((dimension_p,dimension))
    A33 = np.zeros((dimension_p,dimension_p))
    
    # sqzElementRect is different from sqzElement in the step to squeeze matrix
    ABup = np.zeros((dimension,dimension_p))
    B11 = np.zeros((dimension,dimension))
    B33 = np.zeros((dimension_p,dimension_p))
    
    M = np.matmul(invV.T,invV)
    
    if Np<Np_p:
        Mup = np.matmul(invV.T,invV_p[:Np,:])
    else:
        Mup = np.matmul(invV[:Np_p,:].T,invV_p)
    Mp = np.matmul(invV_p.T,invV_p)
    
    for i in range(Ki):
        Ji = model.J[0,i] #This is a number
        rxi = model.rx[0,i] #This is a number
        rho_i = model.rho[i]
        rho_p_i = model.rho_p[i]
        ka_i = ka[i]
        ri = np.diag(x[:,i]) #2*2 matrix if order = 1
        ri_p=np.diag(x_p[:,i])
        g = np.diag(lib.gravfield_lst(x[:,i],rho,radius))
        
        i_order = i*order
        i_order_p = i*order_p
        
        Aelmt11 = -rho_p_i*g@ri@M@ri - rho_i**2/ka_i*g@ri@M@ri@g
        # manage unit: *1e15/1e15
        A11[i_order:i_order+Np,i_order:i_order+Np] = A11[i_order:i_order+Np,i_order:i_order+Np] +\
                Ji*(Aelmt11+Aelmt11.T)/2
        
        Aelmt13 = np.matmul(np.matmul(np.matmul(ri,ri),Mup),Dr_p)*rxi + rho_i/ka_i*g@ri@Mup@ri_p
        # manage unit: *1e12/1e15
        A13[i_order:i_order+Np,i_order_p:i_order_p+Np_p] = A13[i_order:i_order+Np,i_order_p:i_order_p+Np_p] +\
                Ji*Aelmt13*1e-3
        
        Aelmt31 = rxi*np.matmul(np.matmul(np.matmul(Dr_p.T,Mup.T),ri),ri) + rho_i/ka_i*ri_p@Mup.T@ri@g
        # manage unit: *1e12/1e15
        A31[i_order_p:i_order_p+Np_p,i_order:i_order+Np] = A31[i_order_p:i_order_p+Np_p,i_order:i_order+Np] +\
                Ji*Aelmt31*1e-3
        
        Aelmt33 = -np.matmul(np.matmul(ri_p,Mp),ri_p)/ka_i
        # manage unit: *1e19/1e15
        A33[i_order_p:i_order_p+Np_p,i_order_p:i_order_p+Np_p] = A33[i_order_p:i_order_p+Np_p,i_order_p:i_order_p+Np_p] +\
                Ji*(Aelmt33+Aelmt33.T)/2*1e-6 #Symmetric operation to eliminate little error
        
        Belmt11 = rho_i*np.matmul(np.matmul(ri,M),ri)
        # manage unit: *1e21/1e15
        B11[i_order:i_order+Np,i_order:i_order+Np] = B11[i_order:i_order+Np,i_order:i_order+Np] +\
                Ji*(Belmt11+Belmt11.T)/2*1e6 #Symmetric operation to eliminate little error
    
    A13_average = (A13+A31.T)/2 #Symmetric operation to eliminate little error
    A = np.vstack((np.hstack((A11, A13_average)),\
                    np.hstack((A13_average.T,  A33))))
    
    B = np.vstack((np.hstack((B11, ABup)),\
                    np.hstack((ABup.T, B33))))
    block_len = [dimension,dimension_p]
    
    return A,B,block_len

def radial_fluid_noG_mixedV(model,invV,invV_p,order,order_p,Dr,Dr_p):
    '''
    Create A and B matrices.
    For the radial modes in a fluid region.
    Does not include gravity.

    For definitions of variables, see modes/compute_modes.py
    '''
        # implement finite element method on fluid radial modes
    Np = order+1
    Np_p = order_p+1

    x = model.x
    x_p=model.xp
    ka = model.ka
    Ki = len(x[0])
    dimension = Ki*order+1
    dimension_p = Ki*order_p+1
    
    AB0 = np.zeros((dimension,dimension))
    A13 = np.zeros((dimension,dimension_p))
    A31 = np.zeros((dimension_p,dimension))
    A33 = np.zeros((dimension_p,dimension_p))
    
    # sqzElementRect is different from sqzElement in the step to squeeze matrix
    ABup = np.zeros((dimension,dimension_p))
    B11 = np.zeros((dimension,dimension))
    B33 = np.zeros((dimension_p,dimension_p))
    
    M = np.matmul(invV.T,invV)
    
    if Np<Np_p:
        Mup = np.matmul(invV.T,invV_p[:Np,:])
    else:
        Mup = np.matmul(invV[:Np_p,:].T,invV_p)
    Mp = np.matmul(invV_p.T,invV_p)
    
    for i in range(Ki):
        Ji = model.J[0,i] #This is a number
        rxi = model.rx[0,i] #This is a number
        rho_i = model.rho[i]
        ka_i = ka[i]
        ri = np.diag(x[:,i]) #2*2 matrix if order = 1
        ri_p=np.diag(x_p[:,i])
        
        i_order = i*order
        i_order_p = i*order_p
        Aelmt13 = np.matmul(np.matmul(np.matmul(ri,ri),Mup),Dr_p)*rxi
        # manage unit: *1e12/1e15
        A13[i_order:i_order+Np,i_order_p:i_order_p+Np_p] = A13[i_order:i_order+Np,i_order_p:i_order_p+Np_p] +\
                Ji*Aelmt13*1e-3
        if abs(A13[i_order,i_order_p])<tol_FEM:
            A13[i_order,i_order_p] = 0
        
        Aelmt31 = rxi*np.matmul(np.matmul(np.matmul(Dr_p.T,Mup.T),ri),ri)
        # manage unit: *1e12/1e15
        A31[i_order_p:i_order_p+Np_p,i_order:i_order+Np] = A31[i_order_p:i_order_p+Np_p,i_order:i_order+Np] +\
                Ji*Aelmt31*1e-3
        if abs(A31[i_order_p,i_order])<tol_FEM:
            A31[i_order_p,i_order] = 0
        
        Aelmt33 = -np.matmul(np.matmul(ri_p,Mp),ri_p)/ka_i
        # manage unit: *1e9/1e15
        A33[i_order_p:i_order_p+Np_p,i_order_p:i_order_p+Np_p] = A33[i_order_p:i_order_p+Np_p,i_order_p:i_order_p+Np_p] +\
                Ji*(Aelmt33+Aelmt33.T)/2*1e-6 #Symmetric operation to eliminate little error
        
        Belmt11 = rho_i*np.matmul(np.matmul(ri,M),ri)
        # manage unit: *1e21/1e15
        B11[i_order:i_order+Np,i_order:i_order+Np] = B11[i_order:i_order+Np,i_order:i_order+Np] +\
                Ji*(Belmt11+Belmt11.T)/2*1e6 #Symmetric operation to eliminate little error
    
    A13_average = (A13+A31.T)/2 #Symmetric operation to eliminate little error
    A = np.vstack((np.hstack((AB0, A13_average)),\
                    np.hstack((A13_average.T, A33))))
    
    B = np.vstack((np.hstack((B11,  ABup)),\
                    np.hstack((ABup.T, B33))))
    block_len = [dimension,dimension_p]
    
    return A,B,block_len

# Spheroidal modes. -----------------------------------------------------------
def spheroidal_solid_GPmixed(model, invV, invV_P, order, order_P, Dr, Dr_P,
        rho, radius, anelastic = False):
    '''
    Create A and B matrices.
    For the spheroidal modes in a solid region.
    Includes gravity and Eulerian perturbation.

    For definitions of variables, see modes/compute_modes.py
    '''

    g_switch = 2
    
    # Get coordinates, asymptotic wavenumber, number of points per element,
    # number of elements, size of matrix problem, local mass matrices,
    # and initialised output arrays.
    if anelastic:

        (   x, x_P, k, Np, Np_P, Ki, dimension, dimension_P, M, MuP, MP,
            B11, AB0, ABuP, B33,
            A11_mu, A12_mu_ave, A22_mu,
            A11_ka, A12_ka, A21_ka, A22_ka,
            A13, A23, A31, A32, A33) = \
                prepare_block_spheroidal_GP(model, order, order_P, invV, invV_P,
                    anelastic = True)

        #print("\n")
        #print(A11_mu.shape)
        #print(A12_mu_ave.shape)
        #print(A22_mu.shape)
        #print(A13.shape)
        #print(A31.shape)
        #print(A32.shape)
        #print(A33.shape)

    else:

        (   x, x_P, k, Np, Np_P, Ki, dimension, dimension_P, M, MuP, MP, B11,
            AB0, ABuP, B33, A11, A12, A13, A21, A22, A23, A31, A32, A33) = \
                prepare_block_spheroidal_GP(model, order, order_P, invV, invV_P,
                        anelastic = False)

    # Store params in dictionary.
    GP_params = {   'x_P' : x_P, 'order_P' : order_P, 'Np_P' : Np_P,
                    'MuP' : MuP, 'MP' : MP, 'Dr_P' : Dr_P}
    
    # Loop over elements.
    for i in range(Ki):

        # For this element, get Jacobian, metric, material parameters,
        # coordinates, test-function terms, indices of matrix entries,
        # and gravity.
        (Ji, ri, ri_P, rxi, mu_i, ka_i, rho_i, Uprime_phi, U_phi_prime,
            Uprime_phi_prime, i0, i1, iP0, iP1, g) = prepare_element_spheroidal(
                                            model, x, M, Dr, order, Np, i,
                                            g_switch, rho, radius, GP_params)
        GP_params['rxi']    = rxi
        GP_params['ri_P']   = ri_P

        # Get matrix elements.
        (   Aelmt11_mu, Aelmt11_ka, Aelmt12_mu, Aelmt12_ka, Aelmt21_mu,
            Aelmt21_ka, Aelmt22_mu, Aelmt22_ka,
            Aelmt13, Aelmt31, Aelmt23, Aelmt32, Aelmt33, Belmt11) = \
                get_spheroidal_matrix_elements(k, ri, ka_i, mu_i, rho_i,
                    Uprime_phi, U_phi_prime, Uprime_phi_prime, M, g_switch,
                    g = g, GP_params = GP_params)

        # Store the matrix elements in the full matrices.
        if anelastic:

            A11_mu, A22_mu, A12_mu_ave, A11_ka, A12_ka, A21_ka, A22_ka, B11 = \
                store_spheroidal_matrix_elements_anelastic(
                        A11_mu, A22_mu, A12_mu_ave,
                        A11_ka, A12_ka, A21_ka, A22_ka, B11,
                        Aelmt11_mu, Aelmt12_mu, Aelmt21_mu, Aelmt22_mu,
                        Aelmt11_ka, Aelmt12_ka, Aelmt21_ka, Aelmt22_ka,
                        Belmt11, Ji, i, i0, i1)

        else:

            A11, A12, A21, A22, B11 = \
                store_spheroidal_matrix_elements_elastic(
                        A11, A12, A21, A22, B11,
                        Aelmt11_mu, Aelmt12_mu, Aelmt21_mu, Aelmt22_mu,
                        Aelmt11_ka, Aelmt12_ka, Aelmt21_ka, Aelmt22_ka,
                        Belmt11, Ji, i0, i1)

        A13, A31, A23, A32, A33 = \
            store_spheroidal_matrix_elements_anelastic_GP_terms(A13, A31, A23,
                A32, A33, Aelmt13, Aelmt31, Aelmt23, Aelmt32, Aelmt33, Ji, i0,
                i1, iP0, iP1)

        #Aelmt11 = 2*(ka_i-2/3*mu_i)*Uprime_phi + (4*ka_i+4/3*mu_i+mu_i*k**2)*M +\
        #            (ka_i+4/3*mu_i)*Uprime_phi_prime + 2*(ka_i-2/3*mu_i)*U_phi_prime +\
        #            4*np.pi*G*np.power(rho_i,2)*ri@M@ri - 4*rho_i*M@ri@g
        ## manage unit: *1e15/1e15
        #A11[i_order:i_order+Np,i_order:i_order+Np] = A11[i_order:i_order+Np,i_order:i_order+Np] +\
        #        Ji*(Aelmt11+Aelmt11.T)/2
        ## A12 and A21 are symmetric about diagonal
        #Aelmt12 = (-2*ka_i-5/3*mu_i)*k*M + k*mu_i*Uprime_phi+ -(ka_i-2/3*mu_i)*k*U_phi_prime +\
        #            rho_i*k*g@M@ri
        ## manage unit: *1e15/1e15
        #A12[i_order:i_order+Np,i_order:i_order+Np] = A12[i_order:i_order+Np,i_order:i_order+Np] +\
        #        Ji*Aelmt12
        #
        #Aelmt13 = rho_i*ri@ri@MuP@Dr_P*rxi
        ## manage unit: *1e15/1e15
        #A13[i_order:i_order+Np,i_order_P:i_order_P+Np_P] = A13[i_order:i_order+Np,i_order_P:i_order_P+Np_P] +\
        #        Ji*Aelmt13
        #
        #Aelmt21 = (-2*ka_i-5/3*mu_i)*k*M + k*mu_i*U_phi_prime - (ka_i-2/3*mu_i)*k*Uprime_phi +\
        #            rho_i*k*ri@M@g
        ## manage unit: *1e15/1e15
        #A21[i_order:i_order+Np,i_order:i_order+Np] = A21[i_order:i_order+Np,i_order:i_order+Np] +\
        #        Ji*Aelmt21
        #
        #Aelmt22=(k**2*(ka_i+4/3*mu_i)-mu_i)*M - mu_i*Uprime_phi - mu_i*U_phi_prime + mu_i*Uprime_phi_prime
        ## manage unit: *1e15/1e15
        #A22[i_order:i_order+Np,i_order:i_order+Np] = A22[i_order:i_order+Np,i_order:i_order+Np] +\
        #        Ji*(Aelmt22+Aelmt22.T)/2
        #        
        #Aelmt23 = k*rho_i*ri@MuP
        ## manage unit: *1e15/1e15
        #A23[i_order:i_order+Np,i_order_P:i_order_P+Np_P] = A23[i_order:i_order+Np,i_order_P:i_order_P+Np_P] +\
        #        Ji*Aelmt23
        #        
        #Aelmt31 = rho_i*rxi*Dr_P.T@MuP.T@ri@ri;
        ## manage unit: *1e15/1e15
        #A31[i_order_P:i_order_P+Np_P,i_order:i_order+Np] = A31[i_order_P:i_order_P+Np_P,i_order:i_order+Np] +\
        #        Ji*Aelmt31
        #
        #Aelmt32 = k*rho_i*MuP.T@ri
        ## manage unit: *1e15/1e15
        #A32[i_order_P:i_order_P+Np_P,i_order:i_order+Np] = A32[i_order_P:i_order_P+Np_P,i_order:i_order+Np] +\
        #        Ji*Aelmt32
        #
        #Aelmt33 = 1/(4*np.pi*G)*(rxi*Dr_P.T@ri_P@MP@ri_P@Dr_P*rxi+k**2*MP)
        ## manage unit: *1e15/1e15
        #A33[i_order_P:i_order_P+Np_P,i_order_P:i_order_P+Np_P] = A33[i_order_P:i_order_P+Np_P,i_order_P:i_order_P+Np_P] +\
        #        Ji*(Aelmt33+Aelmt33.T)/2
        #
        #Belmt11 = rho_i*ri@M@ri
        ## manage unit: *1e21/1e15
        #B11[i_order:i_order+Np,i_order:i_order+Np] = B11[i_order:i_order+Np,i_order:i_order+Np] +\
        #        Ji*(Belmt11+Belmt11.T)/2*1e6 #average to improve precision

    # Record the size of this block.
    block_len = [dimension, dimension, dimension_P]

    # Combine matrix blocks.
    if anelastic:
        
        A_ka, A_mu, B = combine_spheroidal_matrix_blocks_anelastic_GP(
            A11_mu, A12_mu_ave, A22_mu,
            A11_ka, A12_ka, A21_ka, A22_ka,
            A13, A31, A23, A32, A33,
            B11, AB0, ABuP, B33, Ki)

        # Note negative sign on A matrices due to different convention
        # for defining eigenvalue problem in anelastic case.
        return -A_ka, -A_mu, B, Ki, dimension, block_len

    else:
        
        A, B = combine_spheroidal_matrix_blocks_elastic_GP(A11, A12, A21, A13,
                A31, A22, A23, A32, A33, B11, AB0, ABuP, B33)

        return A, B, Ki, dimension, block_len
    
    ##calculate average of A12 to avoid small error caused by round off
    #A12_average = (A12+A21.T)/2
    #A13_average = (A13+A31.T)/2
    #A23_average = (A23+A32.T)/2
    ## Combine A11,A12,A21,A22 to get A. Combine B11 with 0(B12) to get B
    #A = np.vstack((np.hstack((A11,A12_average,A13_average)),\
    #               np.hstack((A12_average.T,A22,A23_average)),\
    #               np.hstack((A13_average.T,A23_average.T,A33))))
    #B = np.vstack((np.hstack((B11,AB0,ABuP)),\
    #               np.hstack((AB0,B11,ABuP)),\
    #               np.hstack((ABuP.T,ABuP.T,B33))))

    return

def spheroidal_solid_noG_or_G(model, invV, order, Dr, rho, radius, g_switch,
        anelastic = False):
    '''
    Create A and B matrices.
    For the spheroidal modes in a solid region.
    Includes gravity but no Eulerian perturbation.

    For definitions of variables, see modes/compute_modes.py
    '''

    # Get coordinates, asymptotic wavenumber, number of points per element,
    # number of elements, size of matrix problem, local mass matrix,
    # and initialised output arrays.
    if anelastic:

        (x, k, Np, Ki, dimension, M, B11, B12, A11_ka, A12_ka, A21_ka, A22_ka,
                A11_mu, A22_mu, A12_mu_ave) = \
                    prepare_block_spheroidal_noG_or_G(model, order, invV,
                                                anelastic = True)

    else:

        x, k, Np, Ki, dimension, M, B11, B12, A11, A12, A21, A22 = \
                prepare_block_spheroidal_noG_or_G(model, order, invV,
                                            anelastic = False)
    
    # Loop over elements.
    for i in range(Ki):

        # For this element, get Jacobian, metric, material parameters,
        # coordinates, test-function terms, indices of matrix entries,
        # and gravity.
        (Ji, ri, _, rxi, mu_i, ka_i, rho_i, Uprime_phi, U_phi_prime,
            Uprime_phi_prime, i0, i1, _, _, g) = prepare_element_spheroidal(
                                            model, x, M, Dr, order, Np, i,
                                            g_switch, rho, radius, None)

        # Calculate matrix elements.
        (   Aelmt11_mu, Aelmt11_ka, Aelmt12_mu, Aelmt12_ka, Aelmt21_mu,
            Aelmt21_ka, Aelmt22_mu, Aelmt22_ka, Belmt11) = \
                get_spheroidal_matrix_elements(k, ri, ka_i, mu_i,
                        rho_i, Uprime_phi, U_phi_prime, Uprime_phi_prime, M,
                        g_switch, g = g)

        # Store the matrix elements in the full matrices.
        if anelastic:

            A11_mu, A22_mu, A12_mu_ave, A11_ka, A12_ka, A21_ka, A22_ka, B11 = \
                store_spheroidal_matrix_elements_anelastic(
                        A11_mu, A22_mu, A12_mu_ave,
                        A11_ka, A12_ka, A21_ka, A22_ka, B11,
                        Aelmt11_mu, Aelmt12_mu, Aelmt21_mu, Aelmt22_mu,
                        Aelmt11_ka, Aelmt12_ka, Aelmt21_ka, Aelmt22_ka,
                        Belmt11, Ji, i, i0, i1)

        else:

            A11, A12, A21, A22, B11 = \
                store_spheroidal_matrix_elements_elastic(
                        A11, A12, A21, A22, B11,
                        Aelmt11_mu, Aelmt12_mu, Aelmt21_mu, Aelmt22_mu,
                        Aelmt11_ka, Aelmt12_ka, Aelmt21_ka, Aelmt22_ka,
                        Belmt11, Ji, i0, i1)

    # Record the size of this block.
    block_len = [dimension, dimension]

    # Combine matrix blocks.
    if anelastic:

        A_ka, A_mu, B = combine_spheroidal_matrix_blocks_anelastic_noG_or_G(
                            A11_ka, A12_ka, A21_ka, A22_ka,
                            A11_mu, A12_mu_ave, A22_mu,
                            B11, B12, Ki)

        # Note negative sign on A matrices due to different convention
        # for defining eigenvalue problem in anelastic case.
        return -A_ka, -A_mu, B, Ki, dimension, block_len

    else:

        A, B = combine_spheroidal_matrix_blocks_elastic_noG_or_G(
                A11, A12, A21, A22,
                B11, B12)

        #return A, B, block_len
        return A, B, Ki, dimension, block_len
    
    return

def prepare_block_spheroidal_noG_or_G(model, order, invV, anelastic = False):

    # Unpack variables.
    x = model.x
    k = model.k

    # Define sizes of matrices.
    Np = (order + 1)
    Ki = len(x[0])
    dimension = (Ki * order) + 1

    # Note: B12 is 0.
    B11 = np.zeros((dimension, dimension))
    B12 = np.zeros((dimension, dimension))

    # Calculate local mass matrix.
    M = (invV.T @ invV)
    
    if anelastic:

        A11_ka = np.zeros((dimension, dimension))
        A12_ka = np.zeros((dimension, dimension))
        A21_ka = np.zeros((dimension, dimension))
        A22_ka = np.zeros((dimension, dimension))

        A11_mu      = np.zeros((Ki, dimension, dimension))
        A22_mu      = np.zeros((Ki, dimension, dimension))
        A12_mu_ave  = np.zeros((Ki, dimension, dimension))

        return (x, k, Np, Ki, dimension, M, B11, B12, A11_ka, A12_ka, A21_ka,
                A22_ka, A11_mu, A22_mu, A12_mu_ave)

    else:

        A11 = np.zeros((dimension, dimension))
        A12 = np.zeros((dimension, dimension))
        A21 = np.zeros((dimension, dimension))
        A22 = np.zeros((dimension, dimension))

        return x, k, Np, Ki, dimension, M, B11, B12, A11, A12, A21, A22

    return

def prepare_block_spheroidal_GP(model, order, order_P, invV, invV_P,
        anelastic = False):

    # Unpack variables.
    x = model.x
    x_P = model.xP
    k = model.k

    # Define sizes of matrices.
    Ki = len(x[0])
    Np = (order + 1)
    Np_P = (order_P + 1)
    dimension = (Ki * order) + 1
    dimension_P = (Ki * order_P) + 1

    # Calculate local mass matrices.
    M = (invV.T @ invV)
    MP = (invV_P.T @ invV_P)
    if (Np < Np_P):

        MuP = (invV.T @ invV_P[: Np, :])

    else:

        MuP = (invV[: Np_P, :].T @ invV_P)

    # Prepare output arrays.
    A13     = np.zeros((dimension,      dimension_P))
    A23     = np.zeros((dimension,      dimension_P))
    A31     = np.zeros((dimension_P,    dimension))
    A32     = np.zeros((dimension_P,    dimension))
    A33     = np.zeros((dimension_P,    dimension_P))

    B11     = np.zeros((dimension,      dimension))
    AB0     = np.zeros((dimension,      dimension))
    ABuP    = np.zeros((dimension,      dimension_P))
    B33     = np.zeros((dimension_P,    dimension_P))

    if anelastic:

        A11_mu     = np.zeros((Ki, dimension,      dimension))
        A12_mu_ave = np.zeros((Ki, dimension,      dimension))
        A22_mu     = np.zeros((Ki, dimension,      dimension))

        A11_ka     = np.zeros((dimension,      dimension))
        A12_ka     = np.zeros((dimension,      dimension))
        A21_ka     = np.zeros((dimension,      dimension))
        A22_ka     = np.zeros((dimension,      dimension))

        return (x, x_P, k, Np, Np_P, Ki, dimension, dimension_P, M, MuP, MP,
                B11, AB0, ABuP, B33,
                A11_mu, A12_mu_ave, A22_mu,
                A11_ka, A12_ka, A21_ka, A22_ka,
                A13, A23, A31, A32, A33)

    else:

        A11     = np.zeros((dimension,      dimension))
        A12     = np.zeros((dimension,      dimension))
        A21     = np.zeros((dimension,      dimension))
        A22     = np.zeros((dimension,      dimension))

        return (x, x_P, k, Np, Np_P, Ki, dimension, dimension_P, M, MuP, MP,
                B11, AB0, ABuP, B33,
                A11, A12, A13, A21, A22, A23, A31, A32, A33)

    return

def prepare_element_spheroidal(model, x, M, Dr, order, Np, i, g_switch, rho,
        radius, GP_params):
    '''
    '''

    # Unpack gravity perturbation parameters if required.
    if g_switch == 2:

        assert GP_params is not None
        x_P     = GP_params['x_P']
        order_P = GP_params['order_P']
        Np_P    = GP_params['Np_P']

    # Get Jacobian, metric, material parameters and coordinates of this
    # element.
    Ji      = model.J[0, i]
    rxi     = model.rx[0, i]
    mu_i    = model.mu[i]
    ka_i    = model.ka[i]
    rho_i   = model.rho[i]
    ri      = np.diag(x[:, i])

    # Get metric of perturbation (if required).
    if g_switch == 2:

        ri_P = np.diag(x_P[:, i])

    else:

        ri_P = None

    # Get gravity (if required).
    if g_switch == 0:

        g = None

    elif g_switch in [1, 2]:

        g = np.diag(lib.gravfield_lst(x[:, i], rho, radius))

    else:

        raise ValueError
    
    # Calculate terms involving test function.
    Uprime_phi = ri @ M @ Dr * rxi
    U_phi_prime = rxi * Dr.T @ M @ ri
    Uprime_phi_prime = (rxi ** 2.0) * Dr.T @ ri @ M @ ri @ Dr
    
    # Get indices of matrix entries for this element.
    i_order = (i * order)
    i0      = i_order
    i1      = (i0 + Np)

    # Get indices of perturbation matrix entries for this element.
    if g_switch == 2:

        i_order_P = (i * order_P)
        iP0 = i_order_P
        iP1 = (iP0 + Np_P)

    else:

        iP0 = None
        iP1 = None

    return (Ji, ri, ri_P, rxi, mu_i, ka_i, rho_i, Uprime_phi, U_phi_prime,
                Uprime_phi_prime, i0, i1, iP0, iP1, g)

def get_spheroidal_matrix_elements(k, ri, ka_i, mu_i, rho_i, Uprime_phi,
        U_phi_prime, Uprime_phi_prime, M, g_switch, g = None, GP_params = None):

    if g_switch == 2:

        assert GP_params is not None
        MuP     = GP_params['MuP']
        MP      = GP_params['MP']
        Dr_P    = GP_params['Dr_P']
        rxi     = GP_params['rxi']
        ri_P    = GP_params['ri_P']
    
    # Calculate A11 (top-left part of spring matrix) for this element.
    #
    #Aelmt11 = 2*(ka_i-2/3*mu_i)*Uprime_phi + (4*ka_i+4/3*mu_i+mu_i*k**2)*M +\
    #            (ka_i+4/3*mu_i)*Uprime_phi_prime + 2*(ka_i-2/3*mu_i)*U_phi_prime
    #Aelmt11_mu = (  - (4.0 / 3.0) * mu_i * Uprime_phi
    #                + ((4.0 / 3.0) + (k ** 2.0)) * mu_i * M
    #                + (4.0 / 3.0) * mu_i * Uprime_phi_prime
    #                - (4.0 / 3.0) * mu_i * U_phi_prime)
    Aelmt11_mu = mu_i * (   - (4.0 / 3.0) * Uprime_phi
                            + ((4.0 / 3.0) + (k ** 2.0)) * M
                            + (4.0 / 3.0) * Uprime_phi_prime
                            - (4.0 / 3.0) * U_phi_prime)
    Aelmt11_ka = ka_i * (   + 2.0 * Uprime_phi
                            + 4.0 * M
                            + Uprime_phi_prime
                            + 2.0 * U_phi_prime)

    # Calculate A12 (bottom-left part of spring matrix) for this element.
    # (A21 and A12 are symmetric about the diagonal). 
    # manage unit: *1e15/1e15
    #Aelmt12 = (-2*ka_i-5/3*mu_i)*k*M + k*mu_i*Uprime_phi+ -(ka_i-2/3*mu_i)*k*U_phi_prime
    Aelmt12_mu = k * mu_i * (   - (5.0 / 3.0) * M
                                + Uprime_phi
                                + (2.0 / 3.0) * U_phi_prime)
    Aelmt12_ka = -k * ka_i * (  (2.0 * M) + U_phi_prime)

    # Calculate A21 (top-right part of spring matrix) for this element.
    # (A21 and A12 are symmetric about the diagonal). 
    # manage unit: *1e15/1e15
    #Aelmt21 = (-2*ka_i-5/3*mu_i)*k*M + k*mu_i*U_phi_prime - (ka_i-2/3*mu_i)*k*Uprime_phi
    Aelmt21_mu = k * mu_i * (   - (5.0 / 3.0) * M
                                + U_phi_prime
                                + (2.0 / 3.0) * Uprime_phi)
    Aelmt21_ka = -k * ka_i * (  (2.0 * M) +  Uprime_phi)

    # Calculate A22 (bottom-right part of spring matrix) for this element.
    # manage unit: *1e15/1e15
    #Aelmt22=(k**2*(ka_i+4/3*mu_i)-mu_i)*M - mu_i*Uprime_phi - mu_i*U_phi_prime + mu_i*Uprime_phi_prime
    Aelmt22_mu = mu_i * (   + (((k ** 2.0) * (4.0 / 3.0)) - 1.0) * M
                            - Uprime_phi
                            - U_phi_prime
                            + Uprime_phi_prime)
    Aelmt22_ka = (k ** 2.0) * ka_i * M
    
    # Calculate B11 (diagonal part of mass matrix) for this element.
    # manage unit: *1e21/1e15
    Belmt11 = (rho_i * ri @ M @ ri)
    unit_factor_B11 = 1.0E6
    Belmt11 = (Belmt11 * unit_factor_B11)

    # Calculate gravity perturbation terms.
    if g_switch == 2:

        Aelmt13 = (rho_i * ri @ ri @ MuP   @ Dr_P   * rxi)
        Aelmt31 = (rho_i * rxi * Dr_P.T @ MuP.T @ ri @ ri)
        Aelmt23 = (k * rho_i * ri @ MuP  )
        Aelmt32 = (k * rho_i * MuP.T @ ri)
        Aelmt33 = (1.0 / (4.0 * np.pi * G)) * \
                    (rxi * Dr_P.T @ ri_P @ MP @ ri_P @ Dr_P * rxi
                        + (k ** 2.0) * MP)

    # Calculate gravity terms.
    if g_switch in [1, 2]:

        assert g is not None

    if g_switch == 0:

        pass

    elif g_switch in [1, 2]:

        Aelmt11_g = ( 4.0 * np.pi * G * (rho_i ** 2.0) * ri @ M @ ri
                    - 4.0 * rho_i * M @ ri @ g)
        Aelmt12_g = (rho_i * k * g @ M @ ri)
        Aelmt21_g = (rho_i * k * ri @ M @ g)

    else:

        raise ValueError

    # We combine the gravity terms with the kappa terms because they are
    # both not frequency dependent.
    if g_switch in [1, 2]:

        Aelmt11_ka = Aelmt11_ka + Aelmt11_g
        Aelmt12_ka = Aelmt12_ka + Aelmt12_g
        Aelmt21_ka = Aelmt21_ka + Aelmt21_g

    if g_switch in [0, 1]:

        return (Aelmt11_mu, Aelmt11_ka, Aelmt12_mu, Aelmt12_ka, Aelmt21_mu,
                Aelmt21_ka, Aelmt22_mu, Aelmt22_ka, Belmt11)

    else:

        return (Aelmt11_mu, Aelmt11_ka, Aelmt12_mu, Aelmt12_ka, Aelmt21_mu,
                Aelmt21_ka, Aelmt22_mu, Aelmt22_ka,
                Aelmt13, Aelmt31, Aelmt23, Aelmt32, Aelmt33, Belmt11)

    return

def store_spheroidal_matrix_elements_elastic(
        A11, A12, A21, A22, B11,
        Aelmt11_mu, Aelmt12_mu, Aelmt21_mu, Aelmt22_mu,
        Aelmt11_ka, Aelmt12_ka, Aelmt21_ka, Aelmt22_ka,
        Belmt11, Ji, i0, i1):

    # Combine mu and kappa parts of spring matrix.
    Aelmt11 = Aelmt11_mu + Aelmt11_ka
    Aelmt12 = Aelmt12_mu + Aelmt12_ka
    Aelmt21 = Aelmt21_mu + Aelmt21_ka
    Aelmt22 = Aelmt22_mu + Aelmt22_ka

    # Enforce symmetry of diagonal parts of spring matrices.
    Aelmt11 = (Aelmt11 + Aelmt11.T) / 2.0
    Aelmt22 = (Aelmt22 + Aelmt22.T) / 2.0

    # Store (note factor of Jacobian).
    A11[i0 : i1, i0 : i1] += (Ji * Aelmt11)
    A12[i0 : i1, i0 : i1] += (Ji * Aelmt12)
    A21[i0 : i1, i0 : i1] += (Ji * Aelmt21)
    A22[i0 : i1, i0 : i1] += (Ji * Aelmt22)

    # Store mass matrix elements.
    B11 = store_spheroidal_mass_matrix_elements(B11, Belmt11, Ji, i0, i1)

    return A11, A12, A21, A22, B11

def store_spheroidal_matrix_elements_anelastic(
        A11_mu, A22_mu, A12_mu_ave,
        A11_ka, A12_ka, A21_ka, A22_ka, B11,
        Aelmt11_mu, Aelmt12_mu, Aelmt21_mu, Aelmt22_mu,
        Aelmt11_ka, Aelmt12_ka, Aelmt21_ka, Aelmt22_ka,
        Belmt11, Ji, i, i0, i1):

    # Enforce symmetry of diagonal parts of ka spring matrix, mu
    # spring matrix.
    Aelmt11_mu = (Aelmt11_mu + Aelmt11_mu.T) / 2.0
    Aelmt22_mu = (Aelmt22_mu + Aelmt22_mu.T) / 2.0
    #
    Aelmt11_ka = (Aelmt11_ka + Aelmt11_ka.T) / 2.0
    Aelmt22_ka = (Aelmt22_ka + Aelmt22_ka.T) / 2.0

    # Enforce symmetry of off-diagonal part of mu spring matrix.
    Aelmt12_mu_ave = (Aelmt12_mu + Aelmt21_mu.T) / 2.0 

    # Store (note factor of Jacobian).
    A11_mu      [i, i0 : i1, i0 : i1] += (Ji * Aelmt11_mu)
    A22_mu      [i, i0 : i1, i0 : i1] += (Ji * Aelmt22_mu)
    A12_mu_ave  [i, i0 : i1, i0 : i1] += (Ji * Aelmt12_mu_ave)
    #
    A11_ka[i0 : i1, i0 : i1] += (Ji * Aelmt11_ka)
    A12_ka[i0 : i1, i0 : i1] += (Ji * Aelmt12_ka)
    A21_ka[i0 : i1, i0 : i1] += (Ji * Aelmt21_ka)
    A22_ka[i0 : i1, i0 : i1] += (Ji * Aelmt22_ka)

    # Store mass matrix elements.
    B11 = store_spheroidal_mass_matrix_elements(B11, Belmt11, Ji, i0, i1)

    return A11_mu, A22_mu, A12_mu_ave, A11_ka, A12_ka, A21_ka, A22_ka, B11

def store_spheroidal_mass_matrix_elements(B11, Belmt11, Ji, i0, i1):

    # Store mass matrix (note factor of Jacobian).
    Belmt11 = (Belmt11 + Belmt11.T) / 2.0
    B11[i0 : i1, i0 : i1] += (Ji * Belmt11)

    return B11

def store_spheroidal_matrix_elements_anelastic_GP_terms(A13, A31, A23, A32, A33,
        Aelmt13, Aelmt31, Aelmt23, Aelmt32, Aelmt33, Ji, i0, i1, iP0, iP1): 

    # Enforce symmetry of diagonal parts of spring matrix.
    Aelmt33 = (Aelmt33 + Aelmt33.T) / 2.0

    # Store (note factor of Jacobian).
    A13[i0 : i1,    iP0 : iP1] += Ji*Aelmt13
    A31[iP0 : iP1,    i0 : i1] += Ji*Aelmt31
    A23[i0 : i1,    iP0 : iP1] += Ji*Aelmt23
    A32[iP0 : iP1,    i0 : i1] += Ji*Aelmt32
    A33[iP0 : iP1,  iP0 : iP1] += Ji*Aelmt33

    return A13, A31, A23, A32, A33

def combine_spheroidal_matrix_blocks_elastic_noG_or_G(A11, A12, A21, A22,
        B11, B12):

    # Enforce symmetry of off-diagonal parts of spring matrix.
    A12_ave = (A12 + A21.T) / 2.0

    # Combine matrices.
    A = np.vstack((np.hstack((A11,              A12_ave)),
                   np.hstack((A12_ave.T,    A22))))
    B = combine_spheroidal_mass_matrix_blocks(B11, B12)

    return A, B

def combine_spheroidal_matrix_blocks_anelastic_noG_or_G(A11_ka, A12_ka, A21_ka,
        A22_ka, A11_mu, A12_mu_ave, A22_mu, B11, B12, Ki):

    # Enforce symmetry of off-diagonal parts of ka spring matrix.
    A12_ka_ave = (A12_ka + A21_ka.T) / 2.0

    # Combine matrices to get ka spring matrix.
    A_ka = np.vstack((  np.hstack((A11_ka,              A12_ka_ave)),
                        np.hstack((A12_ka_ave.T,    A22_ka))))

    # For each element, combine matrices to get mu spring matrix.
    A_mu = np.zeros((Ki, *A_ka.shape))
    for i in range(Ki):

        A_mu[i, :, :] = \
            np.vstack(( np.hstack((A11_mu[i, :, :], A12_mu_ave[i, :, :])),
                        np.hstack((A12_mu_ave[i, :, :].T,  A22_mu[i, :, :]))))

    # Combine mass matrix blocks.
    B = combine_spheroidal_mass_matrix_blocks(B11, B12)

    return A_ka, A_mu, B

def combine_spheroidal_matrix_blocks_elastic_GP(A11, A12, A21, A13, A31, A22,
        A23, A32, A33, B11, AB0, ABuP, B33):

    # Enforce symmetry of off-diagonal parts of spring matrix.
    A12_ave = (A12 + A21.T) / 2.0
    A13_ave = (A13 + A31.T) / 2
    A23_ave = (A23 + A32.T) / 2

    # Combine matrices.
    A = np.vstack((np.hstack((A11,          A12_ave,    A13_ave)),
                   np.hstack((A12_ave.T,    A22,        A23_ave)),
                   np.hstack((A13_ave.T,    A23_ave.T,  A33))))

    B = np.vstack((np.hstack((B11,      AB0,    ABuP)),
                   np.hstack((AB0,      B11,    ABuP)),
                   np.hstack((ABuP.T,   ABuP.T, B33))))

    return A, B

def combine_spheroidal_matrix_blocks_anelastic_GP(
        A11_mu, A12_mu_ave, A22_mu,
        A11_ka, A12_ka, A21_ka, A22_ka,
        A13, A31, A23, A32, A33,
        B11, AB0, ABuP, B33, Ki):

    # Enforce symmetry of off-diagonal parts of spring matrix.
    A12_ka_ave = (A12_ka + A21_ka.T) / 2.0
    A13_ave = (A13 + A31.T) / 2
    A23_ave = (A23 + A32.T) / 2

    # Combine matrices.
    A_ka = np.vstack((  np.hstack((A11_ka,          A12_ka_ave, A13_ave)),
                        np.hstack((A12_ka_ave.T,    A22_ka,     A23_ave)),
                        np.hstack((A13_ave.T,       A23_ave.T,  A33))))

    # For each element, combine matrices to get mu spring matrix.
    #A_mu = np.zeros((Ki, *A_ka.shape))
    #
    ## Create zero blocks with appropriate shape.
    #A13_mu = np.zeros(A13_ave.shape)
    #A23_mu = np.zeros(A23_ave.shape)
    #A33_mu = np.zeros(A13.shape)
    ##
    #for i in range(Ki):

    #    A_mu[i, :, :] = \
    #     np.vstack((np.hstack((A11_mu[i, :, :],         A12_mu_ave[i, :, :], A13_mu)),
    #                np.hstack((A12_mu_ave[i, :, :].T,   A22_mu[i, :, :],     A23_mu)),
    #                np.hstack((A13_mu,                  A23_mu,              A33_mu))))

    #B = np.vstack((np.hstack((B11,      AB0,    ABuP)),
    #               np.hstack((AB0,      B11,    ABuP)),
    #               np.hstack((ABuP.T,   ABuP.T, B33))))

    # Create zero blocks with appropriate shape.
    size_A_mu = A11_mu.shape[1] + A22_mu.shape[1]
    A_mu = np.zeros((Ki, size_A_mu, size_A_mu))

    for i in range(Ki):
        
        A_mu[i, :, :] = \
            np.vstack((np.hstack((A11_mu[i, :, :],         A12_mu_ave[i, :, :])),
                    np.hstack((A12_mu_ave[i, :, :].T,   A22_mu[i, :, :]    ))))

    B = np.vstack((np.hstack((B11,      AB0,    ABuP)),
                   np.hstack((AB0,      B11,    ABuP)),
                   np.hstack((ABuP.T,   ABuP.T, B33))))

    return A_ka, A_mu, B

def combine_spheroidal_mass_matrix_blocks(B11, B12):

    # Combine B11 with 0 (B12) to get mass matrix.
    B = np.vstack((np.hstack((B11, B12)),
                   np.hstack((B12, B11))))

    return B

def spheroidal_fluid_GP_mixedPV(model,invV,invV_p,invV_P,invV_V,order,order_p,order_P,order_V,Dr,Dr_p,Dr_P,Dr_V,rho,radius):
    '''
    Create A and B matrices.
    For the spheroidal modes in a fluid region.
    Includes gravity and Eulerian perturbation.

    For definitions of variables, see modes/compute_modes.py
    '''

    # implement finite element method on fluid speroidal modes with Euler pertubation and gravitional field
    Np = order+1
    Np_p = order_p+1
    Np_V = order_V+1
    Np_P = order_P+1
    x = model.x
    x_p=model.xp
    x_P = model.xP
    x_V=model.xV
    k = model.k
    ka = model.ka
    Ki = len(x[0])
    dimension = Ki*order+1
    dimension_p = Ki*order_p+1
    dimension_V = Ki*order_V+1
    dimension_P = Ki*order_P+1
    
    A11 = np.zeros((dimension,dimension))
    A13 = np.zeros((dimension,dimension_P))
    A14 = np.zeros((dimension,dimension_p))
    A23 = np.zeros((dimension_V,dimension_P))
    A24 = np.zeros((dimension_V,dimension_p))
    A31 = np.zeros((dimension_P,dimension))
    A32 = np.zeros((dimension_P,dimension_V))
    A33 = np.zeros((dimension_P,dimension_P))
    A41 = np.zeros((dimension_p,dimension))
    A42 = np.zeros((dimension_p,dimension_V))
    A44 = np.zeros((dimension_p,dimension_p))
    ABup = np.zeros((dimension,dimension_p))
    ABvp = np.zeros((dimension_V,dimension_p))
    ABuv = np.zeros((dimension,dimension_V))
    ABvv = np.zeros((dimension_V,dimension_V))
    ABuP = np.zeros((dimension,dimension_P))
    ABPp = np.zeros((dimension_P,dimension_p))
    ABvP = np.zeros((dimension_V,dimension_P))
    B11 = np.zeros((dimension,dimension))
    B22 = np.zeros((dimension_V,dimension_V))
    B33 = np.zeros((dimension_P,dimension_P))
    B44 = np.zeros((dimension_p,dimension_p))
    
    M = invV.T@invV
    MP = invV_P.T@invV_P
    if Np<Np_P:
        MuP = invV.T@invV_P[:Np,:]
    else:
        MuP = invV[:Np_P,:].T@invV_P
    if Np<Np_p:
        Mup = invV.T@invV_p[:Np,:]
    else:
        Mup = invV[:Np_p,:].T@invV_p
    if Np_V<Np_p:
        Mvp = invV_V.T@invV_p[:Np_V,:]
    else:
        Mvp = invV_V[:Np_p,:].T@invV_p
    if Np_V<Np_P:
        MvP = invV_V.T@invV_P[:Np_V,:]
    else:
        MvP = invV_V[:Np_P,:].T@invV_P
    Mv = invV_V.T@invV_V
    Mp = invV_p.T@invV_p
    
    for i in range(Ki):
        Ji = model.J[0,i] #This is a number
        rxi = model.rx[0,i] #This is a number
        rho_i = model.rho[i]
        rho_p_i = model.rho_p[i]
        ka_i = ka[i]
        ri = np.diag(x[:,i]) #2*2 matrix if order = 1
        ri_p=np.diag(x_p[:,i])
        ri_P=np.diag(x_P[:,i])
        ri_V=np.diag(x_V[:,i])
        g = np.diag(lib.gravfield_lst(x[:,i],rho,radius))
        g_xp = np.diag(lib.gravfield_lst(x_p[:,i],rho,radius))
        
        i_order = i*order
        i_order_p = i*order_p
        i_order_P = i*order_P
        i_order_V = i*order_V
        
        Aelmt11 = -rho_p_i*g@ri@M@ri - rho_i**2/ka_i*g@ri@M@ri@g
        # manage unit: *1e15/1e15
        A11[i_order:i_order+Np,i_order:i_order+Np] = A11[i_order:i_order+Np,i_order:i_order+Np] +\
                Ji*(Aelmt11+Aelmt11.T)/2
        
        Aelmt13 = rho_i*ri@ri@MuP@Dr_P*rxi
        # manage unit: *1e15/1e15
        A13[i_order:i_order+Np,i_order_P:i_order_P+Np_P] = A13[i_order:i_order+Np,i_order_P:i_order_P+Np_P] +\
                Ji*Aelmt13
        
        Aelmt14 = ri@ri@Mup@Dr_p*rxi + rho_i/ka_i*g@ri@Mup@ri_p
        # manage unit: *1e12/1e15
        A14[i_order:i_order+Np,i_order_p:i_order_p+Np_p] = A14[i_order:i_order+Np,i_order_p:i_order_p+Np_p] +\
                Ji*Aelmt14*1e-3
        
        Aelmt23 = k*rho_i*ri_V@MvP
        # manage unit: *1e15/1e15
        A23[i_order_V:i_order_V+Np_V,i_order_P:i_order_P+Np_P] = A23[i_order_V:i_order_V+Np_V,i_order_P:i_order_P+Np_P] +\
                Ji*Aelmt23
        
        Aelmt24 = k*ri_V@Mvp
        # manage unit: *1e12/1e15
        A24[i_order_V:i_order_V+Np_V,i_order_p:i_order_p+Np_p] = A24[i_order_V:i_order_V+Np_V,i_order_p:i_order_p+Np_p] +\
                Ji*Aelmt24*1e-3
        
        Aelmt31 = rho_i*rxi*Dr_P.T@MuP.T@ri@ri
        # manage unit: *1e15/1e15
        A31[i_order_P:i_order_P+Np_P,i_order:i_order+Np] = A31[i_order_P:i_order_P+Np_P,i_order:i_order+Np] +\
                Ji*Aelmt31
        
        Aelmt32 = k*rho_i*MvP.T@ri_V
        # manage unit: *1e15/1e15
        A32[i_order_P:i_order_P+Np_P,i_order_V:i_order_V+Np_V] = A32[i_order_P:i_order_P+Np_P,i_order_V:i_order_V+Np_V] +\
                Ji*Aelmt32
        
        Aelmt33 = 1/(4*np.pi*G)*(rxi*Dr_P.T@ri_P@MP@ri_P@Dr_P*rxi + k**2*MP)
        # manage unit: *1e15/1e15
        A33[i_order_P:i_order_P+Np_P,i_order_P:i_order_P+Np_P] = A33[i_order_P:i_order_P+Np_P,i_order_P:i_order_P+Np_P] +\
                Ji*(Aelmt33+Aelmt33.T)/2 #Symmetric operation to eliminate little error
        
        Aelmt41 = rxi*Dr_p.T@Mup.T@ri@ri + rho_i/ka_i*ri_p@g_xp@Mup.T@ri
        # manage unit: *1e12/1e15
        A41[i_order_p:i_order_p+Np_p,i_order:i_order+Np] = A41[i_order_p:i_order_p+Np_p,i_order:i_order+Np] +\
                Ji*Aelmt41*1e-3
        
        Aelmt42 = k*Mvp.T@ri_V
        # manage unit: *1e12/1e15
        A42[i_order_p:i_order_p+Np_p,i_order_V:i_order_V+Np_V] = A42[i_order_p:i_order_p+Np_p,i_order_V:i_order_V+Np_V] +\
                Ji*Aelmt42*1e-3
        
        Aelmt44 = -ri_p@Mp@ri_p/ka_i
        # manage unit: *1e9/1e15
        A44[i_order_p:i_order_p+Np_p,i_order_p:i_order_p+Np_p] = A44[i_order_p:i_order_p+Np_p,i_order_p:i_order_p+Np_p] +\
                Ji*(Aelmt44+Aelmt44.T)/2*1e-6
        
        Belmt11 = rho_i*ri@M@ri
        # manage unit: *1e21/1e15
        B11[i_order:i_order+Np,i_order:i_order+Np] = B11[i_order:i_order+Np,i_order:i_order+Np] +\
                Ji*(Belmt11+Belmt11.T)/2*1e6 #Symmetric operation to eliminate little error
        
        Belmt22 = rho_i*ri_V@Mv@ri_V
        # manage unit: *1e21/1e15
        B22[i_order_V:i_order_V+Np_V,i_order_V:i_order_V+Np_V] = B22[i_order_V:i_order_V+Np_V,i_order_V:i_order_V+Np_V] +\
                Ji*(Belmt22+Belmt22.T)/2*1e6 #Symmetric operation to eliminate little error
    
    A13_average = (A13+A31.T)/2 #Symmetric operation to eliminate little error
    A14_average = (A14+A41.T)/2
    A23_average = (A23+A32.T)/2 #Symmetric operation to eliminate little error
    A24_average = (A24+A42.T)/2
    A = np.vstack((np.hstack((A11, ABuv, A13_average, A14_average)),\
                    np.hstack((ABuv.T, ABvv, A23_average, A24_average)),\
                    np.hstack((A13_average.T, A23_average.T, A33, ABPp)),\
                    np.hstack((A14_average.T, A24_average.T, ABPp.T, A44))))
    
    B = np.vstack((np.hstack((B11, ABuv, ABuP, ABup)),\
                    np.hstack((ABuv.T, B22, ABvP, ABvp)),\
                    np.hstack((ABuP.T, ABvP.T, B33, ABPp)),\
                    np.hstack((ABup.T, ABvp.T, ABPp.T, B44))))
    block_len = [dimension,dimension_V,dimension_P,dimension_p]
    
    return A, B, Ki, block_len

def spheroidal_fluid_G_mixedV(model,invV,invV_p,invV_V,order,order_p,order_V,Dr,Dr_p,Dr_V,rho,radius):
    '''
    Create A and B matrices.
    For the spheroidal modes in a fluid region.
    Includes gravity but no Eulerian perturbation.

    For definitions of variables, see modes/compute_modes.py
    '''

    # implement finite element method on fluid speroidal modes
    Np = order+1
    Np_p = order_p+1
    Np_V = order_V+1
    x = model.x
    x_p=model.xp
    x_V=model.xV
    k = model.k
    ka = model.ka
    Ki = len(x[0])
    dimension = Ki*order+1
    dimension_p = Ki*order_p+1
    dimension_V = Ki*order_V+1
    
    A11 = np.zeros((dimension,dimension))
    A13 = np.zeros((dimension,dimension_p))
    A23 = np.zeros((dimension_V,dimension_p))
    A31 = np.zeros((dimension_p,dimension))
    A32 = np.zeros((dimension_p,dimension_V))
    A33 = np.zeros((dimension_p,dimension_p))
    
    # sqzElementRect is different from sqzElement in the step to squeeze matrix
    ABup = np.zeros((dimension,dimension_p))
    ABvp = np.zeros((dimension_V,dimension_p))
    ABuv = np.zeros((dimension,dimension_V))
    ABvv = np.zeros((dimension_V,dimension_V))
    B11 = np.zeros((dimension,dimension))
    B22 = np.zeros((dimension_V,dimension_V))
    B33 = np.zeros((dimension_p,dimension_p))
    
    M = np.matmul(invV.T,invV)
    
    if Np<Np_p:
        Mup = np.matmul(invV.T,invV_p[:Np,:])
    else:
        Mup = np.matmul(invV[:Np_p,:].T,invV_p)
    if Np_V<Np_p:
        Mvp = np.matmul(invV_V.T,invV_p[:Np_V,:])
    else:
        Mvp = np.matmul(invV_V[:Np_p,:].T,invV_p)
    Mv = np.matmul(invV_V.T,invV_V)
    Mp = np.matmul(invV_p.T,invV_p)
    
    for i in range(Ki):
        Ji = model.J[0,i] #This is a number
        rxi = model.rx[0,i] #This is a number
        rho_i = model.rho[i]
        rho_p_i = model.rho_p[i]
        ka_i = ka[i]
        ri = np.diag(x[:,i]) #2*2 matrix if order = 1
        ri_p=np.diag(x_p[:,i])
        ri_V=np.diag(x_V[:,i])
        g = np.diag(lib.gravfield_lst(x[:,i],rho,radius))
        
        i_order = i*order
        i_order_p = i*order_p
        i_order_V = i*order_V
        
        Aelmt11 = -rho_p_i*g@ri@M@ri - rho_i**2/ka_i*g@ri@M@ri@g
        # manage unit: *1e15/1e15
        A11[i_order:i_order+Np,i_order:i_order+Np] = A11[i_order:i_order+Np,i_order:i_order+Np] +\
                Ji*(Aelmt11+Aelmt11.T)/2
        
        Aelmt13 = np.matmul(np.matmul(np.matmul(ri,ri),Mup),Dr_p)*rxi + rho_i/ka_i*g@ri@Mup@ri_p
        # manage unit: *1e12/1e15
        A13[i_order:i_order+Np,i_order_p:i_order_p+Np_p] = A13[i_order:i_order+Np,i_order_p:i_order_p+Np_p] +\
                Ji*Aelmt13*1e-3
        
        Aelmt23 = k*np.matmul(ri_V,Mvp)
        # manage unit: *1e12/1e15
        A23[i_order_V:i_order_V+Np_V,i_order_p:i_order_p+Np_p] = A23[i_order_V:i_order_V+Np_V,i_order_p:i_order_p+Np_p] +\
                Ji*Aelmt23*1e-3
        
        Aelmt31 = rxi*np.matmul(np.matmul(np.matmul(Dr_p.T,Mup.T),ri),ri) + rho_i/ka_i*ri_p@Mup.T@ri@g
        # manage unit: *1e12/1e15
        A31[i_order_p:i_order_p+Np_p,i_order:i_order+Np] = A31[i_order_p:i_order_p+Np_p,i_order:i_order+Np] +\
                Ji*Aelmt31*1e-3
        
        Aelmt32 = k*np.matmul(Mvp.T,ri_V)
        # manage unit: *1e12/1e15
        A32[i_order_p:i_order_p+Np_p,i_order_V:i_order_V+Np_V] = A32[i_order_p:i_order_p+Np_p,i_order_V:i_order_V+Np_V] +\
                Ji*Aelmt32*1e-3
        
        Aelmt33 = -np.matmul(np.matmul(ri_p,Mp),ri_p)/ka_i
        # manage unit: *1e9/1e15
        A33[i_order_p:i_order_p+Np_p,i_order_p:i_order_p+Np_p] = A33[i_order_p:i_order_p+Np_p,i_order_p:i_order_p+Np_p] +\
                Ji*(Aelmt33+Aelmt33.T)/2*1e-6 #Symmetric operation to eliminate little error
        
        Belmt11 = rho_i*np.matmul(np.matmul(ri,M),ri)
        # manage unit: *1e21/1e15
        B11[i_order:i_order+Np,i_order:i_order+Np] = B11[i_order:i_order+Np,i_order:i_order+Np] +\
                Ji*(Belmt11+Belmt11.T)/2*1e6 #Symmetric operation to eliminate little error
        
        Belmt22 = rho_i*np.matmul(np.matmul(ri_V,Mv),ri_V)
        # manage unit: *1e21/1e15
        B22[i_order_V:i_order_V+Np_V,i_order_V:i_order_V+Np_V] = B22[i_order_V:i_order_V+Np_V,i_order_V:i_order_V+Np_V] +\
                Ji*(Belmt22+Belmt22.T)/2*1e6 #Symmetric operation to eliminate little error
    
    A13_average = (A13+A31.T)/2 #Symmetric operation to eliminate little error
    A23_average = (A23+A32.T)/2 #Symmetric operation to eliminate little error
    A = np.vstack((np.hstack((A11, ABuv, A13_average)),\
                    np.hstack((ABuv.T, ABvv, A23_average)),\
                    np.hstack((A13_average.T, A23_average.T, A33))))
    
    B = np.vstack((np.hstack((B11, ABuv, ABup)),\
                    np.hstack((ABuv.T, B22, ABvp)),\
                    np.hstack((ABup.T, ABvp.T, B33))))
    block_len = [dimension,dimension_V,dimension_p]
    
    return A, B, Ki, block_len

def spheroidal_fluid_noG_mixedV(model, invV, invV_p, invV_V, order, order_p,
        order_V, Dr, Dr_p, Dr_V):
    '''
    Create A and B matrices.
    For the spheroidal modes in a fluid region.
    Does not include gravity.

    For definitions of variables, see modes/compute_modes.py
    '''

    # implement finite element method on fluid speroidal modes
    Np = order+1
    Np_p = order_p+1
    Np_V = order_V+1
    x = model.x
    x_p=model.xp
    x_V=model.xV
    k = model.k
    ka = model.ka
    Ki = len(x[0])
    dimension = Ki*order+1
    dimension_p = Ki*order_p+1
    dimension_V = Ki*order_V+1
    
    AB0 = np.zeros((dimension,dimension))
    A13 = np.zeros((dimension,dimension_p))
    A23 = np.zeros((dimension_V,dimension_p))
    A31 = np.zeros((dimension_p,dimension))
    A32 = np.zeros((dimension_p,dimension_V))
    A33 = np.zeros((dimension_p,dimension_p))
    
    # sqzElementRect is different from sqzElement in the step to squeeze matrix
    ABup = np.zeros((dimension,dimension_p))
    ABvp = np.zeros((dimension_V,dimension_p))
    ABuv = np.zeros((dimension,dimension_V))
    ABvv = np.zeros((dimension_V,dimension_V))
    B11 = np.zeros((dimension,dimension))
    B22 = np.zeros((dimension_V,dimension_V))
    B33 = np.zeros((dimension_p,dimension_p))
    
    M = np.matmul(invV.T,invV)
    
    if Np<Np_p:
        Mup = np.matmul(invV.T,invV_p[:Np,:])
    else:
        Mup = np.matmul(invV[:Np_p,:].T,invV_p)
    if Np_V<Np_p:
        Mvp = np.matmul(invV_V.T,invV_p[:Np_V,:])
    else:
        Mvp = np.matmul(invV_V[:Np_p,:].T,invV_p)
    Mv = np.matmul(invV_V.T,invV_V)
    Mp = np.matmul(invV_p.T,invV_p)
    
    for i in range(Ki):
        Ji = model.J[0,i] #This is a number
        rxi = model.rx[0,i] #This is a number
        rho_i = model.rho[i]
        ka_i = ka[i]
        ri = np.diag(x[:,i]) #2*2 matrix if order = 1
        ri_p=np.diag(x_p[:,i])
        ri_V=np.diag(x_V[:,i])#computation of x_V needs to be fixed
        
        i_order = i*order
        i_order_p = i*order_p
        i_order_V = i*order_V
        Aelmt13 = np.matmul(np.matmul(np.matmul(ri,ri),Mup),Dr_p)*rxi
        # manage unit: *1e12/1e15
        A13[i_order:i_order+Np,i_order_p:i_order_p+Np_p] = A13[i_order:i_order+Np,i_order_p:i_order_p+Np_p] +\
                Ji*Aelmt13*1e-3
        if abs(A13[i_order,i_order_p])<tol_FEM:
            A13[i_order,i_order_p] = 0
        
        Aelmt23 = k*np.matmul(ri_V,Mvp)
        # manage unit: *1e12/1e15
        A23[i_order_V:i_order_V+Np_V,i_order_p:i_order_p+Np_p] = A23[i_order_V:i_order_V+Np_V,i_order_p:i_order_p+Np_p] +\
                Ji*Aelmt23*1e-3
        
        Aelmt31 = rxi*np.matmul(np.matmul(np.matmul(Dr_p.T,Mup.T),ri),ri)
        # manage unit: *1e12/1e15
        A31[i_order_p:i_order_p+Np_p,i_order:i_order+Np] = A31[i_order_p:i_order_p+Np_p,i_order:i_order+Np] +\
                Ji*Aelmt31*1e-3
        if abs(A31[i_order_p,i_order])<tol_FEM:
            A31[i_order_p,i_order] = 0
        
        Aelmt32 = k*np.matmul(Mvp.T,ri_V)
        # manage unit: *1e12/1e15
        A32[i_order_p:i_order_p+Np_p,i_order_V:i_order_V+Np_V] = A32[i_order_p:i_order_p+Np_p,i_order_V:i_order_V+Np_V] +\
                Ji*Aelmt32*1e-3
        
        Aelmt33 = -np.matmul(np.matmul(ri_p,Mp),ri_p)/ka_i
        # manage unit: *1e9/1e15
        A33[i_order_p:i_order_p+Np_p,i_order_p:i_order_p+Np_p] = A33[i_order_p:i_order_p+Np_p,i_order_p:i_order_p+Np_p] +\
                Ji*(Aelmt33+Aelmt33.T)/2*1e-6 #Symmetric operation to eliminate little error
        
        Belmt11 = rho_i*np.matmul(np.matmul(ri,M),ri)
        # manage unit: *1e21/1e15
        B11[i_order:i_order+Np,i_order:i_order+Np] = B11[i_order:i_order+Np,i_order:i_order+Np] +\
                Ji*(Belmt11+Belmt11.T)/2*1e6 #Symmetric operation to eliminate little error
        
        Belmt22 = rho_i*np.matmul(np.matmul(ri_V,Mv),ri_V)
        # manage unit: *1e21/1e15
        B22[i_order_V:i_order_V+Np_V,i_order_V:i_order_V+Np_V] = B22[i_order_V:i_order_V+Np_V,i_order_V:i_order_V+Np_V] +\
                Ji*(Belmt22+Belmt22.T)/2*1e6 #Symmetric operation to eliminate little error
    
    A13_average = (A13+A31.T)/2 #Symmetric operation to eliminate little error
    A23_average = (A23+A32.T)/2 #Symmetric operation to eliminate little error
    A = np.vstack((np.hstack((AB0, ABuv, A13_average)),\
                    np.hstack((ABuv.T, ABvv, A23_average)),\
                    np.hstack((A13_average.T, A23_average.T, A33))))
    
    B = np.vstack((np.hstack((B11, ABuv, ABup)),\
                    np.hstack((ABuv.T, B22, ABvp)),\
                    np.hstack((ABup.T, ABvp.T, B33))))
    block_len = [dimension,dimension_V,dimension_p]
    
    return A, B, Ki, dimension, dimension_V, dimension_p, block_len

# Toroidal modes. -------------------------------------------------------------
def toroidal(model, invV, order, Dr, anelastic = False):
    '''
    Build spring matrix A and mass matrix B such that
        A x = om^2 B x
    for the toroidal modes in one solid region (toroidal modes do not exist
    in fluid regions).

    For definitions of variables, see modes/compute_modes.py
    '''

    # Unpack variables.
    Np  = (order + 1)
    mu  = model.mu
    rho = model.rho
    x   = model.x
    k   = model.k
    Ki  = len(x[0])
    dimension = (Ki * order) + 1
    
    # Prepare output matrices.
    B = np.zeros((dimension, dimension)) 
    if anelastic:
        
        # In the anelastic case, we have a separate spring matrix for each
        # layer.
        A = np.zeros((Ki, dimension, dimension))

    else:
        
        # In the elastic case, we have a single spring matrix.
        A = np.zeros((dimension, dimension)) 
    
    # Calculate local mass matrix M.
    M = (invV.T @ invV)
    
    for i in range(Ki):
        
        # Calculate spring matrix A (four terms on RHS of weak form equation)
        # and mass matrix B (one term on LHS).
        # Enforce symmetry of output and account for units.
        #
        # Ji    Jacobian of i_th element (a scalar).
        # rxi   rx of i_th elment (a scalar).
        # ri    r (metric) of i_th element (a 2x2 matrix).
        Ji  = model.J[0, i]
        rxi = model.rx[0, i]
        ri  = np.diag(x[:, i])
        i_order = (i * order)
        i0 = i_order
        i1 = (i_order + Np)
        #
        Aelmt = Ji * mu[i] * (
                    - (rxi * ((Dr.T @ M) @ ri)
                    + (ri @ (M @ Dr)) * rxi)
                    + (k ** 2.0 - 1.0) * M
                    + (rxi ** 2.0) * ((((Dr.T @ ri) @ M) @ ri) @ Dr))
        #
        unit_factor_A = 1.0 # 1e15 / 1e15
        Aelmt = (Aelmt + Aelmt.T) / 2.0 
        Aelmt = (Aelmt * unit_factor_A)
        #
        Belmt = Ji * rho[i] * ((ri @ M) @ ri)
        #
        unit_factor_B = 1.0E6 # 1e21 / 1e15
        Belmt = (Belmt + Belmt.T) / 2.0 
        Belmt = (Belmt * unit_factor_B)
        
        # Store.
        if anelastic:
            
            # Note negative sign in anelastic case, due to different
            # convention for writing eigenvalue problem.
            A[i, i0 : i1, i0 : i1] = A[i, i0 : i1, i0 : i1] - Aelmt

        else:

            A[i0 : i1, i0 : i1] = A[i0 : i1, i0 : i1] + Aelmt

        B[i0 : i1, i0 : i1] = B[i0 : i1, i0 : i1] + Belmt
    
    #return Mmu, B, Ki, dimension
    return A, B

## Anelastic versions. ---------------------------------------------------------
#def old_solid_noG_an(model,invV,order,Dr):
#    # replace np.matmul with @
#    # implement finite element method on solid speroidal modes
#    Np = order+1
#    #mu = model.mu
#    #rho = model.rho
#    x = model.x
#    k = model.k
#    #ka = model.ka
#    Ki = len(x[0])
#    dimension = Ki*order+1
#    
#    A11=np.zeros((dimension,dimension))
#    A12=np.zeros((dimension,dimension))
#    A21=np.zeros((dimension,dimension))
#    A22=np.zeros((dimension,dimension))
#
#    B11=np.zeros((dimension,dimension))
#    B12=np.zeros((dimension,dimension)) #B12 = 0, just add terms
#    
#    Mmu = np.zeros((Ki,2*dimension,2*dimension))
#    
#    M = invV.T@invV
#    
#    for i in range(Ki):
#        Ji = model.J[0,i] #This is a number
#        rxi = model.rx[0,i] #This is a number
#        mu_i = model.mu[i] #This is a number
#        ka_i = model.ka[i]
#        rho_i = model.rho[i]
#        ri = np.diag(x[:,i]) #2*2 matrix
#        
#        Uprime_phi = ri@M@Dr*rxi
#        U_phi_prime = rxi*Dr.T@M@ri
#        Uprime_phi_prime = rxi**2*Dr.T@ri@M@ri@Dr
#        
#        i_order = i*order
#        Aelmt11 = 2*ka_i*Uprime_phi + 4*ka_i*M +\
#                    ka_i*Uprime_phi_prime + 2*ka_i*U_phi_prime
#        Aelmt11mu = 2*-2/3*mu_i*Uprime_phi + (4/3*mu_i+mu_i*k**2)*M +\
#                    4/3*mu_i*Uprime_phi_prime + 2*-2/3*mu_i*U_phi_prime
#        # manage unit: *1e15/1e15
#        A11[i_order:i_order+Np,i_order:i_order+Np] = A11[i_order:i_order+Np,i_order:i_order+Np] +\
#                Ji*(Aelmt11+Aelmt11.T)/2
#        Mmu[i,i_order:i_order+Np,i_order:i_order+Np] = -Ji*(Aelmt11mu+Aelmt11mu.T)/2
#        # A12 and A21 are symmetric about diagonal
#        Aelmt12 = -2*ka_i*k*M -ka_i*k*U_phi_prime
#        Aelmt12mu = -5/3*mu_i*k*M + k*mu_i*Uprime_phi + 2/3*mu_i*k*U_phi_prime
#        # manage unit: *1e15/1e15
#        A12[i_order:i_order+Np,i_order:i_order+Np] = A12[i_order:i_order+Np,i_order:i_order+Np] +\
#                Ji*Aelmt12
#        
#        Aelmt21 = -2*ka_i*k*M - ka_i*k*Uprime_phi
#        Aelmt21mu = -5/3*mu_i*k*M + k*mu_i*U_phi_prime + 2/3*mu_i*k*Uprime_phi
#        # manage unit: *1e15/1e15
#        A21[i_order:i_order+Np,i_order:i_order+Np] = A21[i_order:i_order+Np,i_order:i_order+Np] +\
#                Ji*Aelmt21
#        Aelmt12mu_ave = (Aelmt12mu+Aelmt21mu.T)/2
#        Mmu[i,i_order:i_order+Np,dimension+i_order:dimension+i_order+Np] = -Ji*Aelmt12mu_ave
#        Mmu[i,dimension+i_order:dimension+i_order+Np,i_order:i_order+Np] = -Ji*Aelmt12mu_ave.T
#        
#        Aelmt22 = k**2*ka_i*M
#        Aelmt22mu = (k**2*4/3*mu_i-mu_i)*M - mu_i*Uprime_phi - mu_i*U_phi_prime + mu_i*Uprime_phi_prime
#        # manage unit: *1e15/1e15
#        A22[i_order:i_order+Np,i_order:i_order+Np] = A22[i_order:i_order+Np,i_order:i_order+Np] +\
#                Ji*(Aelmt22+Aelmt22.T)/2
#        Mmu[i,dimension+i_order:dimension+i_order+Np,dimension+i_order:dimension+i_order+Np] = \
#                -Ji*(Aelmt22mu+Aelmt22mu.T)/2
#                
#        Belmt11 = rho_i*ri@M@ri
#        # manage unit: *1e21/1e15
#        B11[i_order:i_order+Np,i_order:i_order+Np] = B11[i_order:i_order+Np,i_order:i_order+Np] +\
#                Ji*(Belmt11+Belmt11.T)/2*1e6#average to improve precision
#    
#    #calculate average of A12 to avoid small error caused by round off
#    A12_average = (A12+A21.T)/2
#    # Combine A11,A12,A21,A22 to get A. Combine B11 with 0(B12) to get B
#    A = np.vstack((np.hstack((A11,A12_average)),\
#                   np.hstack((A12_average.T,A22))))
#    B = np.vstack((np.hstack((B11,B12)),\
#                   np.hstack((B12,B11))))
#    block_len = [dimension,dimension]
#    
#    return -A,Mmu,B,Ki,dimension,block_len
#
#def old_solid_G_an(model,invV,order,Dr,rho,radius):
#    # implement finite element method on solid speroidal modes with gravity field
#    Np = order+1
#    #mu = model.mu
#    x = model.x
#    k = model.k
#    #ka = model.ka
#    Ki = len(x[0])
#    dimension = Ki*order+1
#    
#    A11=np.zeros((dimension,dimension))
#    A12=np.zeros((dimension,dimension))
#    A21=np.zeros((dimension,dimension))
#    A22=np.zeros((dimension,dimension))
#
#    B11=np.zeros((dimension,dimension))
#    B12=np.zeros((dimension,dimension)) #B12 = 0, just add terms
#    
#    Mmu = np.zeros((Ki,2*dimension,2*dimension))
#    
#    M = invV.T@invV
#    
#    for i in range(Ki):
#        Ji = model.J[0,i] #This is a number
#        rxi = model.rx[0,i] #This is a number
#        mu_i = model.mu[i] #This is a number
#        rho_i = model.rho[i]
#        ka_i = model.ka[i]
#        ri = np.diag(x[:,i]) #2*2 matrix
#        g = np.diag(lib.gravfield_lst(x[:,i],rho,radius))
#        
#        Uprime_phi = ri@M@Dr*rxi
#        U_phi_prime = rxi*Dr.T@M@ri
#        Uprime_phi_prime = rxi**2*Dr.T@ri@M@ri@Dr
#        
#        i_order = i*order
#        Aelmt11 = 2*ka_i*Uprime_phi + 4*ka_i*M +\
#                    ka_i*Uprime_phi_prime + 2*ka_i*U_phi_prime +\
#                    4*np.pi*G*np.power(rho_i,2)*ri@M@ri - 4*rho_i*M@ri@g
#        Aelmt11mu = -4/3*mu_i*Uprime_phi + (4/3*mu_i+mu_i*k**2)*M +\
#                    4/3*mu_i*Uprime_phi_prime - 4/3*mu_i*U_phi_prime
#        # manage unit: *1e15/1e15
#        A11[i_order:i_order+Np,i_order:i_order+Np] = A11[i_order:i_order+Np,i_order:i_order+Np] +\
#                Ji*(Aelmt11+Aelmt11.T)/2
#        Mmu[i,i_order:i_order+Np,i_order:i_order+Np] = -Ji*(Aelmt11mu+Aelmt11mu.T)/2
#        # A12 and A21 are symmetric about diagonal
#        Aelmt12 = -2*ka_i*k*M - ka_i*k*U_phi_prime +\
#                    rho_i*k*g@M@ri
#        Aelmt12mu = -5/3*mu_i*k*M + k*mu_i*Uprime_phi + 2/3*mu_i*k*U_phi_prime
#        # manage unit: *1e15/1e15
#        A12[i_order:i_order+Np,i_order:i_order+Np] = A12[i_order:i_order+Np,i_order:i_order+Np] +\
#                Ji*Aelmt12
#        
#        Aelmt21 = -2*ka_i*k*M - ka_i*k*Uprime_phi +\
#                    rho_i*k*ri@M@g
#        Aelmt21mu = -5/3*mu_i*k*M + k*mu_i*U_phi_prime + 2/3*mu_i*k*Uprime_phi
#        # manage unit: *1e15/1e15
#        A21[i_order:i_order+Np,i_order:i_order+Np] = A21[i_order:i_order+Np,i_order:i_order+Np] +\
#                Ji*Aelmt21
#        Aelmt12mu_ave = (Aelmt12mu+Aelmt21mu.T)/2
#        Mmu[i,i_order:i_order+Np,dimension+i_order:dimension+i_order+Np] = -Ji*Aelmt12mu_ave
#        Mmu[i,dimension+i_order:dimension+i_order+Np,i_order:i_order+Np] = -Ji*Aelmt12mu_ave.T    
#        
#        Aelmt22=k**2*ka_i*M
#        Aelmt22mu = (k**2*4/3*mu_i-mu_i)*M - mu_i*Uprime_phi - mu_i*U_phi_prime + mu_i*Uprime_phi_prime
#        # manage unit: *1e15/1e15
#        A22[i_order:i_order+Np,i_order:i_order+Np] = A22[i_order:i_order+Np,i_order:i_order+Np] +\
#                Ji*(Aelmt22+Aelmt22.T)/2
#        Mmu[i,dimension+i_order:dimension+i_order+Np,dimension+i_order:dimension+i_order+Np] = \
#                -Ji*(Aelmt22mu+Aelmt22mu.T)/2
#        
#                
#        Belmt11 = rho_i*ri@M@ri
#        # manage unit: *1e21/1e15
#        B11[i_order:i_order+Np,i_order:i_order+Np] = B11[i_order:i_order+Np,i_order:i_order+Np] +\
#                Ji*(Belmt11+Belmt11.T)/2*1e6 #average to improve precision
#    
#    #calculate average of A12 to avoid small error caused by round off
#    A12_average = (A12+A21.T)/2
#    # Combine A11,A12,A21,A22 to get A. Combine B11 with 0(B12) to get B
#    A = np.vstack((np.hstack((A11,A12_average)),np.hstack((A12_average.T,A22))))
#    B = np.vstack((np.hstack((B11,B12)),np.hstack((B12,B11))))
#    block_len = [dimension,dimension]
#    
#    return -A,Mmu,B,Ki,block_len
#
#def old_spheroidal_solid_noG(model, invV, order, Dr, anelastic = False):
#    '''
#    Create A and B matrices.
#    For the spheroidal modes in a solid region.
#    Does not include gravity.
#
#    For definitions of variables, see modes/compute_modes.py
#    '''
#    
#    g_switch = 0
#
#    # Get coordinates, asymptotic wavenumber, number of points per element
#    # number of elements, size of matrix problem, local mass matrix,
#    # and initialised output arrays.
#    if anelastic:
#
#        (x, k, Np, Ki, dimension, M, B11, B12, A11_ka, A12_ka, A21_ka, A22_ka,
#                A11_mu, A22_mu, A12_mu_ave) = \
#                    prepare_block_spheroidal_noG_or_G(model, order, invV,
#                                                anelastic = True)
#
#    else:
#
#        x, k, Np, Ki, dimension, M, B11, B12, A11, A12, A21, A22 = \
#                prepare_block_spheroidal_noG_or_G(model, order, invV,
#                                            anelastic = False)
#    
#    # Loop over elements.
#    for i in range(Ki):
#        
#        # For this element, get Jacobian, metric, material parameters,
#        # coordinates, test-function terms and indices of matrix entries.
#        (Ji, ri, _, rxi, mu_i, ka_i, rho_i, Uprime_phi, U_phi_prime,
#            Uprime_phi_prime, i0, i1, _, _, _) = prepare_element_spheroidal(
#                                            model, x, M, Dr, order, Np, i,
#                                            g_switch, None)
#
#        # Calculate matrix elements.
#        (   Aelmt11_mu, Aelmt11_ka, Aelmt12_mu, Aelmt12_ka, Aelmt21_mu,
#            Aelmt21_ka, Aelmt22_mu, Aelmt22_ka, Belmt11) = \
#                get_spheroidal_matrix_elements(k, ri, ka_i, mu_i,
#                        rho_i, Uprime_phi, U_phi_prime, Uprime_phi_prime, M,
#                        g_switch, g = None)
#
#        # Store the matrix elements in the full matrices.
#        if anelastic:
#
#            A11_mu, A22_mu, A12_mu_ave, A11_ka, A12_ka, A21_ka, A22_ka, B11 = \
#                store_spheroidal_matrix_elements_anelastic(
#                        A11_mu, A22_mu, A12_mu_ave,
#                        A11_ka, A12_ka, A21_ka, A22_ka, B11,
#                        Aelmt11_mu, Aelmt12_mu, Aelmt21_mu, Aelmt22_mu,
#                        Aelmt11_ka, Aelmt12_ka, Aelmt21_ka, Aelmt22_ka,
#                        Belmt11, Ji, i, i0, i1)
#
#        else:
#
#            A11, A12, A21, A22, B11 = \
#                store_spheroidal_matrix_elements_elastic(
#                        A11, A12, A21, A22, B11,
#                        Aelmt11_mu, Aelmt12_mu, Aelmt21_mu, Aelmt22_mu,
#                        Aelmt11_ka, Aelmt12_ka, Aelmt21_ka, Aelmt22_ka,
#                        Belmt11, Ji, i0, i1)
#
#    # Record the size of this block.
#    block_len = [dimension, dimension]
#
#    # Combine matrix blocks.
#    if anelastic:
#
#        A_ka, A_mu, B = combine_spheroidal_matrix_blocks_anelastic(
#                            A11_ka, A12_ka, A21_ka, A22_ka,
#                            A11_mu, A12_mu_ave, A22_mu,
#                            B11, B12, Ki)
#
#        # Note negative sign on A matrices due to different convention
#        # for defining eigenvalue problem in anelastic case.
#        return -A_ka, -A_mu, B, Ki, dimension, block_len
#
#    else:
#
#        A, B = combine_spheroidal_matrix_blocks_elastic(A11, A12, A21, A22,
#                B11, B12)
#
#        return A, B, block_len
#    
#    return
