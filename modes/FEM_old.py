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

def solid_GPmixed(model,invV,invV_P,order,order_P,Dr,Dr_P,rho,radius):
    '''
    Create A and B matrices.
    For the spheroidal modes in a solid region.
    Includes gravity and Eulerian perturbation.

    For definitions of variables, see modes/compute_modes.py
    '''
    # implement finite element method on solid speroidal modes with gravity field and Euler perturbation
    Np = order+1
    Np_P = order_P+1
    #mu = model.mu
    x = model.x
    k = model.k
    x_P = model.xP
    #ka = model.ka
    Ki = len(x[0])
    dimension = Ki*order+1
    dimension_P = Ki*order_P+1
    
    A11=np.zeros((dimension,dimension))
    A12=np.zeros((dimension,dimension))
    A13=np.zeros((dimension,dimension_P))
    A21=np.zeros((dimension,dimension))
    A22=np.zeros((dimension,dimension))
    A23=np.zeros((dimension,dimension_P))
    A31=np.zeros((dimension_P,dimension))
    A32=np.zeros((dimension_P,dimension))
    A33=np.zeros((dimension_P,dimension_P))

    B11=np.zeros((dimension,dimension))
    AB0=np.zeros((dimension,dimension))
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
        # A12 and A21 are symmetric about diagonal
        Aelmt12 = (-2*ka_i-5/3*mu_i)*k*M + k*mu_i*Uprime_phi+ -(ka_i-2/3*mu_i)*k*U_phi_prime +\
                    rho_i*k*g@M@ri
        # manage unit: *1e15/1e15
        A12[i_order:i_order+Np,i_order:i_order+Np] = A12[i_order:i_order+Np,i_order:i_order+Np] +\
                Ji*Aelmt12
        
        Aelmt13 = rho_i*ri@ri@MuP@Dr_P*rxi
        # manage unit: *1e15/1e15
        A13[i_order:i_order+Np,i_order_P:i_order_P+Np_P] = A13[i_order:i_order+Np,i_order_P:i_order_P+Np_P] +\
                Ji*Aelmt13
        
        Aelmt21 = (-2*ka_i-5/3*mu_i)*k*M + k*mu_i*U_phi_prime - (ka_i-2/3*mu_i)*k*Uprime_phi +\
                    rho_i*k*ri@M@g
        # manage unit: *1e15/1e15
        A21[i_order:i_order+Np,i_order:i_order+Np] = A21[i_order:i_order+Np,i_order:i_order+Np] +\
                Ji*Aelmt21
        
        Aelmt22=(k**2*(ka_i+4/3*mu_i)-mu_i)*M - mu_i*Uprime_phi - mu_i*U_phi_prime + mu_i*Uprime_phi_prime
        # manage unit: *1e15/1e15
        A22[i_order:i_order+Np,i_order:i_order+Np] = A22[i_order:i_order+Np,i_order:i_order+Np] +\
                Ji*(Aelmt22+Aelmt22.T)/2
                
        Aelmt23 = k*rho_i*ri@MuP
        # manage unit: *1e15/1e15
        A23[i_order:i_order+Np,i_order_P:i_order_P+Np_P] = A23[i_order:i_order+Np,i_order_P:i_order_P+Np_P] +\
                Ji*Aelmt23
                
        Aelmt31 = rho_i*rxi*Dr_P.T@MuP.T@ri@ri;
        # manage unit: *1e15/1e15
        A31[i_order_P:i_order_P+Np_P,i_order:i_order+Np] = A31[i_order_P:i_order_P+Np_P,i_order:i_order+Np] +\
                Ji*Aelmt31
        
        Aelmt32 = k*rho_i*MuP.T@ri
        # manage unit: *1e15/1e15
        A32[i_order_P:i_order_P+Np_P,i_order:i_order+Np] = A32[i_order_P:i_order_P+Np_P,i_order:i_order+Np] +\
                Ji*Aelmt32
        
        Aelmt33 = 1/(4*np.pi*G)*(rxi*Dr_P.T@ri_P@MP@ri_P@Dr_P*rxi+k**2*MP)
        # manage unit: *1e15/1e15
        A33[i_order_P:i_order_P+Np_P,i_order_P:i_order_P+Np_P] = A33[i_order_P:i_order_P+Np_P,i_order_P:i_order_P+Np_P] +\
                Ji*(Aelmt33+Aelmt33.T)/2
        
        Belmt11 = rho_i*ri@M@ri
        # manage unit: *1e21/1e15
        B11[i_order:i_order+Np,i_order:i_order+Np] = B11[i_order:i_order+Np,i_order:i_order+Np] +\
                Ji*(Belmt11+Belmt11.T)/2*1e6 #average to improve precision
    
    #calculate average of A12 to avoid small error caused by round off
    A12_average = (A12+A21.T)/2
    A13_average = (A13+A31.T)/2
    A23_average = (A23+A32.T)/2
    # Combine A11,A12,A21,A22 to get A. Combine B11 with 0(B12) to get B
    A = np.vstack((np.hstack((A11,A12_average,A13_average)),\
                   np.hstack((A12_average.T,A22,A23_average)),\
                   np.hstack((A13_average.T,A23_average.T,A33))))
    B = np.vstack((np.hstack((B11,AB0,ABuP)),\
                   np.hstack((AB0,B11,ABuP)),\
                   np.hstack((ABuP.T,ABuP.T,B33))))
    block_len = [dimension,dimension,dimension_P]
    
    return A,B,block_len

def solid_G(model,invV,order,Dr,rho,radius):
    '''
    Create A and B matrices.
    For the spheroidal modes in a solid region.
    Includes gravity but no Eulerian perturbation.

    For definitions of variables, see modes/compute_modes.py
    '''

    # implement finite element method on solid speroidal modes with gravity field
    Np = order+1
    #mu = model.mu
    x = model.x
    k = model.k
    #ka = model.ka
    Ki = len(x[0])
    dimension = Ki*order+1
    
    A11=np.zeros((dimension,dimension))
    A12=np.zeros((dimension,dimension))
    A21=np.zeros((dimension,dimension))
    A22=np.zeros((dimension,dimension))

    B11=np.zeros((dimension,dimension))
    B12=np.zeros((dimension,dimension)) #B12 = 0, just add terms
    
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
        # A12 and A21 are symmetric about diagonal
        Aelmt12 = (-2*ka_i-5/3*mu_i)*k*M + k*mu_i*Uprime_phi+ -(ka_i-2/3*mu_i)*k*U_phi_prime +\
                    rho_i*k*g@M@ri
        # manage unit: *1e15/1e15
        A12[i_order:i_order+Np,i_order:i_order+Np] = A12[i_order:i_order+Np,i_order:i_order+Np] +\
                Ji*Aelmt12
        
        Aelmt21 = (-2*ka_i-5/3*mu_i)*k*M + k*mu_i*U_phi_prime - (ka_i-2/3*mu_i)*k*Uprime_phi +\
                    rho_i*k*ri@M@g
        # manage unit: *1e15/1e15
        A21[i_order:i_order+Np,i_order:i_order+Np] = A21[i_order:i_order+Np,i_order:i_order+Np] +\
                Ji*Aelmt21
        
        Aelmt22=(k**2*(ka_i+4/3*mu_i)-mu_i)*M - mu_i*Uprime_phi - mu_i*U_phi_prime + mu_i*Uprime_phi_prime
        # manage unit: *1e15/1e15
        A22[i_order:i_order+Np,i_order:i_order+Np] = A22[i_order:i_order+Np,i_order:i_order+Np] +\
                Ji*(Aelmt22+Aelmt22.T)/2
                
        Belmt11 = rho_i*ri@M@ri
        # manage unit: *1e21/1e15
        B11[i_order:i_order+Np,i_order:i_order+Np] = B11[i_order:i_order+Np,i_order:i_order+Np] +\
                Ji*(Belmt11+Belmt11.T)/2*1e6 #average to improve precision
    
    #calculate average of A12 to avoid small error caused by round off
    A12_average = (A12+A21.T)/2
    # Combine A11,A12,A21,A22 to get A. Combine B11 with 0(B12) to get B
    A = np.vstack((np.hstack((A11,A12_average)),np.hstack((A12_average.T,A22))))
    B = np.vstack((np.hstack((B11,B12)),np.hstack((B12,B11))))
    block_len = [dimension,dimension]
    
    return A,B,block_len

def solid_noG(model,invV,order,Dr):
    '''
    Create A and B matrices.
    For the spheroidal modes in a solid region.
    Does not include gravity.

    For definitions of variables, see modes/compute_modes.py
    '''

    # replace np.matmul with @
    # implement finite element method on solid speroidal modes
    Np = order+1
    #mu = model.mu
    #rho = model.rho
    x = model.x
    k = model.k
    #ka = model.ka
    Ki = len(x[0])
    dimension = Ki*order+1
    
    A11=np.zeros((dimension,dimension))
    A12=np.zeros((dimension,dimension))
    A21=np.zeros((dimension,dimension))
    A22=np.zeros((dimension,dimension))

    B11=np.zeros((dimension,dimension))
    B12=np.zeros((dimension,dimension)) #B12 = 0, just add terms
    
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
        # A12 and A21 are symmetric about diagonal
        Aelmt12 = (-2*ka_i-5/3*mu_i)*k*M + k*mu_i*Uprime_phi+ -(ka_i-2/3*mu_i)*k*U_phi_prime
        # manage unit: *1e15/1e15
        A12[i_order:i_order+Np,i_order:i_order+Np] = A12[i_order:i_order+Np,i_order:i_order+Np] +\
                Ji*Aelmt12
        
        Aelmt21 = (-2*ka_i-5/3*mu_i)*k*M + k*mu_i*U_phi_prime - (ka_i-2/3*mu_i)*k*Uprime_phi
        # manage unit: *1e15/1e15
        A21[i_order:i_order+Np,i_order:i_order+Np] = A21[i_order:i_order+Np,i_order:i_order+Np] +\
                Ji*Aelmt21
        
        Aelmt22=(k**2*(ka_i+4/3*mu_i)-mu_i)*M - mu_i*Uprime_phi - mu_i*U_phi_prime + mu_i*Uprime_phi_prime
        # manage unit: *1e15/1e15
        A22[i_order:i_order+Np,i_order:i_order+Np] = A22[i_order:i_order+Np,i_order:i_order+Np] +\
                Ji*(Aelmt22+Aelmt22.T)/2
                
        Belmt11 = rho_i*ri@M@ri
        # manage unit: *1e21/1e15
        B11[i_order:i_order+Np,i_order:i_order+Np] = B11[i_order:i_order+Np,i_order:i_order+Np] +\
                Ji*(Belmt11+Belmt11.T)/2*1e6#average to improve precision
    
    #calculate average of A12 to avoid small error caused by round off
    A12_average = (A12+A21.T)/2
    # Combine A11,A12,A21,A22 to get A. Combine B11 with 0(B12) to get B
    A = np.vstack((np.hstack((A11,A12_average)),\
                   np.hstack((A12_average.T,A22))))
    B = np.vstack((np.hstack((B11,B12)),\
                   np.hstack((B12,B11))))
    block_len = [dimension,dimension]
    
    return A,B,block_len

def fluid_GP_mixedPV(model,invV,invV_p,invV_P,invV_V,order,order_p,order_P,order_V,Dr,Dr_p,Dr_P,Dr_V,rho,radius):
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
    
    return A,B,block_len

def fluid_G_mixedV(model,invV,invV_p,invV_V,order,order_p,order_V,Dr,Dr_p,Dr_V,rho,radius):
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
    
    return A,B,block_len

def fluid_noG_mixedV(model,invV,invV_p,invV_V,order,order_p,order_V,Dr,Dr_p,Dr_V):
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
    
    return A,B,block_len

def toroidal(model,invV,order,Dr):
    '''
    Create A and B matrices.
    For the toroidal modes (necessarily in solid region).

    For definitions of variables, see modes/compute_modes.py
    '''

    # implement finite element method on toroidal modes
    Np = order+1
    mu = model.mu
    rho = model.rho
    x = model.x
    k = model.k
    Ki = len(x[0]) #number of elements
    dimension = Ki*order+1
    
    A = np.zeros((dimension,dimension)) #integraral of test function matrix
    B = np.zeros((dimension,dimension)) #Xi matrix
    
    M = np.matmul(invV.T,invV)
    
    for i in range(Ki):
        Ji = model.J[0,i] #This is a number
        rxi = model.rx[0,i] #This is a number
        ri = np.diag(x[:,i]) #2*2 matrix
        # Dr: 2*2 matrix, mu[i]: number
        # Aelmt: 4 terms of the RHS of the weak form equation
        # Belmt: 1 term of the LHS of the weak form equation
        # must check if A and B are symmetric
        #ri is still different
        i_order = i*order
        Aelmt = Ji*mu[i]*(-(rxi * ((Dr.T @ M) @ ri) +   \
                     (ri @ (M @ Dr)) * rxi) +           \
                     (k**2.0 - 1.0)*M +                 \
                     (rxi**2.0) * ((((Dr.T @ ri) @ M) @ ri) @ Dr))
        # manage unit: *1e15/1e15
        A[i_order:i_order+Np,i_order:i_order+Np] = A[i_order:i_order+Np,i_order:i_order+Np] +\
                    (Aelmt+Aelmt.T)/2 #Symmetric operation to eliminate little error
        
        Belmt = Ji*rho[i]*np.matmul(np.matmul(ri,M),ri)
        # manage unit: *1e21/1e15
        B[i_order:i_order+Np,i_order:i_order+Np] = B[i_order:i_order+Np,i_order:i_order+Np] +\
                    (Belmt+Belmt.T)/2*1e6 #Symmetric operation to eliminate little error
    
    return A,B

# Anelastic versions. ---------------------------------------------------------
def toroidal_an(model,invV,order,Dr):
    # implement finite element method on toroidal modes
    Np = order+1
    mu = model.mu
    rho = model.rho
    x = model.x
    k = model.k
    Ki = len(x[0]) #number of elements
    dimension = Ki*order+1
    
    #A = np.zeros((dimension,dimension)) #integraral of test function matrix
    B = np.zeros((dimension,dimension)) #Xi matrix
    Mmu = np.zeros((Ki,dimension,dimension)) #Matrix for each layer
    M = np.matmul(invV.T,invV)

    for i in range(Ki):
        Ji = model.J[0,i] #This is a number
        rxi = model.rx[0,i] #This is a number
        ri = np.diag(x[:,i]) #2*2 matrix
        # Dr: 2*2 matrix, mu[i]: number
        # Aelmt: 4 terms of the RHS of the weak form equation
        # Belmt: 1 term of the LHS of the weak form equation
        # must check if A and B are symmetric
        #ri is still different
        i_order = i*order
        Aelmt = Ji*mu[i]*(-(rxi*((Dr.T @ M) @ ri)+\
                     (ri @ (M @ Dr))*rxi)+\
                    (k**2-1)*M + \
                    rxi**2*((((Dr.T @ ri) @ M) @ ri) @ Dr))
        # manage unit: *1e15/1e15, change + to -
        Mmu[i,i_order:i_order+Np,i_order:i_order+Np]  = -(Aelmt+Aelmt.T)/2
#        A[i_order:i_order+Np,i_order:i_order+Np] = A[i_order:i_order+Np,i_order:i_order+Np] -\
#                    (Aelmt+Aelmt.T)/2 #Symmetric operation to eliminate little error
        
        Belmt = Ji * rho[i] * ((ri @ M) @ ri)

        # manage unit: *1e21/1e15
        B[i_order:i_order+Np,i_order:i_order+Np] = B[i_order:i_order+Np,i_order:i_order+Np] +\
                    (Belmt+Belmt.T)/2*1e6 #Symmetric operation to eliminate little error

    return Mmu, B, Ki, dimension

def fluid_noG_mixedV_an(model,invV,invV_p,invV_V,order,order_p,order_V,Dr,Dr_p,Dr_V):
    # implement finite element method on fluid speroidal modes
    Np = order+1
    Np_p = order_p+1
    Np_V = order_V+1
    x = model.x
    x_p=model.xp
    x_V=model.xV
    k = model.k
    #ka = model.ka
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
        ka_i = model.ka[i]
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
    
    return -A,B,Ki,block_len

def solid_noG_an(model,invV,order,Dr):
    # replace np.matmul with @
    # implement finite element method on solid speroidal modes
    Np = order+1
    #mu = model.mu
    #rho = model.rho
    x = model.x
    k = model.k
    #ka = model.ka
    Ki = len(x[0])
    dimension = Ki*order+1
    
    A11=np.zeros((dimension,dimension))
    A12=np.zeros((dimension,dimension))
    A21=np.zeros((dimension,dimension))
    A22=np.zeros((dimension,dimension))

    B11=np.zeros((dimension,dimension))
    B12=np.zeros((dimension,dimension)) #B12 = 0, just add terms
    
    Mmu = np.zeros((Ki,2*dimension,2*dimension))
    
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
        Aelmt11 = 2*ka_i*Uprime_phi + 4*ka_i*M +\
                    ka_i*Uprime_phi_prime + 2*ka_i*U_phi_prime
        Aelmt11mu = 2*-2/3*mu_i*Uprime_phi + (4/3*mu_i+mu_i*k**2)*M +\
                    4/3*mu_i*Uprime_phi_prime + 2*-2/3*mu_i*U_phi_prime
        # manage unit: *1e15/1e15
        A11[i_order:i_order+Np,i_order:i_order+Np] = A11[i_order:i_order+Np,i_order:i_order+Np] +\
                Ji*(Aelmt11+Aelmt11.T)/2
        Mmu[i,i_order:i_order+Np,i_order:i_order+Np] = -Ji*(Aelmt11mu+Aelmt11mu.T)/2
        # A12 and A21 are symmetric about diagonal
        Aelmt12 = -2*ka_i*k*M -ka_i*k*U_phi_prime
        Aelmt12mu = -5/3*mu_i*k*M + k*mu_i*Uprime_phi + 2/3*mu_i*k*U_phi_prime
        # manage unit: *1e15/1e15
        A12[i_order:i_order+Np,i_order:i_order+Np] = A12[i_order:i_order+Np,i_order:i_order+Np] +\
                Ji*Aelmt12
        
        Aelmt21 = -2*ka_i*k*M - ka_i*k*Uprime_phi
        Aelmt21mu = -5/3*mu_i*k*M + k*mu_i*U_phi_prime + 2/3*mu_i*k*Uprime_phi
        # manage unit: *1e15/1e15
        A21[i_order:i_order+Np,i_order:i_order+Np] = A21[i_order:i_order+Np,i_order:i_order+Np] +\
                Ji*Aelmt21
        Aelmt12mu_ave = (Aelmt12mu+Aelmt21mu.T)/2
        Mmu[i,i_order:i_order+Np,dimension+i_order:dimension+i_order+Np] = -Ji*Aelmt12mu_ave
        Mmu[i,dimension+i_order:dimension+i_order+Np,i_order:i_order+Np] = -Ji*Aelmt12mu_ave.T
        
        Aelmt22 = k**2*ka_i*M
        Aelmt22mu = (k**2*4/3*mu_i-mu_i)*M - mu_i*Uprime_phi - mu_i*U_phi_prime + mu_i*Uprime_phi_prime
        # manage unit: *1e15/1e15
        A22[i_order:i_order+Np,i_order:i_order+Np] = A22[i_order:i_order+Np,i_order:i_order+Np] +\
                Ji*(Aelmt22+Aelmt22.T)/2
        Mmu[i,dimension+i_order:dimension+i_order+Np,dimension+i_order:dimension+i_order+Np] = \
                -Ji*(Aelmt22mu+Aelmt22mu.T)/2
                
        Belmt11 = rho_i*ri@M@ri
        # manage unit: *1e21/1e15
        B11[i_order:i_order+Np,i_order:i_order+Np] = B11[i_order:i_order+Np,i_order:i_order+Np] +\
                Ji*(Belmt11+Belmt11.T)/2*1e6#average to improve precision
    
    #calculate average of A12 to avoid small error caused by round off
    A12_average = (A12+A21.T)/2
    # Combine A11,A12,A21,A22 to get A. Combine B11 with 0(B12) to get B
    A = np.vstack((np.hstack((A11,A12_average)),\
                   np.hstack((A12_average.T,A22))))
    B = np.vstack((np.hstack((B11,B12)),\
                   np.hstack((B12,B11))))
    block_len = [dimension,dimension]
    
    return -A,Mmu,B,Ki,dimension,block_len

def solid_G_an(model,invV,order,Dr,rho,radius):
    # implement finite element method on solid speroidal modes with gravity field
    Np = order+1
    #mu = model.mu
    x = model.x
    k = model.k
    #ka = model.ka
    Ki = len(x[0])
    dimension = Ki*order+1
    
    A11=np.zeros((dimension,dimension))
    A12=np.zeros((dimension,dimension))
    A21=np.zeros((dimension,dimension))
    A22=np.zeros((dimension,dimension))

    B11=np.zeros((dimension,dimension))
    B12=np.zeros((dimension,dimension)) #B12 = 0, just add terms
    
    Mmu = np.zeros((Ki,2*dimension,2*dimension))
    
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
        Aelmt11 = 2*ka_i*Uprime_phi + 4*ka_i*M +\
                    ka_i*Uprime_phi_prime + 2*ka_i*U_phi_prime +\
                    4*np.pi*G*np.power(rho_i,2)*ri@M@ri - 4*rho_i*M@ri@g
        Aelmt11mu = -4/3*mu_i*Uprime_phi + (4/3*mu_i+mu_i*k**2)*M +\
                    4/3*mu_i*Uprime_phi_prime - 4/3*mu_i*U_phi_prime
        # manage unit: *1e15/1e15
        A11[i_order:i_order+Np,i_order:i_order+Np] = A11[i_order:i_order+Np,i_order:i_order+Np] +\
                Ji*(Aelmt11+Aelmt11.T)/2
        Mmu[i,i_order:i_order+Np,i_order:i_order+Np] = -Ji*(Aelmt11mu+Aelmt11mu.T)/2
        # A12 and A21 are symmetric about diagonal
        Aelmt12 = -2*ka_i*k*M - ka_i*k*U_phi_prime +\
                    rho_i*k*g@M@ri
        Aelmt12mu = -5/3*mu_i*k*M + k*mu_i*Uprime_phi + 2/3*mu_i*k*U_phi_prime
        # manage unit: *1e15/1e15
        A12[i_order:i_order+Np,i_order:i_order+Np] = A12[i_order:i_order+Np,i_order:i_order+Np] +\
                Ji*Aelmt12
        
        Aelmt21 = -2*ka_i*k*M - ka_i*k*Uprime_phi +\
                    rho_i*k*ri@M@g
        Aelmt21mu = -5/3*mu_i*k*M + k*mu_i*U_phi_prime + 2/3*mu_i*k*Uprime_phi
        # manage unit: *1e15/1e15
        A21[i_order:i_order+Np,i_order:i_order+Np] = A21[i_order:i_order+Np,i_order:i_order+Np] +\
                Ji*Aelmt21
        Aelmt12mu_ave = (Aelmt12mu+Aelmt21mu.T)/2
        Mmu[i,i_order:i_order+Np,dimension+i_order:dimension+i_order+Np] = -Ji*Aelmt12mu_ave
        Mmu[i,dimension+i_order:dimension+i_order+Np,i_order:i_order+Np] = -Ji*Aelmt12mu_ave.T    
        
        Aelmt22=k**2*ka_i*M
        Aelmt22mu = (k**2*4/3*mu_i-mu_i)*M - mu_i*Uprime_phi - mu_i*U_phi_prime + mu_i*Uprime_phi_prime
        # manage unit: *1e15/1e15
        A22[i_order:i_order+Np,i_order:i_order+Np] = A22[i_order:i_order+Np,i_order:i_order+Np] +\
                Ji*(Aelmt22+Aelmt22.T)/2
        Mmu[i,dimension+i_order:dimension+i_order+Np,dimension+i_order:dimension+i_order+Np] = \
                -Ji*(Aelmt22mu+Aelmt22mu.T)/2
        
                
        Belmt11 = rho_i*ri@M@ri
        # manage unit: *1e21/1e15
        B11[i_order:i_order+Np,i_order:i_order+Np] = B11[i_order:i_order+Np,i_order:i_order+Np] +\
                Ji*(Belmt11+Belmt11.T)/2*1e6 #average to improve precision
    
    #calculate average of A12 to avoid small error caused by round off
    A12_average = (A12+A21.T)/2
    # Combine A11,A12,A21,A22 to get A. Combine B11 with 0(B12) to get B
    A = np.vstack((np.hstack((A11,A12_average)),np.hstack((A12_average.T,A22))))
    B = np.vstack((np.hstack((B11,B12)),np.hstack((B12,B11))))
    block_len = [dimension,dimension]
    
    return -A,Mmu,B,Ki,block_len

def fluid_G_mixedV_an(model,invV,invV_p,invV_V,order,order_p,order_V,Dr,Dr_p,Dr_V,rho,radius):
    # implement finite element method on fluid speroidal modes
    Np = order+1
    Np_p = order_p+1
    Np_V = order_V+1
    x = model.x
    x_p=model.xp
    x_V=model.xV
    k = model.k
    #ka = model.ka
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
        ka_i = model.ka[i]
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
    
    return -A,B,Ki,block_len
