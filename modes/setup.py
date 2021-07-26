'''
Various functions related to the initialisation of the model and the finite elements.
The finite-element functions are mostly based on the theory and Matlab examples described in
Hesthaven, Warburton (2008) 'Nodal discontinuous Galerkin methods: Algoriths, analysis and applications' Springer Texts in Applied Mathematics 54.
'''

import numpy as np
from numpy.linalg import inv
from scipy.special import gamma

def Dmatrix1D(N,r,V):
    
    # Dmatrix1D(N,r,V)
    # Purpose : Initialize the (r) differentiation matrices on the interval,
    #	        evaluated at (r) at order N
    
    Vr = GradVandermonde1D(N,r)
    # Solve equation Dr*V=Vr by solving V.T*Dr.T=Vr.T using numpy.linalg.lstsq
    # In matlab this is only Dr=Vr/V
    solution = np.linalg.lstsq(V.T,Vr.T,rcond=-1)
    Dr = solution[0].T #Get to transpose of solution
    
    return Dr

def GeometricFactors1D(x,Dr):
    
    # GeometricFactors1D(x,Dr)
    # Purpose  : Compute the metric elements for the local mappings of the 1D elements
    
    J = np.dot(Dr,x)
    #print (J)
    rx = 1/J
    
    return rx,J

def GradJacobiP(r,alpha,beta,N):
    
    # GradJacobiP(r, alpha, beta, N);
    # Purpose : Evaluate the derivative of the Jacobi polynomial of type (alpha,beta)>-1,
    #           at points r for order N and returns dP[1:length(r))]
    
    if N == 0:
        dP = np.zeros(np.shape(r))
    else:
        #print (JacobiP(r,alpha+1,beta+1,N-1))
        #print (np.sqrt(N*(N+alpha+beta+1)))
        dP = np.sqrt(N*(N+alpha+beta+1)) * JacobiP(r,alpha+1,beta+1,N-1)
    
    return dP

def GradVandermonde1D(N,r):
    # GradVandermonde1D(N,r)
    # Purpose : Initialize the gradient of the modal basis (i) at (r) at order N
    
    DVr = GradJacobiP(r,0,0,0)
    #print ('DVr',DVr)
    #print (GradJacobiP(r,0,0,1))
    #print (np.row_stack((DVr, GradJacobiP(r,0,0,1))))
    for i in range(1,N+1):
        DVr = np.row_stack((DVr, GradJacobiP(r,0,0,i)))# check how to merge them together
        
    return DVr.T #transpose to get the same result as in matlab

def JacobiP(x,alpha,beta,N):
    
    # function JacobiP(x,alpha,beta,N),x must be a numpy array
    # Return the coefficients of normalized Jacobi Polinomial
    # Purpose: Evaluate Jacobi Polynomial of type (alpha,beta) > -1
    #          (alpha+beta <>(inequal) -1) at points x for order N and returns P[1:length(xp))]
    # Note   : They are normalized to be orthonormal.
    
    PL = np.zeros([N+1, np.shape(x)[0]])
    
    # Initial values P_0(x) and P_1(x)
    gamma0 = 2**(alpha+beta+1)*gamma(alpha+1)*gamma(beta+1)/gamma(alpha+beta+2)
    PL[0,:] = 1/np.sqrt(gamma0)
    if N==0:
        P = PL.T
        return P[:,0] # return a one dimentional matrix
    gamma1 = (alpha+1)*(beta+1)/(alpha+beta+3)*gamma0
    #print (PL,(alpha+beta+2)*xp/2)
    PL[1,:] = ((alpha+beta+2)*x/2 + (alpha-beta)/2)/np.sqrt(gamma1)
    if N==1:
        P=PL[N,:].T
        return P
    
    # Repeat value in recurrence.
    aold = 2/(2+alpha+beta)*np.sqrt((alpha+1)*(beta+1)/(alpha+beta+3))
        
    # Forward recurrence using the symmetry of the recurrence.    
    for i in range(N-1):
        h1 = 2*(i+1)+alpha+beta
        anew = 2/(h1+2)*np.sqrt((i+2)*(i+2+alpha+beta)*(i+2+alpha)*(i+2+beta)/(h1+1)/(h1+3))
        bnew = - (alpha**2-beta**2)/h1/(h1+2)
        #print (anew,bnew,alpha,beta)
        PL[i+2,:] = 1/anew*(-aold*PL[i,:] + (x-bnew)*PL[i+1,:])
        aold = anew
    P = PL[N,:].T
    
    return P

def JacobiGL(alpha,beta,N):
    
    # JacobiGL(alpha,beta,N)
    # Purpose: Compute the N'th order Gauss Lobatto quadrature 
    # points, x, associated with the Jacobi polynomial,
    # of type (alpha,beta) > -1 ( <> -0.5). 
    
    if N==1:
        x = np.array([-1.,1.])
        return x
    xint = JacobiGQ(alpha+1,beta+1,N-2)
    x = np.append(-1,np.append(xint,1))
    
    return x # x are in different oder as in matlab

def JacobiGQ(alpha,beta,N):
    
    # JacobiGQ(alpha,beta,N)
    # Purpose: Compute the N'th order Gauss quadrature points, x, 
    # and weights, w, associated with the Jacobi 
    # polynomial, of type (alpha,beta) > -1 ( <> -0.5).
    
    if N==0:
        x = np.array([-(alpha-beta)/(alpha+beta+2)])
        #w = np.array([2])
        return x
        
    #Form symmetric matrix from recurrence.
    h1 = 2*np.arange(N+1)+alpha+beta
    J = np.diag(-1/2*(alpha**2-beta**2)/(h1+2)/h1) + \
        np.diag(2/(h1[0:N]+2)*np.sqrt(np.arange(1,N+1)*(np.arange(1,N+1)+alpha+beta) * \
               (np.arange(1,N+1)+alpha)*(np.arange(1,N+1)+beta)/(h1[0:N]+1)/(h1[0:N]+3)),1)
    if alpha+beta<10*np.finfo(float).eps:
        J[1,1]=0.0
    J = J + J.T
    
    # Compute quadrature by eigenvalue solve. x: eigenvalues, V: eigenvectors
    x,V = np.linalg.eig(J) #the eigenvalues are from large to small, different from matlab
    #print (V,J)
    #w = np.power((V[0,:].T),2)*(2**(alpha+beta+1))/(alpha+beta+1)*gamma(alpha+1)*gamma(beta+1)/gamma(alpha+beta+1)
    return np.flip(x,0)#,w # x and w are in different oder as in matlab

def StartUp(order, va, vb):
    '''
    Construct grid, operators and metric required for 1-D finite-element 
    problem.
    '''
    
    # Compute basic Legendre-Gauss-Lobatto grid of r (normalised coordinate)
    # within a single element.
    r_elmt = JacobiGL(0, 0, order)

    # Build matrix operators based on reference element.
    V  = Vandermonde1D(order, r_elmt)
    invV = inv(V)
    Dr = Dmatrix1D(order, r_elmt, V)
    
    # Build coordinates of all the nodes
    # What is this suppose to mean

    x = np.tile(va,(order+1,1)) + 0.5*(r_elmt.reshape(order+1,1)+1)*np.tile(vb,(order+1,1))
    
    # Calculate geometric factors
    rx,J = GeometricFactors1D(x,Dr)
    
    return x,J,rx,invV,Dr

def StartUp4Perturbation(order_P,va,vb):
    
    #Np_P = order_P+1

    # Compute basic Legendre Gauss Lobatto grid
    r_elmt_P = JacobiGL(0,0,order_P)

    # Build reference element matrices
    V_P  = Vandermonde1D(order_P, r_elmt_P)
    invV_P = inv(V_P)
    Dr_P = Dmatrix1D(order_P, r_elmt_P, V_P)

    # build coordinates of all the nodes
    x_P = np.tile(va,(order_P+1,1)) + 0.5*(r_elmt_P.reshape(order_P+1,1)+1)*np.tile(vb,(order_P+1,1))
    
    return invV_P,Dr_P,x_P
    
def StartUp4pressure(order_p,va,vb):
    
    #Np_p = order_p+1

    # Compute basic Legendre Gauss Lobatto grid
    r_elmt_p = JacobiGL(0,0,order_p)

    # Build reference element matrices
    V_p  = Vandermonde1D(order_p, r_elmt_p)
    invV_p = inv(V_p)
    Dr_p = Dmatrix1D(order_p, r_elmt_p, V_p)

    # build coordinates of all the nodes
    x_p = np.tile(va,(order_p+1,1)) + 0.5*(r_elmt_p.reshape(order_p+1,1)+1)*np.tile(vb,(order_p+1,1))
    
    return invV_p,Dr_p,x_p
    
def StartUp4V(order_V,va,vb):
    
    #Np_V = order_V+1

    # Compute basic Legendre Gauss Lobatto grid
    r_elmt_V = JacobiGL(0,0,order_V)

    # Build reference element matrices
    V  = Vandermonde1D(order_V, r_elmt_V)
    invV_V = inv(V)
    Dr_V = Dmatrix1D(order_V, r_elmt_V, V)

    # build coordinates of all the nodes
    x_V = np.tile(va,(order_V+1,1)) + 0.5*(r_elmt_V.reshape(order_V+1,1)+1)*np.tile(vb,(order_V+1,1))
    
    return invV_V,Dr_V,x_V
    
def Vandermonde1D(N,r):
    
    # Vandermonde1D(N,r)
    # Purpose : Initialize the 1D Vandermonde Matrix, V_{ij} = phi_j(r_i);
    #print (len(r))
    V1D = JacobiP(r,0,0,0)
    #print (np.shape(V1D[:,1]))
    for j in range(1,N+1):
        V1D = np.row_stack((V1D, JacobiP(r,0,0,j))) # JacobiP(r,0,0,j) is reduced to one dimention.
    return V1D.T #transpose to get the same result as in matlab

class model_para:
    # Can I set default as None? So that I do not need to add the parameters unused
    def __init__(self, mu=None, ka=None, rho=None, x=None, alpha=None, \
                 beta=None, J=None, rx=None, new_anelastic_params = None): 
        self.mu = mu
        self.ka = ka
        self.rho = rho
        self.x = x
        self.alpha = alpha
        self.beta = beta
        self.J = J
        self.rx = rx
        #self.rho_p = rho_p
        #self.xp = xp
        #self.xP = xP
        #self.xV = xV
        
        if new_anelastic_params is not None:

            if 'eta2' in new_anelastic_params.keys():

                self.eta2 = new_anelastic_params['eta2']

            if 'mu2' in new_anelastic_params.keys():

                self.mu2 = new_anelastic_params['mu2']
        
    def add_alpha(self,alpha):
        self.alpha = alpha
        
    def add_beta(self,beta):
        self.beta = beta
    
    def add_rho_p(self,rho_p):
        self.rho_p = rho_p
        
    def add_radius(self,radius):
        self.radius = radius
        
    def add_xp(self,xp):
        self.xp = xp
        
    def add_xP(self,xP):
        self.xP = xP
        
    def add_xV(self,xV):
        self.xV = xV
    
    def set_k(self,k):
        self.k = k
