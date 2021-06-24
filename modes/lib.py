'''
Miscellaneous functions used in initialising the FEM problem.
'''

import numpy as np
import copy

from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import LinearOperator
import scipy.sparse as sps

from Ouroboros.constants import G
# Convert gravitational constant to units of cm^3 g^(-1) s^(-2) * e-6.
G = G*1.0E9 

# Define numerical tolerance.
tol = 1e-12

def equalEndMesh(ub, lb, n):
    '''
    Create a mesh between lb (lower bound) and ub (upper bound) with n points.
    The mesh is finer near the boundaries: the first and last fifths of the
    mesh each have the same number of points as the inner three-fifths.
    (See Ye (2017), section 4.2.)
    '''

    # Find how many points in each of the two end regions.
    N_end = np.floor(n / 3.0)

    # Find the length of one-fifth of the interval.
    intv = (ub - lb)/5.0
    
    # Find the mesh spacing in the end regions (interval1) and the middle
    # region (interval2).
    interval1 = intv / (N_end - 1.0)
    interval2 = (ub - (2.0 * intv) - lb) / (n - (2.0 * N_end) + 1.0)

    # Generate points in each region.
    VX1 = np.arange(lb, lb + intv - (interval1 / 2.0), interval1)
    VX2 = np.arange(lb + intv, ub - intv - (interval2 / 2.0), interval2)
    VX3 = np.arange(ub - intv, ub + (interval1 / 2.0), interval1)
    
    # Join together the arrays for each region.
    VX = np.concatenate((VX1, VX2, VX3), axis = None)

    return VX

def equivForm(A_input,pos,cut_off_pos,isA):
    #Solve the zero block problem for efficiency
    #Return a full matrix
    #See equation 3.19 in the master thesis
    #pos must be a list, not a number

    Af_inv = None
    E = None
    
    tempA_row = A_input[pos,:]
    A_input = np.delete(A_input,pos,0)
    A_input = np.vstack((A_input,tempA_row))
    
    tempA_col = A_input[:,pos]
    A_input = np.delete(A_input,pos,1)
    A_input = np.hstack((A_input,tempA_col))
    
    if isA == 1:
        A_temp = A_input[:cut_off_pos,:cut_off_pos]
        E = A_input[:cut_off_pos,cut_off_pos:]
        Af = A_input[cut_off_pos:,cut_off_pos:]
    
        Af_inv = np.linalg.inv(Af)
        A_output = A_temp-np.matmul(np.matmul(E,Af_inv),E.T)
        A_output = (A_output+A_output.T)/2
    
    elif isA == 0:
        A_output = A_input[:cut_off_pos,:cut_off_pos]
    
    else:
        E = A_input[:cut_off_pos,cut_off_pos:]
        Af = A_input[cut_off_pos:,cut_off_pos:]
    
        Af_inv = np.linalg.inv(Af)
        A_output = np.matmul(np.matmul(E,Af_inv),E.T)
        A_output = (A_output+A_output.T)/2
    
    return A_output,Af_inv,E

def sparse_equivForm(A_input,pos,cut_off_pos,isA):
    #Solve the zero block problem for efficiency
    #Return a sparse matrix
    #See equation 3.19 in the master thesis
    '''
    Caustion: This function won't work if len(pos) == 1. Please use the non-sparse version if so.
    '''
    
    Af_inv = None
    E = None
    tempA_row = A_input[pos,:]
    #delete other rows of A_input
    pos2 = np.arange(np.shape(A_input)[0])
    pos2 = np.delete(pos2,pos)
    A_input = A_input[pos2,:]
    A_input = sps.vstack((A_input,tempA_row)).tocsr()
    
    tempA_col = A_input[:,pos]
    #delete other colomns of A_input
    pos2 = np.arange(np.shape(A_input)[0])
    pos2 = np.delete(pos2,pos)
    A_input = A_input[:,pos2]
    A_input = sps.hstack((A_input,tempA_col)).tocsc()
    
    if isA == 1:
#        A_temp = A_input[:cut_off_pos,:cut_off_pos].toarray()
#        E = A_input[:cut_off_pos,cut_off_pos:].toarray()
#        Af = A_input[cut_off_pos:,cut_off_pos:]
#    
#        Af_inv = sps.linalg.inv(Af).toarray()
#        A_output = A_temp-np.matmul(np.matmul(E,Af_inv),E.T)
#        A_output = (A_output+A_output.T)/2
        
        A_temp = A_input[:cut_off_pos,:cut_off_pos]
        E = A_input[:cut_off_pos,cut_off_pos:]
        Af = A_input[cut_off_pos:,cut_off_pos:]
    
        Af_inv = sps.linalg.inv(Af)
        #if len(pos) == 1:
        #    Af_inv = Af_inv.reshape((1,1))
        A_output = A_temp-E@Af_inv@E.T
        A_output = (A_output+A_output.T)/2
    
    elif isA == 0:
        A_output = A_input[:cut_off_pos,:cut_off_pos]
    
    else:
#        E = A_input[:cut_off_pos,cut_off_pos:].toarray()
#        Af = A_input[cut_off_pos:,cut_off_pos:].toarray()
#    
#        Af_inv = sps.linalg.inv(Af).toarray()
#        A_output = np.matmul(np.matmul(E,Af_inv),E.T)
#        A_output = (A_output+A_output.T)/2
        
        E = A_input[:cut_off_pos,cut_off_pos:]
        Af = A_input[cut_off_pos:,cut_off_pos:]
    
        Af_inv = sps.linalg.inv(Af)
        #if len(pos) == 1:
        #    Af_inv = Af_inv.reshape((1,1))
        A_output = E@Af_inv@E.T
        A_output = (A_output+A_output.T)/2
    
    return A_output,Af_inv,E

def findDiscon(r, dtol = 1.0E-8):
    '''
    Find the index of discontinuity (where r(i) and r(i+1) are closer than dtol) 
    and return a vector of indexes
    '''

    # Loop over points and look for discontinuities.
    discon = []
    for i in range(len(r)-1):

        if abs(r[i] - r[i+1]) < dtol:

            discon.append(i+1)
    
    return discon

def gravfield(r,rho,radius):
    q=0;
    if abs(r)<tol:
        g=0;
    else:
        for i in range(len(rho)):
            if radius[i+1]>radius[i]:
                if radius[i+1]<r:
                    q=q+rho[i]*(radius[i+1]**3-radius[i]**3)/3
                else:
                    q=q+rho[i]*(r**3-radius[i]**3)/3
                    break
        g = 4*np.pi*G*q/(r**2)
    return g

def gravfield_lst(rlist,rho,radius):
    length = len(rlist)
    g = np.zeros(length)
    
    for idx in range(length):
        r = rlist[idx]
        q = 0
        
        if abs(r)<tol:
            g[idx] = 0
        else:
            for i in range(len(rho)):
                if radius[i+1] > radius[i]:
                    if radius[i+1] < r:
                        q = q+rho[i]*(radius[i+1]**3-radius[i]**3)/3
                    else:
                        q = q+rho[i]*(r**3-radius[i]**3)/3;
                        break
            g[idx] = 4*np.pi*G*q/(r**2)
    return g

def mantlePoint_equalEnd(x, N, Rin, Rout): 
    '''
    Create a mesh of nodal points.
    Rin     Inner radius.
    Rout    Outer radius.
    N       Number of points.
    x       Grid points of input model (used to find discontinuities).
    '''

    # Find discontinuities.
    discon = findDiscon(x)

    # Create mesh without discontinuities.
    # Note the number of points is reduced so that the requested number
    # of points is preserved after the discontinuities are added back in.
    VX_no_discon = equalEndMesh(Rin, Rout, N - len(discon))

    # Add the discontinuity points to the mesh, and sort.
    VX = np.concatenate([VX_no_discon, x[discon], x[discon]])
    VX.sort()
    
    return VX

def model_para_inv(r, para, x):
    '''
    Get the parameter values for each element.
    First, the parameter values are interpolated at the vertices.
    Then, each element is assigned the mean value of its two vertices.

    Input:

    r       Radial coordinate of input model values.
    para    Input model values.
    x       Vertex coordinates.

    Output:

    f       Values within each element (between each consecutive vertex pair). 
    '''

    # Interpolate at the nodal points.
    nodalPara = model_para_nodal(r, para, x)

    # Prepare output array.
    # Duplicated values at the discontinuities will be ignored, so the number
    # of elements is the number of nodal points, minus the number of 
    # discontinuities, minus 1.
    discon = findDiscon(x) 

    # Prepare for loop.
    index = 0
    f = np.zeros(len(x) - len(discon) - 1)

    # Loop over nodal points.
    for i in range(len(nodalPara)-1):

        # No element exists between two repeated points.
        if (i + 1) in discon:

            continue

        # Assign the mean value at the two vertices to the element.
        f[index] = (nodalPara[i] + nodalPara[i+1]) / 2.0

        # Counter (ignoring discontinuities).
        index = index + 1

    return f
    
def model_para_nodal(r, para, x):
    '''
    Interpolate parameters at nodal points.

    Input:

    r       Radial coordinate of input model values.
    para    Input model values.
    x       Radial coordinates of element vertices.

    Note: x and r must have the same number of discontinuities.

    Output:

    f_para  Values interpolated at each nodal point.
    '''

    # Find indices of discontinuitiies in input and output point lists.
    disR = findDiscon(r)
    disX = findDiscon(x)
    
    # Add discontinuities representing the start and end of the point lists.
    disX.insert(0,0)
    disX.append(len(x))
    disR.insert(0,0)
    disR.append(len(r))

    # Check the number of discontinuities is the same for both lists.
    if len(disX) != len(disR):

        raise ValueError('Number of discontinuities in r and x not equal!')
    
    # Do interpolation separately within each discontinuity-bounded region.
    f_para = []
    for i in range(len(disX) - 1):

        # Interpolation for one region.
        f_temp = model_para_pcw(r[disR[i] : disR[i + 1]],
                    para[disR[i] : disR[i + 1]],
                    x[disX[i] : disX[i + 1]])

        # Append the values from this region to the master list.
        f_para = np.append(f_para, f_temp)
        
    return f_para

def model_para_pcw(r, para, x):
    '''
    Interpolate parameter from input model to mesh node points.
    It is assumed that the input is a discontinuity-free section.

    Input:

    r       Radial coordinate of input model values.
    para    Input model values.
    x       Radial coordinate of elements.

    Output:

    f       Values interpolated at each nodal point.
    '''

    # Prepare for loop.
    f = np.zeros(len(x))
    starting = 0

    # Loop over all nodal points except the last one.
    for j in range(len(x) - 1):

        # Loop over remaining points.
        for i in np.arange(starting, len(r) - 1):

            # If the node point is not close to an input point, the value
            # is found by linear interpolation.
            if (x[j] - r[i] > tol) and (r[i + 1] - x[j] > tol):

                # Linear interpolation.
                f[j] = (para[i + 1] - para[i]) * (x[j] - r[i]) / (r[i + 1] - r[i]) \
                        + para[i]

                # Move on to next nodal point.
                starting = i
                break         

            # If the node point is effectively the same as the input point,
            # no need for interpolation.
            elif abs(x[j] - r[i]) < tol:

                f[j] = para[i]

                # Move on to next nodal point.
                starting = i
                break

    # The final value is the same for input and output because they
    # cover the same interval.
    f[-1] = para[-1]
    
    return f
            
def model_para_prime_inv(r, para, x):
    '''
    Calculate first derivative of parameter within elements.

    Input:

    r       Radial coordinate of input model values.
    para    Input model values.
    x       Radial coordinates of element vertices.

    Output:

    f       Numerical estimate of first derivative of parameter within each
            element.
    '''

    # Interpolate the parameter at the vertex points.
    nodalPara = model_para_nodal(r, para, x)

    # Find discontinuities in the vertex points.
    discon = findDiscon(x)

    # Prepare for loop.
    index = 0
    f = np.zeros(len(x) - len(discon) - 1)

    # Loop over vertices.
    for i in range(len(nodalPara)-1):

        # There are no elements in between repeated points.
        if i + 1 in discon:

            continue

        # Estimate the first derivative within the element based on a simple
        # ratio d param / d r.
        f[index] = (nodalPara[i + 1] - nodalPara[i]) / (x[i + 1] - x[i])

        # Move to next vertex (counter ignores repeated points).
        index = index + 1
    
    return f
    
def modelDiv(model,pos):
    new_model = copy.deepcopy(model)
    #print (new_model.rho,np.shape(new_model.rho))
    #print (pos)
    new_model.mu = new_model.mu[pos]
    new_model.rho = new_model.rho[pos]
    if hasattr(new_model,'rho_p'):
        new_model.rho_p = new_model.rho_p[pos]
    new_model.ka = new_model.ka[pos]
    new_model.x = new_model.x[:,pos]
    if hasattr(new_model,'xp'):
        new_model.xp = new_model.xp[:,pos]
    if hasattr(new_model,'xP'):
        new_model.xP = new_model.xP[:,pos]
    if hasattr(new_model,'xV'):
        new_model.xV = new_model.xV[:,pos]
    if hasattr(new_model,'alpha'):
        new_model.alpha = new_model.alpha[pos]
    if hasattr(new_model,'beta'):
        new_model.beta = new_model.beta[pos]
    new_model.J = new_model.J[:,pos]
    new_model.rx = new_model.rx[:,pos]
    
    return new_model
        
def remDiscon(x,isBall):
    discon=findDiscon(x)
    # remove the discontinuity inside the mantle
    if isBall == 0:
        new_x = np.delete(x,discon) #remove all the discontinuity
    elif isBall == 2:
        new_x = np.delete(x,discon[1:]) #remove 1st
    elif isBall == 1:
        new_x = np.delete(x,discon[2:])
    else:
        raise ValueError('remove discontinuity error')
        
    return new_x

def sqzx(xi,Ki,orderi):
    #Squeeze FEM 2d matrix xi to 1d xx
    Npi = orderi+1
    xx = xi.flatten('F')
    posM = np.arange(orderi,Npi*Ki-1,Npi)
    xx = np.delete(xx,posM)
    
    return xx
