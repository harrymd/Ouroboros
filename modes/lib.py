'''
Miscellaneous functions used in initialising the FEM problem.
'''

import numpy as np
import copy

from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import LinearOperator
import scipy.sparse as sps

# Define constants.
# G     Gravitational constant, units of cm^3 g^(-1) s^(-2) * e-6
# Wikipedia.
#G = 6.67408e-2 
# Mineos. 
G = 6.6723e-2 
tol = 1e-12

def equalEndMesh(ub,lb,n): #ub:upper bound, lb lower bound, N:number of points
    N_end = np.floor(n/3)
    intv = (ub-lb)/5 #interval
    
    #print (lb,lb+intv,intv/(N_end-1))
    interval1 = intv/(N_end-1)
    interval2 = (ub-2*intv-lb)/(n-2*N_end+1)
    VX1 = np.arange(lb,lb+intv-interval1/2,interval1)
    VX2 = np.arange(lb+intv,ub-intv-interval2/2,interval2)
    VX3 = np.arange(ub-intv,ub+interval1/2,interval1)
    
    #print (VX1,VX2,VX3)
    #print (np.shape(VX1),np.shape(VX2),np.shape(VX3))
    VX=np.concatenate((VX1,VX2,VX3), axis=None)
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

def findDiscon(r):
    # Find the index of discontinuity(where r(i) and r(i+1) are closer than 10^(-8))
    # Return a vector of indexes
    discon = []
    for i in range(len(r)-1):
        if abs(r[i] - r[i+1]) < 1e-8:
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

def mantlePoint_equalEnd(x,N,Rin,Rout): 
    # Return location of nodal point
    # Rin: inner bound, Rout: outer bound, N:number of points
    discon = findDiscon(x)
    VX_no_discon = equalEndMesh(Rin,Rout,N-len(discon))
    VX = np.concatenate([VX_no_discon,x[discon],x[discon]])
    VX.sort()
    
    return VX

def model_para_inv(r,para,x):
    # This function returns the new model parameters based on the interval
    # Return a vector of points between the nodal points
    nodalPara = model_para_nodal(r,para,x)
    discon = findDiscon(x) #place of discontinuity has '1' difference
    index = 0
    f = np.zeros(len(x)-len(discon)-1); #number of elements is nodal points minus discontinuity nimus 1
    for i in range(len(nodalPara)-1):
        if i+1 in discon:   # no points between discontinuity
            continue
        f[index]=(nodalPara[i]+nodalPara[i+1])/2 #mean value of two nodal points
        index=index+1
    return f
    
def model_para_nodal(r,para,x):
    # calculate parameters of nodal points
    disR = findDiscon(r) # find discontinuity
    disX = findDiscon(x)
    
    disX.insert(0,0) # add 0(km?) to discontinuity
    disX.append(len(x))
    disR.insert(0,0)
    disR.append(len(r))
    if len(disX) != len(disR):
        raise ValueError('Length of r and x not equal!')
    
    f_para = []
    for i in range(len(disX)-1):
        f_temp = model_para_pcw(r[disR[i]:disR[i+1]],para[disR[i]:disR[i+1]],x[disX[i]:disX[i+1]])
        # feed in peices between discontinuity
        f_para = np.append(f_para,f_temp)
        
    return f_para

def model_para_pcw(r,para,x):
    # feed in peices between discontinuity. 
    # r: Location from input; para: corrosponding parameter; x: location of nodal points for interpolation
    # Interpolate parameters from r to a set of points x
    # Return value of interpolated parameters f
    
    #print (r,para,x)
    f = np.zeros(len(x))
    starting = 0
    for j in range(len(x)-1):
        for i in np.arange(starting,len(r)-1):
            if x[j]-r[i]>tol and r[i+1]-x[j]>tol:
                f[j] = (para[i+1]-para[i])*(x[j]-r[i])/(r[i+1]-r[i])+para[i] #figure out what this is 
                starting = i
                break         
            elif abs(x[j]-r[i])<tol:
                f[j] = para[i]
                starting = i
                break
    f[-1] = para[-1]
    
    return f
            
def model_para_prime_inv(r,para,x):
    # This function returns the new model parameters' first derivative based on the interval
    nodalPara = model_para_nodal(r,para,x)
    #print(nodalPara)
    #print(np.shape(nodalPara))
    discon = findDiscon(x)
    index = 0
    f = np.zeros(len(x)-len(discon)-1)
    for i in range(len(nodalPara)-1):
        if i+1 in discon:
            continue
        f[index] = (nodalPara[i+1]-nodalPara[i])/(x[i+1]-x[i]);
        index = index+1
    
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
