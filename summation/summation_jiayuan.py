#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 10:42:42 2019
@author: jiayuanhan
This code does modes summation to generate synthetic seismogram
Equations from Theoretical Global Seismology, Chapter 10.3, P368-370
"""

from numba import jit
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math
from math import cos
from math import sin
#from scipy.special import lpmn
import scipy.special as special
import numpy as np
import pandas as pd
import distaz

tol = 1e-6

def syndat_s(x_sphe, x1_sphe, xx, eigen_U, eigen_V, data_omega, t, M, theta, phi):
    '''
#    x_loc: receiver
#    x1_loc: source
#    Green tensor and Moment tensor format:
#        \r\r     \r\theta     \r\phi
#        \theta\r \theta\theta \theta\phi
#        \phi\r   \phi\theta   \phi\phi
'''
    
    lmax = np.max((np.max(data_omega['l']),2))
    #lmax = 3

    cos_theta = cos(theta)
    sin_theta = sin(theta)

    #we only need the first array of lpmn, the associated Legendre function
    [associ_leg,associ_leg_prime] = special.lpmn(2,lmax,cos_theta)
    
    s_r = 0
    s_theta = 0
    s_phi = 0
#    am_r = []
#    am_theta = []
#    am_phi = []
    for i in range(len(data_omega)):
        l = data_omega['l'][i]
        #n = data_omega['n'][i]
        k2 = l*(l+1)
        k = math.sqrt(k2)

        for s in range(len(xx)):
            #linear interpolation of receiver
            if x_sphe[2] == xx[s]:
                U = eigen_U[i][s]
                V = eigen_V[i][s]
                break
            if x_sphe[2]>xx[s]:
                U = eigen_U[i][s] + (eigen_U[i][s]-eigen_U[i][s-1])/(xx[s]-xx[s-1])*(x_sphe[2]-xx[s])
                V = eigen_V[i][s] + (eigen_V[i][s]-eigen_V[i][s-1])/(xx[s]-xx[s-1])*(x_sphe[2]-xx[s])
                #W_prime = eigen_W_prime[s-1] + (eigen_W_prime[s]-eigen_W_prime[s-1])/(xx[s]-xx[s-1])*(x_sphe[2]-xx[s-1])
                break
            
        for s in range(len(xx)):
            #linear interpolation of source
            if x1_sphe[2] == xx[s]:
                U1 = eigen_U[i][s]
                V1 = eigen_V[i][s]
                if s == 0:
                    U1_prime = (eigen_U[i][s+1]-eigen_U[i][s])/(xx[s+1]-xx[s])
                    V1_prime = (eigen_V[i][s+1]-eigen_V[i][s])/(xx[s+1]-xx[s])
                else:
                    U1_prime = (eigen_U[i][s+1]-eigen_U[i][s-1])/(xx[s+1]-xx[s-1])
                    V1_prime = (eigen_V[i][s+1]-eigen_V[i][s-1])/(xx[s+1]-xx[s-1])
                break
            if x1_sphe[2]>xx[s]:
                U1 = eigen_U[i][s] + (eigen_U[i][s]-eigen_U[i][s-1])/(xx[s]-xx[s-1])*(x1_sphe[2]-xx[s])
                U1_prime = (eigen_U[i][s]-eigen_U[i][s-1])/(xx[s]-xx[s-1])
                V1 = eigen_V[i][s] + (eigen_V[i][s]-eigen_V[i][s-1])/(xx[s]-xx[s-1])*(x1_sphe[2]-xx[s])
                V1_prime = (eigen_V[i][s]-eigen_V[i][s-1])/(xx[s]-xx[s-1])
                break
        U = U*data_omega['angular frequency'][i]
        U1 = U1*data_omega['angular frequency'][i]
        U1_prime = U1_prime*data_omega['angular frequency'][i]
        V= V*data_omega['angular frequency'][i]*k
        V1 = V1*data_omega['angular frequency'][i]*k
        V1_prime = V1_prime*data_omega['angular frequency'][i]*k
        
        # 10.43-10.48.
        A0 = M[0][0]*U1_prime+(M[1][1]+M[2][2])/x1_sphe[2]*(U1-0.5*k*V1)
        #B0 = 0
        A1 = M[0][1]*(V1_prime-V1/x1_sphe[2]+k*U1/x1_sphe[2])/k
        B1 = M[0][2]*(V1_prime-V1/x1_sphe[2]+k*U1/x1_sphe[2])/k
        A2 = 0.5*(M[1][1]-M[2][2])*V1/(k*x1_sphe[2])
        B2 = M[1][2]*V1/(k*x1_sphe[2])
    
        coeff = (2*l+1)/(4*math.pi)
        Ar = coeff*U*(associ_leg[0][l]*A0+\
                      (-associ_leg[1][l])*(A1*math.cos(phi)+B1*math.sin(phi))+\
                      associ_leg[2][l]*(A2*math.cos(2*phi)+B2*math.sin(2*phi)))
        
        Atheta = coeff/k*V*((-sin_theta)*associ_leg_prime[0][l]*A0+\
                (-sin_theta)*(-associ_leg_prime[1][l])*(A1*math.cos(phi)+B1*math.sin(phi))+\
                (-sin_theta)*associ_leg_prime[2][l]*(A2*math.cos(2*phi)+B2*math.sin(2*phi))) 

        Aphi = coeff/k*V/sin_theta*(0+\
                (-associ_leg[1][l])*(-A1*math.sin(phi)+B1*math.cos(phi))+\
                associ_leg[2][l]*(-A2*2*math.sin(2*phi)+B2*2*math.cos(2*phi)))
        
        #coefficient for amplitude
        #s_coeff = 1/(data_omega['angular frequency'][i]**2)*(1-np.cos(data_omega['angular frequency'][i]*t))
        #coefficient for acceleration
        s_coeff = np.cos(data_omega['angular frequency'][i]*t)
        #print('Atheta = ',Atheta,'Aphi = ',Aphi)
        #print('zero',-sin_theta*associ_leg_prime[0][l]*A0)
        #print('fist',(-sin_theta)*(-associ_leg_prime[1][l])*(A1*math.cos(phi)+B1*math.sin(phi)))
        #print('second',(-sin_theta)*associ_leg_prime[2][l]*(A2*math.cos(2*phi)+B2*math.sin(2*phi))) 
#        am_r.append([n,l,Ar])
#        am_theta.append([n,l,Atheta])
#        am_phi.append([n,l,Aphi])
#        plt.figure()
#        line1, = plt.plot(t,Atheta*s_coeff)
#        line2, = plt.plot(t,Aphi*s_coeff)
#        plt.legend((line1,line2),('A theta','A phi'))
#        plt.show()
        s_r = s_r + Ar*s_coeff
        s_theta = s_theta + Atheta*s_coeff
        s_phi = s_phi + Aphi*s_coeff
#    am_r = np.array(am_r)
#    am_theta = np.array(am_theta)
#    am_phi = np.array(am_phi)    
#    plt.figure()
#    plt.plot(am_theta[:,2])
#    plt.plot(am_phi[:,2])
#    plt.show()
#    plt.figure()
#    line1, = plt.plot(t,s_theta)
#    line2, = plt.plot(t,s_phi)
#    plt.legend((line1,line2),('s theta','s phi'))
#    plt.show()
    return s_r,s_theta,s_phi#,am_r,am_theta,am_phi


#jit
#def cal_green(x_sphe, x1_sphe, xx, eigen_U, eigen_V, data_omega, t):
def syndat_t(x_sphe, x1_sphe, xx, eigen_W, data_omega, t, M, theta, phi):
    '''
    x_loc: receiver
    x1_loc: source
    Green tensor and Moment tensor format:
        \r\r     \r\theta     \r\phi
        \theta\r \theta\theta \theta\phi
        \phi\r   \phi\theta   \phi\phi
    '''
    
    lmax = np.max((np.max(data_omega['l']),3))

    cos_theta = cos(theta)
    sin_theta = sin(theta)

    #we only need the first array of lpmn, the associated Legendre function
    [associ_leg,associ_leg_prime] = special.lpmn(3,lmax,cos_theta)
    
    #s_r = 0
    s_theta = 0
    s_phi = 0
#    am_theta = []
#    am_phi = []
    for i in range(len(data_omega)):
        l = data_omega['l'][i]
        #n = data_omega['n'][i]
        k2 = l*(l+1)
        k = math.sqrt(k2)
    
        for s in range(len(xx)):
            #linear interpolation of receiver
            if x_sphe[2] == xx[s]:
                W = eigen_W[i][s]
                break
            if x_sphe[2]>xx[s]:
                W = eigen_W[i][s] + (eigen_W[i][s]-eigen_W[i][s-1])/(xx[s]-xx[s-1])*(x_sphe[2]-xx[s])
                #W_prime = eigen_W_prime[s-1] + (eigen_W_prime[s]-eigen_W_prime[s-1])/(xx[s]-xx[s-1])*(x_sphe[2]-xx[s-1])
                break
            
        for s in range(len(xx)):
            #linear interpolation of source
            if x1_sphe[2] == xx[s]:
                W1 = eigen_W[i][s]
                if s == 0:
                    W1_prime = (eigen_W[i][s+1]-eigen_W[i][s])/(xx[s+1]-xx[s])
                else:
                    W1_prime = (eigen_W[i][s+1]-eigen_W[i][s-1])/(xx[s+1]-xx[s-1])
                break
            if x1_sphe[2]>xx[s]:
                W1 = eigen_W[i][s] + (eigen_W[i][s]-eigen_W[i][s-1])/(xx[s]-xx[s-1])*(x1_sphe[2]-xx[s])
                W1_prime = (eigen_W[i][s]-eigen_W[i][s-1])/(xx[s]-xx[s-1])
                break
        W = W*data_omega['angular frequency'][i]*k
        W1 = W1*data_omega['angular frequency'][i]*k
        W1_prime = W1_prime*data_omega['angular frequency'][i]*k
        #what is the unit of eigenfunctions?
        #A0 = 0
        #B0 = 0
        A1 = -M[0][2]*(W1_prime-W1/x1_sphe[2])/k
        B1 = M[0][1]*(W1_prime-W1/x1_sphe[2])/k
        A2 = -M[1][2]*W1/(k*x1_sphe[2])
        B2 = (0.5*(M[1][1]-M[2][2])*W1)/(k*x1_sphe[2])
    
        coeff = (2*l+1)/(4*math.pi)
        #Ar = 0
        Atheta = coeff/k*W/sin_theta*(0+\
                (-associ_leg[1][l])*(-A1*math.sin(phi)+B1*math.cos(phi))+\
                associ_leg[2][l]*(-A2*2*math.sin(2*phi)+B2*2*math.cos(2*phi)))
        #two representation of prime of associated Legendre function
#        Aphi1 = coeff/k*(-W)*(0+\
#                0.5*(k2*associ_leg[0][l]-associ_leg[2][l])*(A1*math.cos(phi)+B1*math.sin(phi)) +\
#                -0.5*((l+2)*(l-1)*associ_leg[1][l]-associ_leg[3][l])*(A2*math.cos(2*phi)+B2*math.sin(2*phi)))
        Aphi = coeff/k*(-W)*(0+\
                (-sin_theta)*(-associ_leg_prime[1][l])*(A1*math.cos(phi)+B1*math.sin(phi))+\
                (-sin_theta)*associ_leg_prime[2][l]*(A2*math.cos(2*phi)+B2*math.sin(2*phi)))      
        #coefficient for amplitude
        #s_coeff = 1/(data_omega['angular frequency'][i]**2)*(1-np.cos(data_omega['angular frequency'][i]*t))
        #coefficient for acceleration
        s_coeff = np.cos(data_omega['angular frequency'][i]*t)
        #print('Atheta = ',Atheta,'Aphi = ',Aphi)
        #print('zero',-sin_theta*associ_leg_prime[0][l]*A0)
        #print('fist',(-sin_theta)*(-associ_leg_prime[1][l])*(A1*math.cos(phi)+B1*math.sin(phi)))
        #print('second',(-sin_theta)*associ_leg_prime[2][l]*(A2*math.cos(2*phi)+B2*math.sin(2*phi))) 
#        am_theta.append([n,l,Atheta])
#        am_phi.append([n,l,Aphi])
#        plt.figure()
#        line1, = plt.plot(t,Atheta*s_coeff)
#        line2, = plt.plot(t,Aphi*s_coeff)
#        plt.legend((line1,line2),('A theta','A phi'))
#        plt.show()
        #s_r = s_r + Ar*s_coeff
        s_theta = s_theta + Atheta*s_coeff
        s_phi = s_phi + Aphi*s_coeff
#    am_theta = np.array(am_theta)
#    am_phi = np.array(am_phi)    
#    plt.figure()
#    plt.plot(am_theta[:,2])
#    plt.plot(am_phi[:,2])
#    plt.show()
#    plt.figure()
#    line1, = plt.plot(t,s_theta)
#    line2, = plt.plot(t,s_phi)
#    plt.legend((line1,line2),('s theta','s phi'))
#    plt.show()
    return s_theta,s_phi#,am_theta,am_phi

#read input data
f_input = open('syndat/syndat_input.txt','r')
lines = f_input.readlines()
modes_T = int(lines[0])
temp = lines[1].split()
fT_min,fT_max = [float(temp[0]),float(temp[1])]
fileT = lines[2][:-1]
folderT = lines[3][:-1]+'/'

modes_S = int(lines[4])
temp = lines[5].split()
fS_min,fS_max = [float(temp[0]),float(temp[1])]
fileS = lines[6][:-1]
folderS = lines[7][:-1]+'/'

temp = lines[8].split()
lat = float(temp[0])
lon = float(temp[1])
dep = float(temp[2])
temp = lines[9].split()
tstart = int(temp[0])
tend = int(temp[1])
dt = float(temp[2])

event = lines[10]
if '\n' in event:
    event = event[:-1]

#start preparing data
t = np.arange(tstart,tend,dt)
nt = len(t)    
df = 1/ (nt*dt)
fny = 1/(2*dt)
f = np.arange(-fny,fny,df)

#theta = math.radians(19.122979902490304)
#phi = math.radians(180 - 224.733844133464)
#f_eigval = open('output/T.0000000.0000002.ASC','r')
#lines = f_eigval.readlines()
#header = lines[0].split()
#n = int(header[0])
#l = int(header[1])
#data_omega = 1000/float(header[4])
#s_rT,s_thetaT,s_phiT,s_rS,s_thetaS,s_phiS= [0,0,0,0,0,0]

if modes_T == 1:
    f_eigval = open(fileT,'r')

    lines = f_eigval.readlines()
    data_omegaT = []
    
    xx = []
    flag = 0
    
    eigen_W = []
    
    for line in lines[:]:
        temp = line.split()
        temp_n = int(temp[0])
        temp_l = int(temp[1])
        temp_omega = float(temp[2])*2*np.pi/1000
        if float(temp[2])>fT_max:
            continue
        if float(temp[2])<fT_min:
            continue
#        if temp_l > 20:
#            continue
#        if temp_n > 20 or temp_n == 0:
#            continue
#        if temp_n!=2 or temp_l!=1:
#            continue
        #temp_omega = float(temp[2])*2*math.pi/1000
        data_omegaT.append([temp_n,temp_l,temp_omega])
        #filename = 'S.'+format(temp_n,'07d')+'.'+format(temp_l,'07d')+'.ASC'
        filename = 'T.'+format(temp_n,'07d')+'.'+format(temp_l,'07d')+'.ASC'
        #f_eigfun = open('output/eigen_T/'+filename,'r')
    #    f_eigfun = open('output/eigen_T/'+filename,'r')
    #    lines_eigfun = f_eigfun.readlines()
    #    temp_W = []
    #    #temp_W_prime = []
    #
    #    for eig_line in lines_eigfun:
    #        eig_temp = eig_line.split()
            #read in radius data if haven't
    #        if flag == 0:
    #            xx.append(float(eig_temp[0]))
    #        temp_W.append(float(eig_temp[1]))
            #temp_W_prime.append(float(eig_temp[2]))
    #
    #    if len(xx) !=0:
    #        flag = 1
        
        temp = np.loadtxt(folderT+filename)
        #temp = np.loadtxt('output/eigen_T/'+filename)
        #read in radius data if haven't
        if flag == 0:
            xx = temp[:,0]
            flag = 1
        eigen_W.append(temp[:,1])
        #eigen_W.append(temp_W)
        #eigen_W_prime.append(temp_W_prime)
    
        #f_eigfun.close() 
    eigen_W = np.array(eigen_W)
    data_omegaT = pd.DataFrame(data_omegaT, columns=['n', 'l', 'angular frequency'])
        
    #s_rT,s_thetaT,s_phiT = syndat_t(x_sphe, x1_sphe, xx, eigen_W, data_omega, t, M, theta, phi)

if modes_S == 1:
    f_eigval = open(fileS,'r')
    
    lines = f_eigval.readlines()
    data_omegaS = []
    
    xx = []
    flag = 0
    
    eigen_U = []
    eigen_V = []
    
    for line in lines[:]:
        temp = line.split()
        temp_n = int(temp[0])
        temp_l = int(temp[1])
        temp_omega = float(temp[2])*2*np.pi/1000
        if float(temp[2])>fS_max:
            continue
        if float(temp[2])<fS_min:
            continue
        # identify and save spercific modes
        '''
        if temp_n==5 and temp_l==10:
            R10 = float(temp[2])/1000
        if temp_n==6 and temp_l==10:
            S10 = float(temp[2])/1000
        if temp_n==2 and temp_l==25:
            S25 = float(temp[2])/1000
        if temp_n==3 and temp_l==26:
            S26 = float(temp[2])/1000
        if temp_n==3 and temp_l==25:
            R25 = float(temp[2])/1000
        if temp_n==2 and temp_l==26:
            R26 = float(temp[2])/1000
        '''
#        if temp_l > 20:
#            continue
#        if temp_n > 20 or temp_n == 0:
#            continue
#        if temp_n!=2 or temp_l!=1:
#            continue
        #temp_omega = float(temp[2])*2*math.pi/1000
        data_omegaS.append([temp_n,temp_l,temp_omega])
        #filename = 'S.'+format(temp_n,'07d')+'.'+format(temp_l,'07d')+'.ASC'
        filename = 'S.'+format(temp_n,'07d')+'.'+format(temp_l,'07d')+'.ASC'
        #f_eigfun = open('output/eigen_T/'+filename,'r')
    #    f_eigfun = open('output/eigen_T/'+filename,'r')
    #    lines_eigfun = f_eigfun.readlines()
    #    temp_W = []
    #    #temp_W_prime = []
    #
    #    for eig_line in lines_eigfun:
    #        eig_temp = eig_line.split()
            #read in radius data if haven't
    #        if flag == 0:
    #            xx.append(float(eig_temp[0]))
    #        temp_W.append(float(eig_temp[1]))
            #temp_W_prime.append(float(eig_temp[2]))
    #
    #    if len(xx) !=0:
    #        flag = 1
        
        temp = np.loadtxt(folderS+filename)
        #temp = np.loadtxt('output/eigen_S/'+filename)
        #read in radius data if haven't
        if flag == 0:
            xx = temp[:,0]
            flag = 1
        eigen_U.append(temp[:,1])
        eigen_V.append(temp[:,2])
        #eigen_V.append(temp[:,3])
    
        #f_eigfun.close() 
    eigen_U = np.array(eigen_U)
    eigen_V = np.array(eigen_V)
    data_omegaS = pd.DataFrame(data_omegaS, columns=['n', 'l', 'angular frequency'])

f_eigval.close()
xx = np.array(xx)


if xx[0]<xx[1]:
    xx = np.flip(xx)
    xx = xx*1000
    if modes_T == 1:
        eigen_W = np.flip(eigen_W,1)
    if modes_S == 1:
        eigen_U = np.flip(eigen_U,1)
        eigen_V = np.flip(eigen_V,1)
#eigen_W = np.array(eigen_W)

'''
x_loc: receiver
x1_loc: source
Green tensor and Moment tensor format:
    \r\r     \r\theta     \r\phi
    \theta\r \theta\theta \theta\phi
    \phi\r   \phi\theta   \phi\phi
'''
#read moment tensor
f_event = open(event,'r')
line = f_event.readline()
#moment tensor
M = np.zeros((3,3))
data_event = line.split()
M[0][0] = float(data_event[12])
M[1][1] = float(data_event[13])
M[2][2] = float(data_event[14])
M[0][1] = float(data_event[15])
M[1][0] = float(data_event[15])
M[0][2] = float(data_event[16])
M[2][0] = float(data_event[16])
M[1][2] = float(data_event[17])
M[2][1] = float(data_event[17])
#moment tensor normalization coefficient
Mn = float(data_event[18])
Mn = Mn*1e-5*1e-2
 
#BJT station loction 
#x_loc = [40.0183,116.1679,0]
x_loc = [lat,lon,dep]
#source
x1_loc = [float(data_event[6]),float(data_event[7]),float(data_event[8])]
cal_dis = distaz.DistAz(x_loc[0], x_loc[1], x1_loc[0],x1_loc[1])
#print("%f  %f  %f" % (cal_dis.getDelta(), cal_dis.getAz(), cal_dis.getBaz()))
theta = math.radians(cal_dis.getDelta())
phi = math.radians(180 - cal_dis.getAz())
baz = cal_dis.getBaz()
#angles at receiver
co_phi = math.radians(baz-180) #turn angle baz from 224 to 44 for easier understanding
#x: receiver x1: source
#location in latitude, longitude, depth(km)
#theta=colatitude phi=longitude depth=R-r
#location in spheroidal form: colatitude, longitude, increasing radius

x_sphe = [90-x_loc[0],x_loc[1],(6371-x_loc[2])*1000]
x1_sphe = [90-x1_loc[0],x1_loc[1],(6371-x1_loc[2])*1000]

s_thetaT,s_phiT,s_rS,s_thetaS,s_phiS= [0,0,0,0,0]
if modes_T == 1:
    s_thetaT,s_phiT = syndat_t(x_sphe, x1_sphe, xx, eigen_W, data_omegaT, t, M, theta, phi)
if modes_S == 1:
    s_rS,s_thetaS,s_phiS = syndat_s(x_sphe, x1_sphe, xx, eigen_U, eigen_V, data_omegaS, t, M, theta, phi)
    #print((x_sphe, x1_sphe,  data_omegaS, M, theta, phi))

s_r = s_rS*Mn
s_theta = (s_thetaT+s_thetaS)*Mn
s_phi = (s_phiT+s_phiS)*Mn

#acceleration results are in nm/s^2
s_e = s_theta*math.sin(co_phi)-s_phi*math.cos(co_phi)
s_n = s_theta*math.cos(co_phi)+s_phi*math.sin(co_phi)
s_am = [np.max(np.abs(s_e)), np.max(np.abs(s_n)), np.max(np.abs(s_r))]
#s_am = [np.abs(s_e[0]), np.abs(s_n[0]), np.abs(s_r[0])]

#fourior transform
s_ef = np.abs(np.fft.fft(s_e))
s_ef = np.fft.fftshift(s_ef)
s_nf = np.abs(np.fft.fft(s_n))
s_nf = np.fft.fftshift(s_nf)
if modes_S == 1:
    s_rf = np.abs(np.fft.fft(s_r))
    s_rf = np.fft.fftshift(s_rf)

'''
#compare our results with mineos
path = 'Syndat_ASCT400/'
dirs = os.listdir(path)
#tmin = 0
#tmax = 30000
#dt = 0.2
#nt = 30000
#df = 1/ (nt*dt)
#fny = 1/(2*dt)
#f = np.arange(-fny,fny,df)
am = []
signal = []
for file in [dirs[21],dirs[14],dirs[16]]:
    temp = np.loadtxt(path+file,skiprows=3)    
    t = temp[:,0]
    data = temp[:,1]
    maximum = np.max(np.abs(data))
    #maximum = np.abs(data[0])
    am.append(maximum)
    signal.append(data)
#    print(maximum)
#    fig = plt.figure()
#    plt.plot(t[tmin:tmax],data[tmin:tmax])
#    plt.title('MINEOS Toroidal Modes Summation, '+file[28:31])
#    plt.xlabel('t/s')
#    plt.ylabel('Acceleration')
#    plt.show()
#    fig.savefig('../MINEOS_'+file[28:31]+'.jpg',dpi = 1000)
#    data_f = np.abs(np.fft.fft(data))
#    data_f = np.fft.fftshift(data_f)
#    fig = plt.figure()
#    plt.plot(f[38500:40000],data_f[38500:40000])
#    plt.title('MINEOS Frequency Spectrum, '+file[28:31])
#    plt.xlabel('Frequency/Hz')
#    plt.ylabel('Absolute Value')
#    plt.show()
#    fig.savefig('../frequency_'+file[28:31]+'.jpg',dpi = 1000)
    
#amplitude ratio
print(s_am[1]/s_am[0],s_am[2]/s_am[0])
print(am[1]/am[0],am[2]/am[0])
print('normalization ratio', am[0]/s_am[0], am[1]/s_am[1])
'''
#plot seismograms
tmin = 000
tmax = 9000
#fmin = nt//2+80
#fmax = nt//2+120
fmin = nt//2+800
fmax = nt//2+1200

#norm = (s_am[0]/am[0]+s_am[1]/am[1])/2
if modes_T == 1 and modes_S != 1:
    fig = plt.figure()
    line1, = plt.plot(t[tmin:tmax],s_e[tmin:tmax])
    #line2, = plt.plot(t[tmin:tmax],signal[0][tmin:tmax],linestyle='dashed')
    #plt.legend((line1),('our result'))
    plt.title('Modes Summation, East')
    plt.xlabel('t/s')
    plt.ylabel('Acceleration/(nm*s^-2)')
    plt.show()
    fig.savefig('../sum_east'+str(tmax)+'.jpg',dpi = 1000)
    
    fig = plt.figure()
    line1, = plt.plot(f[fmin:fmax],s_ef[fmin:fmax])
    #plt.legend((line1),('our result'))
    plt.title('Frequency Spectrum 3-10 mHz, East')
    plt.xlabel('Frequency/Hz')
    plt.ylabel('Amplitude')
    plt.show()
    fig.savefig('../fre_east'+str(tmax)+'.jpg',dpi = 1000)
    
    fig = plt.figure()
    line1, = plt.plot(t[tmin:tmax],s_n[tmin:tmax])
    #plt.legend((line1),('our result'))
    plt.title('Modes Summation, North')
    plt.xlabel('t/s')
    plt.ylabel('Acceleration/(nm*s^-2)')
    plt.show()
    fig.savefig('../sum_north'+str(tmax)+'.jpg',dpi = 1000)
    
    fig = plt.figure()
    line1, = plt.plot(f[fmin:fmax],s_nf[fmin:fmax])
    #plt.legend((line1),('our result'))
    plt.title('Frequency Spectrum 3-10 mHz, North')
    plt.xlabel('Frequency/Hz')
    plt.ylabel('Amplitude')
    plt.show()
    fig.savefig('../fre_north'+str(tmax)+'.jpg',dpi = 1000)

else:
    fig = plt.figure()
    line1, = plt.plot(t[tmin:tmax],s_e[tmin:tmax])
    #line2, = plt.plot(t[tmin:tmax],signal[0][tmin:tmax],linestyle='dashed')
    #plt.legend((line1),('our result'))
    plt.title('Modes Summation 3-10 mHz, East')
    plt.xlabel('t/s')
    plt.ylabel('Acceleration/(nm*s^-2)')
    plt.show()
    fig.savefig('../sum_east'+str(tmax)+'.jpg',dpi = 1000)
    
    fig = plt.figure()
    line1, = plt.plot(f[fmin:fmax],s_ef[fmin:fmax])
#    plt.axvline(x=S10,color='r',linestyle='dashed')
#    plt.axvline(x=R10,color='m',linestyle='dashed')
#    plt.axvline(x=S25,color='r',linestyle='dashed')
#    plt.axvline(x=R25,color='m',linestyle='dashed')
#    plt.axvline(x=S26,color='r',linestyle='dashed')
#    plt.axvline(x=R26,color='m',linestyle='dashed')
#    plt.xticks(np.arange(f[fmin],f[fmax]+0.0005,0.0005))
#    plt.text(0.00428,80000,'S10',color='r')
#    plt.text(0.00405,80000,'R10',color='m')
#    plt.text(0.00528,80000,'S25',color='r')
#    plt.text(0.00546,85000,'R25',color='m')
#    plt.text(0.00567,80000,'S26',color='r')
#    plt.text(0.00548,75000,'R26',color='m')

    #plt.legend((line1),('our result'))
    plt.title('Frequency Spectrum 3-10 mHz, East')
    plt.xlabel('Frequency/Hz')
    plt.ylabel('Amplitude')
    plt.show()
    fig.savefig('../fre_east'+str(tmax)+'.jpg',dpi = 1000)
    
    fig = plt.figure()
    line1, = plt.plot(t[tmin:tmax],s_n[tmin:tmax])
    #line2, = plt.plot(t[tmin:tmax],signal[1][tmin:tmax],linestyle='dashed')
    #plt.legend((line1),('our result'))
    plt.title('Modes Summation 3-10 mHz, North')
    plt.xlabel('t/s')
    plt.ylabel('Acceleration/(nm*s^-2)')
    plt.show()
    fig.savefig('../sum_north'+str(tmax)+'.jpg',dpi = 1000)
    
    fig = plt.figure()
    line1, = plt.plot(f[fmin:fmax],s_nf[fmin:fmax])
#    plt.axvline(x=S10,color='r',linestyle='dashed')
#    plt.axvline(x=R10,color='m',linestyle='dashed')
#    plt.axvline(x=S25,color='r',linestyle='dashed')
#    plt.axvline(x=R25,color='m',linestyle='dashed')
#    plt.axvline(x=S26,color='r',linestyle='dashed')
#    plt.axvline(x=R26,color='m',linestyle='dashed')
#    plt.xticks(np.arange(f[fmin],f[fmax]+0.0005,0.0005))
#    plt.text(0.00428,80000,'S10',color='r')
#    plt.text(0.00405,80000,'R10',color='m')
#    plt.text(0.00528,80000,'S25',color='r')
#    plt.text(0.00546,85000,'R25',color='m')
#    plt.text(0.00567,80000,'S26',color='r')
#    plt.text(0.00548,75000,'R26',color='m')
    
    #plt.legend((line1),('our result'))
    plt.title('Frequency Spectrum 3-10 mHz, North')
    plt.xlabel('Frequency/Hz')
    plt.ylabel('Amplitude')
    plt.show()
    fig.savefig('../fre_north'+str(tmax)+'.jpg',dpi = 1000)
    
    fig = plt.figure()
    line1, = plt.plot(t[tmin:tmax],s_r[tmin:tmax])
    #line2, = plt.plot(t[tmin:tmax],signal[2][tmin:tmax],linestyle='dashed')
    #plt.legend((line1),('our result'))
    plt.title('Modes Summation 3-10 mHz, Vertical')
    plt.xlabel('t/s')
    plt.ylabel('Acceleration/(nm*s^-2)')
    plt.show()
    fig.savefig('../sum_vertical'+str(tmax)+'.jpg',dpi = 1000)

    fig = plt.figure()
    line1, = plt.plot(f[fmin:fmax],s_rf[fmin:fmax])
#    plt.axvline(x=S10,color='r',linestyle='dashed')
#    plt.axvline(x=R10,color='m',linestyle='dashed')
#    plt.axvline(x=S25,color='r',linestyle='dashed')
#    plt.axvline(x=R25,color='m',linestyle='dashed')
#    plt.axvline(x=S26,color='r',linestyle='dashed')
#    plt.axvline(x=R26,color='m',linestyle='dashed')
#    plt.xticks(np.arange(f[fmin],f[fmax]+0.0005,0.0005))
#    plt.text(0.00428,17500,'S10',color='r')
#    plt.text(0.00405,17500,'R10',color='m')
#    plt.text(0.00528,17500,'S25',color='r')
#    plt.text(0.00546,20000,'R25',color='m')
#    plt.text(0.00567,17500,'S26',color='r')
#    plt.text(0.00548,16000,'R26',color='m')

    #plt.legend((line1),('our result'))
    plt.title('Frequency Spectrum 3-10 mHz, Vertical')
    plt.xlabel('Frequency/Hz')
    plt.ylabel('Amplitude')
    plt.show()
    fig.savefig('../fre_vertical'+str(tmax)+'.jpg',dpi = 1000)
'''
if modes_T == 1 and modes_S != 1:
    fig = plt.figure()
    line1, = plt.plot(t[tmin:tmax],s_e[tmin:tmax]/s_am[0])
    line2, = plt.plot(t[tmin:tmax],signal[0][tmin:tmax]/am[0],linestyle='dashed')
    plt.legend((line1,line2),('our result','mineos'))
    #plt.title('Toroidal Modes Summation, East, 400 modes')
    plt.title('Toroidal Modes Summation, East, 400 modes')
    plt.xlabel('t/s')
    plt.ylabel('Acceleration')
    plt.show()
    fig.savefig('../sum_east'+str(tmax)+'.jpg',dpi = 1000)
    
    fig = plt.figure()
    line1, = plt.plot(t[tmin:tmax],s_n[tmin:tmax]/s_am[1])
    line2, = plt.plot(t[tmin:tmax],signal[1][tmin:tmax]/am[1],linestyle='dashed')
    plt.legend((line1,line2),('our result','mineos'))
    #plt.title('Toroidal Modes Summation, North, 400 modes')
    plt.title('Toroidal Modes Summation, North, 400 modes')
    plt.xlabel('t/s')
    plt.ylabel('Acceleration')
    plt.show()
    fig.savefig('../sum_north'+str(tmax)+'.jpg',dpi = 1000)
else:
    fig = plt.figure()
    line1, = plt.plot(t[tmin:tmax],s_e[tmin:tmax]/s_am[0])
    line2, = plt.plot(t[tmin:tmax],signal[0][tmin:tmax]/am[0],linestyle='dashed')
    plt.legend((line1,line2),('our result','mineos'))
    #plt.title('Toroidal Modes Summation, East, 400 modes')
    plt.title('Spheroidal Modes Summation, East, 400 modes')
    plt.xlabel('t/s')
    plt.ylabel('Acceleration')
    plt.show()
    fig.savefig('../sum_east'+str(tmax)+'.jpg',dpi = 1000)
    
    fig = plt.figure()
    line1, = plt.plot(t[tmin:tmax],s_n[tmin:tmax]/s_am[1])
    line2, = plt.plot(t[tmin:tmax],signal[1][tmin:tmax]/am[1],linestyle='dashed')
    plt.legend((line1,line2),('our result','mineos'))
    #plt.title('Toroidal Modes Summation, North, 400 modes')
    plt.title('Spheroidal Modes Summation, North, 400 modes')
    plt.xlabel('t/s')
    plt.ylabel('Acceleration')
    plt.show()
    fig.savefig('../sum_north'+str(tmax)+'.jpg',dpi = 1000)
    fig = plt.figure()
    line1, = plt.plot(t[tmin:tmax],s_r[tmin:tmax]/s_am[2])
    line2, = plt.plot(t[tmin:tmax],signal[2][tmin:tmax]/am[2],linestyle='dashed')
    plt.legend((line1,line2),('our result','mineos'))
    #plt.title('Toroidal Modes Summation, North, 400 modes')
    plt.title('Spheroidal Modes Summation, Up, 400 modes')
    plt.xlabel('t/s')
    plt.ylabel('Acceleration')
    plt.show()
    fig.savefig('../sum_up'+str(tmax)+'.jpg',dpi = 1000)
'''







'''
#start preparing data
if modes_T == 1:
    f_eigval = open('output/test_T.eigen','r')
    lines = f_eigval.readlines()
    data_omegaT1 = []
    
    xx = []
    flag = 0
    
    eigen_W1 = []
    
    for line in lines[:]:
        temp = line.split()
        temp_n = int(temp[0])
        temp_l = int(temp[1])
        temp_omega = 1/float(temp[4])*2*math.pi
#        if 1000/float(temp[4])>3:
#            continue
#        if temp_l > 20:
#            continue
#        if temp_n > 20 or temp_n == 0:
#            continue
#        if temp_n!=2 or temp_l!=1:
#            continue
        #temp_omega = float(temp[2])*2*math.pi/1000
        data_omegaT1.append([temp_n,temp_l,temp_omega])
        #filename = 'S.'+format(temp_n,'07d')+'.'+format(temp_l,'07d')+'.ASC'
        filename = 'T.'+format(temp_n,'07d')+'.'+format(temp_l,'07d')+'.ASC'
        #f_eigfun = open('output/eigen_T/'+filename,'r')
    #    f_eigfun = open('output/eigen_T/'+filename,'r')
    #    lines_eigfun = f_eigfun.readlines()
    #    temp_W = []
    #    #temp_W_prime = []
    #
    #    for eig_line in lines_eigfun:
    #        eig_temp = eig_line.split()
            #read in radius data if haven't
    #        if flag == 0:
    #            xx.append(float(eig_temp[0]))
    #        temp_W.append(float(eig_temp[1]))
            #temp_W_prime.append(float(eig_temp[2]))
    #
    #    if len(xx) !=0:
    #        flag = 1
        
        temp = np.loadtxt('output/eigen_T/'+filename)
        #read in radius data if haven't
        if flag == 0:
            xx = temp[:,0]
            flag = 1
        eigen_W1.append(temp[:,1])
        #eigen_W.append(temp_W)
        #eigen_W_prime.append(temp_W_prime)
    
        #f_eigfun.close() 
    data_omegaT1 = pd.DataFrame(data_omegaT1, columns=['n', 'l', 'angular frequency'])
        
    #s_rT,s_thetaT,s_phiT = syndat_t(x_sphe, x1_sphe, xx, eigen_W, data_omega, t, M, theta, phi)
if modes_S == 1:
    f_eigval = open('output/test_S.eigen','r')
    
    lines = f_eigval.readlines()
    data_omegaS1 = []
    
    xx = []
    flag = 0
    
    eigen_U1 = []
    eigen_V1 = []
    
    for line in lines[:]:
        temp = line.split()
        temp_n = int(temp[0])
        temp_l = int(temp[1])
        temp_omega = 1/float(temp[4])*2*math.pi
    #    if 1000/float(temp[4])>3:
    #        continue
    #    if temp_l != 2:
    #        continue
#        if temp_n!=2 or temp_l!=1:
#            continue
        #temp_omega = float(temp[2])*2*math.pi/1000
        data_omegaS1.append([temp_n,temp_l,temp_omega])
        #filename = 'S.'+format(temp_n,'07d')+'.'+format(temp_l,'07d')+'.ASC'
        filename1 = 'S.'+format(temp_n,'07d')+'.'+format(temp_l,'07d')+'.ASC'
        #f_eigfun = open('output/eigen_T/'+filename,'r')
    #    f_eigfun = open('output/eigen_T/'+filename,'r')
    #    lines_eigfun = f_eigfun.readlines()
    #    temp_W = []
    #    #temp_W_prime = []
    #
    #    for eig_line in lines_eigfun:
    #        eig_temp = eig_line.split()
            #read in radius data if haven't
    #        if flag == 0:
    #            xx.append(float(eig_temp[0]))
    #        temp_W.append(float(eig_temp[1]))
            #temp_W_prime.append(float(eig_temp[2]))
    #
    #    if len(xx) !=0:
    #        flag = 1
        
        temp = np.loadtxt('output/eigen_S/'+filename1)
        #read in radius data if haven't
        if flag == 0:
            xx = temp[:,0]
            flag = 1
        eigen_U1.append(temp[:,1])
        eigen_V1.append(temp[:,3])
        #eigen_W.append(temp_W)
        #eigen_W_prime.append(temp_W_prime)
    
        #f_eigfun.close() 
    data_omegaS1 = pd.DataFrame(data_omegaS1, columns=['n', 'l', 'angular frequency'])
f_eigval.close()
xx = np.array(xx)
#eigen_W = np.array(eigen_W)
s_thetaT1,s_phiT1,s_rS1,s_thetaS1,s_phiS1= [0,0,0,0,0]
if modes_T == 1:
    s_thetaT1,s_phiT1 = syndat_t(x_sphe, x1_sphe, xx, eigen_W1, data_omegaT1, t, M, theta, phi)
if modes_S == 1:
    s_rS1,s_thetaS1,s_phiS1 = syndat_s(x_sphe, x1_sphe, xx, eigen_U1, eigen_V1, data_omegaS1, t, M, theta, phi)
    #print((x_sphe, x1_sphe,  data_omegaS, M, theta, phi))
s_r1 = s_rS1
s_theta1 = s_thetaT1+s_thetaS1
s_phi1 = s_phiT1+s_phiS1
s_e1 = s_theta1*math.sin(co_phi)-s_phi1*math.cos(co_phi)
s_n1 = s_theta1*math.cos(co_phi)+s_phi1*math.sin(co_phi)
s_am = [np.max(np.abs(s_e1)), np.max(np.abs(s_n1)), np.max(np.abs(s_r1))]
#s_am = [np.abs(s_e[0]), np.abs(s_n[0]), np.abs(s_r[0])]
    
#amplitude ratio
print(s_am[1]/s_am[0],s_am[2]/s_am[0])
print(am[1]/am[0],am[2]/am[0])
print('normalization ratio', am[0]/s_am[0], am[1]/s_am[1])
tmin = 000
tmax = 9000
norm = (s_am[0]/am[0]+s_am[1]/am[1])/2
if modes_T == 1 and modes_S != 1:
    fig = plt.figure()
    line1, = plt.plot(t[tmin:tmax],s_e1[tmin:tmax]/s_am[0])
    line2, = plt.plot(t[tmin:tmax],signal[0][tmin:tmax]/am[0],linestyle='dashed')
    plt.legend((line1,line2),('our result','mineos'))
    #plt.title('Toroidal Modes Summation, East, 400 modes')
    plt.title('Toroidal Modes Summation, East, 400 modes')
    plt.xlabel('t/s')
    plt.ylabel('Acceleration')
    plt.show()
    fig.savefig('../sum_east'+str(tmax)+'.jpg',dpi = 1000)
    
    fig = plt.figure()
    line1, = plt.plot(t[tmin:tmax],s_n1[tmin:tmax]/s_am[1])
    line2, = plt.plot(t[tmin:tmax],signal[1][tmin:tmax]/am[1],linestyle='dashed')
    plt.legend((line1,line2),('our result','mineos'))
    #plt.title('Toroidal Modes Summation, North, 400 modes')
    plt.title('Toroidal Modes Summation, North, 400 modes')
    plt.xlabel('t/s')
    plt.ylabel('Acceleration')
    plt.show()
    fig.savefig('../sum_north'+str(tmax)+'.jpg',dpi = 1000)
else:
    fig = plt.figure()
    line1, = plt.plot(t[tmin:tmax],s_e1[tmin:tmax]/s_am[0])
    line2, = plt.plot(t[tmin:tmax],signal[0][tmin:tmax]/am[0],linestyle='dashed')
    plt.legend((line1,line2),('our result','mineos'))
    #plt.title('Toroidal Modes Summation, East, 400 modes')
    plt.title('Spheroidal Modes Summation, East, 400 modes')
    plt.xlabel('t/s')
    plt.ylabel('Acceleration')
    plt.show()
    fig.savefig('../sum_east'+str(tmax)+'.jpg',dpi = 1000)
    
    fig = plt.figure()
    line1, = plt.plot(t[tmin:tmax],s_n1[tmin:tmax]/s_am[1])
    line2, = plt.plot(t[tmin:tmax],signal[1][tmin:tmax]/am[1],linestyle='dashed')
    plt.legend((line1,line2),('our result','mineos'))
    #plt.title('Toroidal Modes Summation, North, 400 modes')
    plt.title('Spheroidal Modes Summation, North, 400 modes')
    plt.xlabel('t/s')
    plt.ylabel('Acceleration')
    plt.show()
    fig.savefig('../sum_north'+str(tmax)+'.jpg',dpi = 1000)
    fig = plt.figure()
    line1, = plt.plot(t[tmin:tmax],s_r1[tmin:tmax]/s_am[2])
    line2, = plt.plot(t[tmin:tmax],signal[2][tmin:tmax]/am[2],linestyle='dashed')
    plt.legend((line1,line2),('our result','mineos'))
    #plt.title('Toroidal Modes Summation, North, 400 modes')
    plt.title('Spheroidal Modes Summation, Up, 400 modes')
    plt.xlabel('t/s')
    plt.ylabel('Acceleration')
    plt.show()
    fig.savefig('../sum_up'+str(tmax)+'.jpg',dpi = 1000)
'''
