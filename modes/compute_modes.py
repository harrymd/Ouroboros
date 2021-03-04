"""
Scripts to compute modes.

Some parameters:
    order, order_V, order_p, order_P
                FEM orders of specified variables.
    brk_num     Number of solid-liquid discontinuities ('breaks') in the model. 
    brk_radius  Radius of the breaks.
    thickness   Thickness of layers between breaks. 
    count_blk_size
                A parameter to count how many grids in the matrix the code has processed.
    block_pos   Index of the end grids of the current block.
    block_len   Record the length of blocks.
"""

import os

import numpy as np
from scipy.linalg import eigh
from scipy.linalg import block_diag
from scipy.interpolate import interp1d
import scipy.sparse as sps

from Ouroboros.common import mkdir_if_not_exist, load_model, get_path_adjusted_model
from Ouroboros.modes import FEM
from Ouroboros.modes import lib
from Ouroboros.modes import setup

# Set G value (units of cm^3 g^(-1)s^(-2)*e-6). 
## G value updated to current value based on Wikipedia.
#G = 6.67408e-2  
# G value in Mineos.
G = 6.6723e-2

# Toroidal modes. -------------------------------------------------------------
def toroidal_modes(run_info):
    '''
    Get toroidal modes of each layer from inside to outside(inner core to mantle for earth), returning with several layers of modes.
    '''

    # Unpack input.
    if run_info['use_attenuation']:

        model_path = get_path_adjusted_model(run_info)

    else:

        model_path  = run_info['path_model']

    dir_output  = run_info['dir_output']
    dir_type    = run_info['dirs_type']['T']
    lmin, lmax  = run_info['l_lims']
    nmin, nmax  = run_info['n_lims']
    num_elmt    = run_info['n_layers']
    
    # Set the gravity control variable.
    switch = 'T'
    
    # Set up the model and various finite-element parameters.
    (model, vs, count_thick, thickness, essen,
    invV, invV_p, invV_V, invV_P,
    order, order_p, order_V, order_P,
    Dr, Dr_p, Dr_V, Dr_P,
    x, x_V, x_P, VX,
    rho, radius,
    block_type, brk_radius, brk_num, layers,
    dir_eigenfunc_list, path_eigenvalues_list) = \
        prep_fem(model_path, dir_type, num_elmt, switch)

    for l in range(lmin,lmax+1):
        
        print('toroidal_modes: l = {:>5d} (from {:>5d} to {:>5d})'.format(l, lmin, lmax))

        # No modes with l = 0.
        if l == 0:

            print('Toroidal modes with l = 0 do not exist without external torques; skipping.')
            continue
        
        # Build the matrices, solve and save.
        solve_toroidal(l, nmin, nmax, model, x, vs, layers, brk_num, count_thick, thickness, invV, order, Dr, dir_eigenfunc_list, path_eigenvalues_list)

    return 

def solve_toroidal(l, nmin, nmax, model, x, vs, layers, brk_num, count_thick, thickness, invV, order, Dr, dir_eigenfunc_list, path_eigenvalues_list):
    
    # Calculate asymptotic wavenumber.
    k = np.sqrt(l*(l + 1.0))
    model.set_k(k)

    #path_eigenvalues_list = iter(path_eigenvalues_list)
    #dir_eigenfunc_list = iter(dir_eigenfunc_list)
    
    # Loop over the layers of the model.
    j = 0
    for i in range(layers):

        # Toroidal modes only exist in solid layers.
        if vs[brk_num[i]] != 0:

            # Build the matrices A and B and solve to get the eigenvalues
            # and eigenvectors.
            eigvals, eigvecs = build_matrices_toroidal_and_solve(model, count_thick, i, invV, order, Dr)

            # 
            process_eigen_toroidal(l, eigvals, eigvecs, nmin, nmax, count_thick, thickness, order, x, i, path_eigenvalues_list[j], dir_eigenfunc_list[j], save = True)

            j = j + 1

    return

def build_matrices_toroidal_and_solve(model, count_thick, i, invV, order, Dr):

    cur_model = lib.modelDiv(model,np.arange(count_thick[i],count_thick[i+1]))
    # generate matrices A and B such that Ax  =  omega^2*Bx
    [A,B] = FEM.toroidal(cur_model,invV,order,Dr)
    
    if i==0:

        #pos must be a list even with one number
        pos = [0]
        cut_off_pos = np.shape(A)[0]-1
        #A_singularity = lib.equivForm(A,pos,cut_off_pos,1)[0]
        #B_singularity = lib.equivForm(B,pos,cut_off_pos,0)[0]
        #sparse form doesn't work for only on singularity, use non-sparse form instead
        #A = sps.csc_matrix(A)
        #B = sps.csc_matrix(B)
        #A_list = lib.sparse_equivForm(A,pos,cut_off_pos,1)
        #A_singularity = A_list[0].toarray()
        #A0_inv = A_list[1].toarray()
        #E_singularity = A_list[2].toarray()
        #B_singularity = lib.equivForm(B,pos,cut_off_pos,0)[0].toarray()
        A_list = lib.equivForm(A,pos,cut_off_pos,1)
        A_singularity = A_list[0]
        A0_inv = A_list[1]
        E_singularity = A_list[2]
        B_singularity = lib.equivForm(B,pos,cut_off_pos,0)[0]
        #test matlab code by computing T0 of each eigenfunction to check if it is done correctly
        eigvals,eigvecs0 = eigh(A_singularity,B_singularity)
        T0 = -A0_inv@E_singularity.T@eigvecs0
        eigvecs = np.vstack((T0,eigvecs0))

    else:
        # eigensolver
        eigvals,eigvecs = eigh(A,B)

    return eigvals, eigvecs

def process_eigen_toroidal(l, eigvals, eigvecs, nmin, nmax, count_thick, thickness, order, x, i_layer, path_eigenvalues, dir_eigenfunc, save = True):

    # Transform from eigenvalues (square of angular frequency (rad/s)) to
    # frequencies (mHz).
    omega = np.sqrt(eigvals)/(2.0*np.pi)
    omega = omega*1000.0
    
    # Get output paths and directories.
    #path_eigenvalues    = next(path_eigenvalues_list)
    #dir_eigenfunc       = next(dir_eigenfunc_list)
    
    # Loop over radial order.
    for n in range(nmin, nmax+1):
    
        # Skip modes that don't exist without external forcing.
        if (n == 0) and (l == 0 or l == 1):
    
            continue
    
        # Scale and get radial coordinate. 
        W_eigen = eigvecs[:,n]/np.sqrt(l*(l+1))/(omega[n]*2*np.pi)
        xx = lib.sqzx(x[:,count_thick[i_layer]:count_thick[i_layer + 1]], thickness[i_layer], order)

        if save:
        
            # Write out the eigenvalues.
            with open(path_eigenvalues, 'a') as f_out:
    
                f_out.write('{:>10d} {:>10d} {:>16.12f}\n'.format(n, l, omega[n]))
    
            # Write eigenfunction.
            file_eigenfunc = '{:>05d}_{:>05d}.npy'.format(n, l)
            path_eigenfunc = os.path.join(dir_eigenfunc, file_eigenfunc)
            out_arr = np.array([1000.0*xx, W_eigen])
            np.save(path_eigenfunc, out_arr)
    
    return omega

# Radial modes. ---------------------------------------------------------------
def radial_modes(run_info):

    # Unpack input.
    if run_info['use_attenuation']:

        model_path = get_path_adjusted_model(run_info)

    else:

        model_path  = run_info['path_model']
    dir_output  = run_info['dir_output']
    dir_type    = run_info['dirs_type']['R']
    nmin, nmax  = run_info['n_lims']
    num_elmt    = run_info['n_layers']
    switch      = run_info['switch']
    
    # Set up the model and various finite-element parameters.
    (model, vs, count_thick, thickness, essen,
    invV, invV_p, invV_V, invV_P,
    order, order_p, order_V, order_P,
    Dr, Dr_p, Dr_V, Dr_P,
    x, x_V, x_P, VX,
    rho, radius,
    block_type, brk_radius, brk_num, layers,
    dir_eigenfunc, path_eigenvalues) = \
        prep_fem(model_path, dir_type, num_elmt, switch)

    print('radial_modes (switch = {:})'.format(switch))
        
    # Construct the matrices A and B.
    # Note use l = -1 to make sure k = 0.
    A_singularity, B_singularity, A0_inv,                   \
    E_singularity, B_eqv_pressure, block_type, block_len =  \
        build_matrices_radial_or_spheroidal(
            -1, model, count_thick,
            invV, invV_p, invV_V, invV_P,
            order, order_p, order_V, order_P,
            Dr, Dr_p, Dr_V, Dr_P,
            rho, radius,
            block_type, brk_radius, brk_num, layers, switch)
    
    # Find the eigenvalues and eigenvectors. 
    eigvals, eigvecs = eigh(A_singularity, B_singularity)
    
    # Check this section. Look at Jia's code if necessary.
    # Probably no essential spectrum to remove.
    # Convert to mHz, remove essential spectrum, renormalise and save.            
    process_eigen_radial_or_spheroidal(
        0, eigvals, eigvecs,
        count_thick, thickness, essen, layers,
        nmin, nmax, order, order_V,
        x, x_V,
        block_type, block_len, A0_inv, E_singularity, B_eqv_pressure,
        path_eigenvalues, dir_eigenfunc, switch)
    
    return 

def helmholtz_R_noGP_or_G(
                block_type, block_len, layers,
                A, A_bdr_cond, B, B_bdr_cond, switch):

    # first layer
    if block_type[0] == 0:
        #cut off singularity in pressure is different if it is liquid
        pos_singularity_p = [block_len[0][0]]
        cut_off_pos_singularity_p = np.shape(A)[0]-1
        #A_singularity_p = lib.equivForm(A_bdr_cond,pos_singularity_p,cut_off_pos_singularity_p,0)[0]
        #B_singularity_p = lib.equivForm(B_bdr_cond,pos_singularity_p,cut_off_pos_singularity_p,0)[0]
        #sparse matrix version
        A_singularity_p = lib.sparse_equivForm(A_bdr_cond,pos_singularity_p,cut_off_pos_singularity_p,0)[0]
        B_singularity_p = lib.sparse_equivForm(B_bdr_cond,pos_singularity_p,cut_off_pos_singularity_p,0)[0]

        if switch == 'R_noGP':
            
            block_len[0][1] = block_len[0][1]-1

        elif switch == 'R_G':

            block_len[0][2] = block_len[0][2]-1

    else:
        A_singularity_p = A_bdr_cond
        B_singularity_p = B_bdr_cond
        
    # generalized Helmholtz Decomposition: equivalent form for pressure
    pos_pressure = []
    count_blk_size = 0
    pressure_size = 0
    for i in range(layers):
        if block_type[i] == 0:
            if pressure_size == 0:
                pos_pressure = np.arange(count_blk_size+block_len[i][0],\
                                         count_blk_size+block_len[i][0]+block_len[i][1])
            else:
                pos_pressure = np.hstack((pos_pressure, np.arange(count_blk_size+block_len[i][0],\
                                                count_blk_size+block_len[i][0]+block_len[i][1])))
            pressure_size = pressure_size+block_len[i][1]
            block_len[i].pop(1)
        count_blk_size = count_blk_size + np.sum(block_len[i])
    
    if pressure_size == 0:
        A_eqv_pressure = A_singularity_p
        B_eqv_pressure = B_singularity_p
    else:
        cut_off_pos_pressure = np.shape(A_singularity_p)[0] - pressure_size
        #A_eqv_pressure = lib.equivForm(A_singularity_p,pos_pressure,cut_off_pos_pressure,1)[0]
        #B_eqv_pressure = lib.equivForm(B_singularity_p,pos_pressure,cut_off_pos_pressure,0)[0]
        #sparse matrix version
        A_eqv_pressure = lib.sparse_equivForm(A_singularity_p,pos_pressure,cut_off_pos_pressure,1)[0].toarray()
        B_eqv_pressure = lib.sparse_equivForm(B_singularity_p,pos_pressure,cut_off_pos_pressure,0)[0].toarray()
    
    # equivalent form for singularity
    pos_singularity = [0]
    cut_off_pos_singularity = np.shape(A_eqv_pressure)[0]-1
    if block_type[0] == 0:
        A_singularity = lib.equivForm(A_eqv_pressure,pos_singularity,cut_off_pos_singularity,0)[0]
        B_singularity = lib.equivForm(B_eqv_pressure,pos_singularity,cut_off_pos_singularity,0)[0]
        
        # hrmd
        A0_inv = None
        E_singularity = None

        #sparse matrix version won't be used because only one element needed to be change here
        #A_singularity = lib.sparse_equivForm(A_eqv_pressure,pos_singularity,cut_off_pos_singularity,0)[0].toarray()
        #B_singularity = lib.sparse_equivForm(B_eqv_pressure,pos_singularity,cut_off_pos_singularity,0)[0].toarray()

    else:
        A_singularity = lib.equivForm(A_eqv_pressure,pos_singularity,cut_off_pos_singularity,1)[0]
        B_singularity = lib.equivForm(B_eqv_pressure,pos_singularity,cut_off_pos_singularity,0)[0]
        #sparse matrix version won't be used because only one element needed to be change here
        #A_singularity = lib.sparse_equivForm(A_eqv_pressure,pos_singularity,cut_off_pos_singularity,1)[0].toarray()
        #B_singularity = lib.sparse_equivForm(B_eqv_pressure,pos_singularity,cut_off_pos_singularity,0)[0].toarray()

        # hrmd
        A0 = A_eqv_pressure[:cut_off_pos_singularity, :cut_off_pos_singularity]
        A0_inv = np.linalg.inv(A0)

        E_singularity = A_eqv_pressure[:cut_off_pos_singularity,cut_off_pos_singularity:]
        E_singularity = np.squeeze(E_singularity)

    return A_singularity, B_singularity, A0_inv, E_singularity, B_eqv_pressure

def helmholtz_R_GP(
                block_type, block_len, layers,
                A, A_bdr_cond, B, B_bdr_cond):

    #squeeze P
    #do I really need to do that?
    #squeeze perturbation in fluid end points to solid
    pos = []
    count_blk_size = 0
    E_inv = np.eye(np.shape(A_bdr_cond)[0])
    for i in range(layers-1):
        if block_type[i] == 0:
            pos.append(count_blk_size+block_len[i][0]+block_len[i][1]-1)
            E_inv[pos[i],count_blk_size+np.sum(block_len[i])+block_len[i+1][0]] = 1
        else:
            pos.append(count_blk_size+np.sum(block_len[i])+block_len[i+1][0])
            E_inv[pos[i],count_blk_size+block_len[i][0]+block_len[i][1]-1] = 1   
        
        count_blk_size = count_blk_size+np.sum(block_len[i])
    
    #E_inv = sps.csc_matrix(E_inv)
    A_sqz_P = E_inv.T@A_bdr_cond@E_inv
    B_sqz_P = E_inv.T@B_bdr_cond@E_inv
    
    A_sqz_P = np.delete(A_sqz_P,pos,0)
    B_sqz_P = np.delete(B_sqz_P,pos,0)
    A_sqz_P = np.delete(A_sqz_P,pos,1)
    B_sqz_P = np.delete(B_sqz_P,pos,1)
            
    #A_sqz_P = (A_sqz_P+A_sqz_P.T)/2
    #B_sqz_P = (B_sqz_P+B_sqz_P.T)/2
    # sparse matrix version
    A_sqz_P = sps.csc_matrix((A_sqz_P+A_sqz_P.T)/2)
    B_sqz_P = sps.csc_matrix((B_sqz_P+B_sqz_P.T)/2)
        
    #correct block_len to the right number
    for i in range(layers-1):
        if block_type[i] == 0:
            block_len[i][1] = block_len[i][1]-1
        else:
            block_len[i+1][1] = block_len[i+1][1]-1
    
    # first layer
    if block_type[0] == 0:
        #cut off singularity in pressure is different if it is liquid
        #pos_singularity_p is different from withough Perturbation because P array is ahead of pressure array
        pos_singularity_p = [block_len[0][0]+block_len[0][1]]
        cut_off_pos_singularity_p = np.shape(A_sqz_P)[0]-1
        #A_singularity_p = lib.equivForm(A_sqz_P,pos_singularity_p,cut_off_pos_singularity_p,0)[0]
        #B_singularity_p = lib.equivForm(B_sqz_P,pos_singularity_p,cut_off_pos_singularity_p,0)[0]
        #sparse matrix version
        A_singularity_p = lib.sparse_equivForm(A_sqz_P,pos_singularity_p,cut_off_pos_singularity_p,0)[0]
        B_singularity_p = lib.sparse_equivForm(B_sqz_P,pos_singularity_p,cut_off_pos_singularity_p,0)[0]
        block_len[0][1] = block_len[0][1]-1
    else:
        A_singularity_p = A_sqz_P
        B_singularity_p = B_sqz_P
        
    #I can put equivalent form of pressure and perturbation in the same time
    # generalized Helmholtz Decomposition: equivalent form for pressure and perturbation
    pos_pP = []
    count_blk_size = 0
    pP_size = 0
    for i in range(layers):
        if pP_size == 0:
            pos_pP = np.arange(count_blk_size+block_len[i][0],\
                               count_blk_size+np.sum(block_len[i]))
                
        else:
            pos_pP = np.hstack((pos_pP, np.arange(count_blk_size+block_len[i][0],\
                                count_blk_size+np.sum(block_len[i]))))
        
        count_blk_size = count_blk_size + np.sum(block_len[i])

        if block_type[i] == 0: #fluid
            pP_size = pP_size+block_len[i][1]+block_len[i][2]
            block_len[i].pop(2)
            block_len[i].pop(1)
        else: #solid
            pP_size = pP_size+block_len[i][1]
            block_len[i].pop(1)
        
    cut_off_pos_pP = np.shape(A_singularity_p)[0] - pP_size
    #A_eqv_pressure = lib.equivForm(A_singularity_p,pos_pP,cut_off_pos_pP,1)[0]
    #B_eqv_pressure = lib.equivForm(B_singularity_p,pos_pP,cut_off_pos_pP,0)[0]
    #sparse matrix version
    A_eqv_pressure = lib.sparse_equivForm(A_singularity_p,pos_pP,cut_off_pos_pP,1)[0].toarray()
    B_eqv_pressure = lib.sparse_equivForm(B_singularity_p,pos_pP,cut_off_pos_pP,0)[0].toarray()
    
    # equivalent form for singularity
    pos_singularity = [0]
    cut_off_pos_singularity = np.shape(A_eqv_pressure)[0]-1
    if block_type[0] == 0:
        A_singularity = lib.equivForm(A_eqv_pressure,pos_singularity,cut_off_pos_singularity,0)[0]
        B_singularity = lib.equivForm(B_eqv_pressure,pos_singularity,cut_off_pos_singularity,0)[0]
        #sparse matrix version won't be used because only one element needed to be change here
        #A_singularity = lib.sparse_equivForm(A_eqv_pressure,pos_singularity,cut_off_pos_singularity,0)[0].toarray()
        #B_singularity = lib.sparse_equivForm(B_eqv_pressure,pos_singularity,cut_off_pos_singularity,0)[0].toarray()

        # hrmd
        A0_inv = None
        E_singularity = None

    else:
        A_singularity = lib.equivForm(A_eqv_pressure,pos_singularity,cut_off_pos_singularity,1)[0]
        B_singularity = lib.equivForm(B_eqv_pressure,pos_singularity,cut_off_pos_singularity,0)[0]
        #sparse matrix version won't be used because only one element needed to be change here
        #A_singularity = lib.sparse_equivForm(A_eqv_pressure,pos_singularity,cut_off_pos_singularity,1)[0].toarray()
        #B_singularity = lib.sparse_equivForm(B_eqv_pressure,pos_singularity,cut_off_pos_singularity,0)[0].toarray()

        # hrmd
        A0 = A_eqv_pressure[:cut_off_pos_singularity, :cut_off_pos_singularity]
        A0_inv = np.linalg.inv(A0)

        E_singularity = A_eqv_pressure[:cut_off_pos_singularity,cut_off_pos_singularity:]
        E_singularity = np.squeeze(E_singularity)

    return A_singularity, B_singularity, A0_inv, E_singularity, B_eqv_pressure

# Spheroidal modes. -----------------------------------------------------------
def spheroidal_modes(run_info):

    # Unpack input.
    if run_info['use_attenuation']:

        model_path = get_path_adjusted_model(run_info)

    else:

        model_path  = run_info['path_model']
    dir_output  = run_info['dir_output']
    dir_type    = run_info['dirs_type']['S']
    lmin, lmax  = run_info['l_lims']
    nmin, nmax  = run_info['n_lims']
    num_elmt    = run_info['n_layers']
    switch      = run_info['switch']

    # Set up the model and various finite-element parameters.
    (model, vs, count_thick, thickness, essen,
    invV, invV_p, invV_V, invV_P,
    order, order_p, order_V, order_P,
    Dr, Dr_p, Dr_V, Dr_P,
    x, x_V, x_P, VX,
    rho, radius,
    block_type, brk_radius, brk_num, layers,
    dir_eigenfunc, path_eigenvalues) = \
        prep_fem(model_path, dir_type, num_elmt, switch)

    # Loop over angular order.
    for l in range(lmin, lmax + 1):

        print('spheroidal_modes (switch = {:}): l = {:>5d} (from {:>5d} to {:>5d})'.format(switch, l, lmin, lmax))
        
        if l == 0:

            print('Spheroidal modes with l = 0 are known as radial modes and should be calculated separately using radial_modes().') 
            continue

        # Construct the matrices A and B.
        A_singularity, B_singularity, A0_inv,                   \
        E_singularity, B_eqv_pressure, block_type, block_len =  \
            build_matrices_radial_or_spheroidal(
                l, model, count_thick,
                invV, invV_p, invV_V, invV_P,
                order, order_p, order_V, order_P,
                Dr, Dr_p, Dr_V, Dr_P,
                rho, radius,
                block_type, brk_radius, brk_num, layers, switch)
        
        # Find the eigenvalues and eigenvectors. 
        eigvals, eigvecs = eigh(A_singularity, B_singularity)
        
        # Convert to mHz, remove essential spectrum, renormalise and save.            
        process_eigen_radial_or_spheroidal(
            l, eigvals, eigvecs,
            count_thick, thickness, essen, layers,
            nmin, nmax, order, order_V,
            x, x_V,
            block_type, block_len, A0_inv, E_singularity, B_eqv_pressure,
            path_eigenvalues, dir_eigenfunc, switch, save = True)

    return

def helmholtz_S_noGP_or_G(
        block_type, block_len, layers,
        A, A_bdr_cond, B, B_bdr_cond):
    
    # first layer
    if block_type[0] == 0:
        #cut off singularity in pressure is different if it is liquid
        pos_singularity_p = [block_len[0][0]+block_len[0][1]]
        cut_off_pos_singularity_p = np.shape(A)[0]-1
        #change number of essential spectrum
        #essen = essen - 1
        
        #A_singularity_p = lib.equivForm(A_bdr_cond,pos_singularity_p,cut_off_pos_singularity_p,0)[0]
        #B_singularity_p = lib.equivForm(B_bdr_cond,pos_singularity_p,cut_off_pos_singularity_p,0)[0]
        #sparse matrix version
        A_singularity_p = lib.sparse_equivForm(A_bdr_cond,pos_singularity_p,cut_off_pos_singularity_p,0)[0]
        B_singularity_p = lib.sparse_equivForm(B_bdr_cond,pos_singularity_p,cut_off_pos_singularity_p,0)[0]
        A_singularity_p = A_bdr_cond
        B_singularity_p = B_bdr_cond
        block_len[0][2] = block_len[0][2]-1

    else:

        A_singularity_p = A_bdr_cond
        B_singularity_p = B_bdr_cond
        
    # generalized Helmholtz Decomposition: equivalent form for pressure
    pos_pressure = []
    count_blk_size = 0
    pressure_size = 0
    for i in range(layers):
        if block_type[i] == 0:
            if pressure_size == 0:
                pos_pressure = np.arange(count_blk_size+block_len[i][0]+block_len[i][1],\
                                         count_blk_size+block_len[i][0]+block_len[i][1]+block_len[i][2])
            else:
                pos_pressure = np.hstack((pos_pressure, np.arange(count_blk_size+block_len[i][0]+block_len[i][1],\
                                                count_blk_size+block_len[i][0]+block_len[i][1]+block_len[i][2])))
            pressure_size = pressure_size+block_len[i][2]
            block_len[i].pop(2)
        count_blk_size = count_blk_size + np.sum(block_len[i])
     
    if pressure_size == 0:
        A_eqv_pressure = A_singularity_p
        B_eqv_pressure = B_singularity_p
    else:
        cut_off_pos_pressure = np.shape(A_singularity_p)[0] - pressure_size
        #A_eqv_pressure = lib.equivForm(A_singularity_p,pos_pressure,cut_off_pos_pressure,1)[0]
        #B_eqv_pressure = lib.equivForm(B_singularity_p,pos_pressure,cut_off_pos_pressure,0)[0]
        #sparse matrix version
        A_eqv_pressure = lib.sparse_equivForm(A_singularity_p,pos_pressure,cut_off_pos_pressure,1)[0]
        B_eqv_pressure = lib.sparse_equivForm(B_singularity_p,pos_pressure,cut_off_pos_pressure,0)[0]
    
    # equivalent form for singularity
    # do not need to change block_len here because it will be resumed later while writing eigenfunctions
    pos_singularity = [0,block_len[0][0]]
    cut_off_pos_singularity = np.shape(A_eqv_pressure)[0]-2
    
    if block_type[0] == 0:
        #A_singularity = lib.equivForm(A_eqv_pressure,pos_singularity,cut_off_pos_singularity,0)[0]
        #B_singularity = lib.equivForm(B_eqv_pressure,pos_singularity,cut_off_pos_singularity,0)[0]
        #sparse matrix version
        A_singularity = lib.sparse_equivForm(A_eqv_pressure,pos_singularity,cut_off_pos_singularity,0)[0].toarray()
        #A_list = lib.sparse_equivForm(A_eqv_pressure,pos_singularity,cut_off_pos_singularity,0)
        B_singularity = lib.sparse_equivForm(B_eqv_pressure,pos_singularity,cut_off_pos_singularity,0)[0].toarray()
    else:
        #A_singularity = lib.equivForm(A_eqv_pressure,pos_singularity,cut_off_pos_singularity,1)[0]
        #B_singularity = lib.equivForm(B_eqv_pressure,pos_singularity,cut_off_pos_singularity,0)[0]
        #sparse matrix version
        #A_singularity = lib.sparse_equivForm(A_eqv_pressure,pos_singularity,cut_off_pos_singularity,1)[0].toarray()
        
        A_list = lib.sparse_equivForm(A_eqv_pressure,pos_singularity,cut_off_pos_singularity,1)
        A_singularity = A_list[0].toarray()
        A0_inv = A_list[1].toarray()
        E_singularity = A_list[2].toarray()
        B_singularity = lib.sparse_equivForm(B_eqv_pressure,pos_singularity,cut_off_pos_singularity,0)[0].toarray()
        
    #boundary condition at r = 0 is U=V=0, delete these lines and colomns
    #A_singularity = lib.sparse_equivForm(A_eqv_pressure,pos_singularity,cut_off_pos_singularity,0)[0].toarray()
    #B_singularity = lib.sparse_equivForm(B_eqv_pressure,pos_singularity,cut_off_pos_singularity,0)[0].toarray()

    return A_singularity, B_singularity, A0_inv, E_singularity, B_eqv_pressure 

def helmholtz_S_GP(
        block_type, block_len, layers,
        A, A_bdr_cond, B, B_bdr_cond):

    #squeeze P
    #do I really need to do that?
    #squeeze perturbation in fluid end points to solid
    pos = []
    count_blk_size = 0
    E_inv = np.eye(np.shape(A_bdr_cond)[0])
    for i in range(layers-1):
        if block_type[i] == 0:
            pos.append(count_blk_size+block_len[i][0]+block_len[i][1]+block_len[i][2]-1)
            E_inv[pos[i],count_blk_size+np.sum(block_len[i])+block_len[i+1][0]+block_len[i+1][1]] = 1
        else:
            pos.append(count_blk_size+np.sum(block_len[i])+block_len[i+1][0]+block_len[i+1][1])
            E_inv[pos[i],count_blk_size+block_len[i][0]+block_len[i][1]+block_len[i][2]-1] = 1   
        
        count_blk_size = count_blk_size+np.sum(block_len[i])
    
    #E_inv = sps.csc_matrix(E_inv)
    A_sqz_P = E_inv.T@A_bdr_cond@E_inv
    B_sqz_P = E_inv.T@B_bdr_cond@E_inv
    
    A_sqz_P = np.delete(A_sqz_P,pos,0)
    B_sqz_P = np.delete(B_sqz_P,pos,0)
    A_sqz_P = np.delete(A_sqz_P,pos,1)
    B_sqz_P = np.delete(B_sqz_P,pos,1)
            
    #A_sqz_P = (A_sqz_P+A_sqz_P.T)/2
    #B_sqz_P = (B_sqz_P+B_sqz_P.T)/2
    # sparse matrix version
    A_sqz_P = sps.csc_matrix((A_sqz_P+A_sqz_P.T)/2)
    B_sqz_P = sps.csc_matrix((B_sqz_P+B_sqz_P.T)/2)
        
    #correct block_len to the right number
    for i in range(layers-1):
        if block_type[i] == 0:
            block_len[i][2] = block_len[i][2]-1
        else:
            block_len[i+1][2] = block_len[i+1][2]-1
    
    # first layer
    if block_type[0] == 0:
        #cut off singularity in pressure is different if it is liquid
        #pos_singularity_p is different from withough Perturbation because P array is ahead of pressure array
        pos_singularity_p = [block_len[0][0]+block_len[0][1]+block_len[0][2]]
        cut_off_pos_singularity_p = np.shape(A_sqz_P)[0]-1
        #change number of essential spectrum
        #essen = essen - 1
        
        #A_singularity_p = lib.equivForm(A_sqz_P,pos_singularity_p,cut_off_pos_singularity_p,0)[0]
        #B_singularity_p = lib.equivForm(B_sqz_P,pos_singularity_p,cut_off_pos_singularity_p,0)[0]
        #sparse matrix version
        A_singularity_p = lib.sparse_equivForm(A_sqz_P,pos_singularity_p,cut_off_pos_singularity_p,0)[0]
        B_singularity_p = lib.sparse_equivForm(B_sqz_P,pos_singularity_p,cut_off_pos_singularity_p,0)[0]
        block_len[0][2] = block_len[0][2]-1
    else:
        A_singularity_p = A_sqz_P
        B_singularity_p = B_sqz_P
        
    #I can put equivalent form of pressure and perturbation in the same time
    # generalized Helmholtz Decomposition: equivalent form for pressure and perturbation
    pos_pP = []
    count_blk_size = 0
    pP_size = 0
    for i in range(layers):
        if pP_size == 0:
            pos_pP = np.arange(count_blk_size+block_len[i][0]+block_len[i][1],\
                               count_blk_size+np.sum(block_len[i]))
                
        else:
            pos_pP = np.hstack((pos_pP, np.arange(count_blk_size+block_len[i][0]+block_len[i][1],\
                                count_blk_size+np.sum(block_len[i]))))
        
        count_blk_size = count_blk_size + np.sum(block_len[i])

        if block_type[i] == 0: #fluid
            pP_size = pP_size+block_len[i][2]+block_len[i][3]
            block_len[i].pop(3)
            block_len[i].pop(2)
        else: #solid
            pP_size = pP_size+block_len[i][2]
            block_len[i].pop(2)
        
    cut_off_pos_pP = np.shape(A_singularity_p)[0] - pP_size
    #A_eqv_pressure = lib.equivForm(A_singularity_p,pos_pP,cut_off_pos_pP,1)[0]
    #B_eqv_pressure = lib.equivForm(B_singularity_p,pos_pP,cut_off_pos_pP,0)[0]
    #sparse matrix version
    A_eqv_pressure = lib.sparse_equivForm(A_singularity_p,pos_pP,cut_off_pos_pP,1)[0]
    B_eqv_pressure = lib.sparse_equivForm(B_singularity_p,pos_pP,cut_off_pos_pP,0)[0]
    
    # equivalent form for singularity
    pos_singularity = [0,block_len[0][0]]
    cut_off_pos_singularity = np.shape(A_eqv_pressure)[0]-2
    
    if block_type[0] == 0:
        #A_singularity = lib.equivForm(A_eqv_pressure,pos_singularity,cut_off_pos_singularity,0)[0]
        #B_singularity = lib.equivForm(B_eqv_pressure,pos_singularity,cut_off_pos_singularity,0)[0]
        #sparse matrix version
        A_singularity = lib.sparse_equivForm(A_eqv_pressure,pos_singularity,cut_off_pos_singularity,0)[0].toarray()
        B_singularity = lib.sparse_equivForm(B_eqv_pressure,pos_singularity,cut_off_pos_singularity,0)[0].toarray()
    else:
        #A_singularity = lib.equivForm(A_eqv_pressure,pos_singularity,cut_off_pos_singularity,1)[0]
        #B_singularity = lib.equivForm(B_eqv_pressure,pos_singularity,cut_off_pos_singularity,0)[0]
        #sparse matrix version
        #A_singularity = lib.sparse_equivForm(A_eqv_pressure,pos_singularity,cut_off_pos_singularity,1)[0].toarray()
        A_list = lib.sparse_equivForm(A_eqv_pressure,pos_singularity,cut_off_pos_singularity,1)
        A_singularity = A_list[0].toarray()
        A0_inv = A_list[1].toarray()
        E_singularity = A_list[2].toarray()
        B_singularity = lib.sparse_equivForm(B_eqv_pressure,pos_singularity,cut_off_pos_singularity,0)[0].toarray()
    
    #boundary condition at r = 0 is U=V=0, delete these lines and colomns
    #A_singularity = lib.sparse_equivForm(A_eqv_pressure,pos_singularity,cut_off_pos_singularity,0)[0].toarray()
    #B_singularity = lib.sparse_equivForm(B_eqv_pressure,pos_singularity,cut_off_pos_singularity,0)[0].toarray()
    #A_singularity = A_eqv_pressure.toarray()
    #B_singularity = B_eqv_pressure.toarray()

    return A_singularity, B_singularity, A0_inv, E_singularity, B_eqv_pressure 

# Radial and spheroidal modes. ------------------------------------------------
def build_matrices_radial_or_spheroidal(
        l, model, count_thick,
        invV, invV_p, invV_V, invV_P,
        order, order_p, order_V, order_P,
        Dr, Dr_p, Dr_V, Dr_P,
        rho, radius,
        block_type, brk_radius, brk_num, layers, switch):
    
    # Calculate k (asymptotic wavenumber).
    k = np.sqrt(l*(l + 1.0))

    # Set model parameters.
    model.set_k(k)
    # generate matrices A and B such that Ax  =  omega^2*Bx
    A = []
    B = []
    #length of U,V,p,P, etc
    #U,V for solid, U,V,p for fluid
    block_len = []
    block_pos = [0]
    
    for i in range(layers):
        cur_model = lib.modelDiv(model,np.arange(count_thick[i],count_thick[i+1]))
        #basically follow the original order
        if block_type[i] == 0:

            if switch == 'S_noGP':

                [tempA,tempB,temp_block_len] = FEM.fluid_noG_mixedV(cur_model,invV,invV_p,invV_V,order,order_p,order_V,Dr,Dr_p,Dr_V)

            elif switch == 'S_G':

                [tempA,tempB,temp_block_len] = FEM.fluid_G_mixedV(cur_model,invV,invV_p,invV_V,order,order_p,order_V,Dr,Dr_p,Dr_V,rho,radius)

            elif switch == 'S_GP':

                [tempA,tempB,temp_block_len] = FEM.fluid_GP_mixedPV(cur_model,invV,invV_p,invV_P,invV_V,order,order_p,order_P,order_V,Dr,Dr_p,Dr_P,Dr_V,rho,radius)

            elif switch == 'R_noGP':

                [tempA,tempB,temp_block_len] = FEM.radial_fluid_noG_mixedV(cur_model,invV,invV_p,order,order_p,Dr,Dr_p)

            elif switch == 'R_G':

                [tempA,tempB,temp_block_len] = FEM.radial_fluid_G_mixedV(cur_model,invV,invV_p,order,order_p,Dr,Dr_p,rho,radius)

            elif switch == 'R_GP':

                [tempA,tempB,temp_block_len] = FEM.radial_fluid_GP_mixedPV(cur_model,invV,invV_p,invV_P,order,order_p,order_P,Dr,Dr_p,Dr_P,rho,radius)

            #block_type.append(0)
            #cut off singularity in pressure after boundary condition
        else:

            if switch == 'S_noGP':

                [tempA,tempB,temp_block_len] = FEM.solid_noG(cur_model,invV,order,Dr)

            elif switch == 'S_G':

                [tempA,tempB,temp_block_len] = FEM.solid_G(cur_model,invV,order,Dr,rho,radius)

            elif switch == 'S_GP':

                [tempA,tempB,temp_block_len] = FEM.solid_GPmixed(cur_model,invV,invV_P,order,order_P,Dr,Dr_P,rho,radius)

            elif switch == 'R_noGP':

                [tempA,tempB,temp_block_len] = FEM.radial_solid_noG(cur_model,invV,order,Dr)

            elif switch == 'R_G':

                [tempA,tempB,temp_block_len] = FEM.radial_solid_G(cur_model,invV,order,Dr,rho,radius)

            elif switch == 'R_GP':

                [tempA,tempB,temp_block_len] = FEM.radial_solid_GPmixed(cur_model,invV,invV_P,order,order_P,Dr,Dr_P,rho,radius)

            #block_type.append(1)
            
        if i == 0:

            A = tempA
            B = tempB

        else:

            A = block_diag(A,tempA)
            B = block_diag(B,tempB)

        block_len.append(temp_block_len)
        for j in range(len(temp_block_len)):
            block_pos.append(block_pos[-1]+temp_block_len[j])

    if switch == 'S_GP':

        #boundary condition of the outer bound
        P_end = brk_radius[-1]*(l+1)/(4*np.pi*G);
        A[-1,-1] = A[-1,-1]+P_end;
    
    # impose boundary condition
    C = np.zeros(np.shape(A))
    count_blk_size = 0
    for i in range(layers-1):
        count_blk_size = count_blk_size + np.sum(block_len[i])
        if block_type[i] == 1: #solid-fluid
            # manage unit: *1e12/1e15

            if switch in ['S_noGP', 'S_G', 'R_GP']: 

                C[count_blk_size+block_len[i+1][0]+block_len[i+1][1],block_len[i][0]-1] = brk_radius[i+1]**2*1e-3
                C[block_len[i][0]-1,count_blk_size+block_len[i+1][0]+block_len[i+1][1]] = brk_radius[i+1]**2*1e-3

            elif switch == 'S_GP':

                C[count_blk_size+block_len[i+1][0]+block_len[i+1][1]+block_len[i+1][2],block_len[i][0]-1] = brk_radius[i+1]**2*1e-3
                C[block_len[i][0]-1,count_blk_size+block_len[i+1][0]+block_len[i+1][1]+block_len[i+1][2]] = brk_radius[i+1]**2*1e-3

            elif switch in ['R_noGP', 'R_G']:
                
                C[count_blk_size+block_len[i+1][0],block_len[i][0]-1] = brk_radius[i+1]**2*1e-3
                C[block_len[i][0]-1,count_blk_size+block_len[i+1][0]] = brk_radius[i+1]**2*1e-3

            if switch in ['R_G', 'R_GP', 'S_G', 'S_GP']:

                g_Rc_plus = lib.gravfield(brk_radius[i+1],rho,radius)
                C[block_len[i][0]-1,block_len[i][0]-1] = -rho[brk_num[i+1]]*g_Rc_plus*brk_radius[i+1]**2

        else: #fluid-solid

            # manage unit: *1e12/1e15
            C[count_blk_size-1,count_blk_size] = -brk_radius[i+1]**2*1e-3
            C[count_blk_size,count_blk_size-1] = -brk_radius[i+1]**2*1e-3

            if switch in ['R_G', 'R_GP', 'S_G', 'S_GP']:
        
                g_Rb_minus = lib.gravfield(brk_radius[i+1],rho,radius)
                C[count_blk_size,count_blk_size] = rho[brk_num[i+1]-1]*g_Rb_minus*brk_radius[i+1]**2
    #A_bdr_cond = A+C
    #B_bdr_cond = B
    # sparse matrix version
    # It is hard to use scipy.sparse, so the implimentation is subtle here for sparse matrix
    A_bdr_cond = sps.csc_matrix(A+C)
    B_bdr_cond = sps.csc_matrix(B)
    
    if switch in ['S_noGP', 'S_G']:

        A_singularity, B_singularity, A0_inv, E_singularity, B_eqv_pressure = \
            helmholtz_S_noGP_or_G(
                block_type, block_len, layers,
                A, A_bdr_cond, B, B_bdr_cond)

    elif switch == 'S_GP':

        A_singularity, B_singularity, A0_inv, E_singularity, B_eqv_pressure = \
            helmholtz_S_GP(
                block_type, block_len, layers,
                A, A_bdr_cond, B, B_bdr_cond)

    elif switch in ['R_noGP', 'R_G']:

        A_singularity, B_singularity, A0_inv, E_singularity, B_eqv_pressure = \
            helmholtz_R_noGP_or_G(
                block_type, block_len, layers,
                A, A_bdr_cond, B, B_bdr_cond, switch)

    elif switch == 'R_GP':

        A_singularity, B_singularity, A0_inv, E_singularity, B_eqv_pressure = \
            helmholtz_R_GP(
                block_type, block_len, layers,
                A, A_bdr_cond, B, B_bdr_cond)


    return A_singularity, B_singularity, A0_inv, E_singularity, B_eqv_pressure, block_type, block_len

def process_eigen_radial_or_spheroidal(
        l, eigvals, eigvecs,
        count_thick, thickness, essen, layers,
        nmin, nmax, order, order_V,
        x, x_V,
        block_type, block_len, A0_inv, E_singularity, B_eqv_pressure,
        path_eigenvalues, dir_eigenfunc, switch, save = True):
    
    # Transform from eigenvalues (square of angular frequency (rad/s)) to
    # frequencies (mHz).
    omega = np.sqrt(eigvals)/(2.0*np.pi)
    omega = omega*1000.0

    if switch in ['S_noGP', 'S_G', 'S_GP']: 

        # Remove the essential spectrum.
        # Case of solid inner core.
        if block_type[0] == 1:

            # Specific handling of l = 1 to avoid wrong matching of modes to their names.
            if l == 1:

                # Skip the essential spectrum.
                eigen       = omega[essen+1:]
                eigen_coeff = eigvecs[:,essen+1:]

            else:

                # Skip the essential spectrum.
                eigen       = omega[essen:]
                eigen_coeff = eigvecs[:,essen:]

        # Case of liquid inner core.
        else:

            # Skip the essential spectrum.
            eigen       = omega[essen-1:]
            eigen_coeff = eigvecs[:,essen-1:]

    else:
        
        # No need to skip essential spectrum for radial modes (?).
        eigen = omega
        eigen_coeff = eigvecs
    
    # Loop over radial order.
    for n in range(nmin, nmax+1):
        
        # Skip modes that don't exist.
        if (n == 0 or n==1) and l==1:
            continue
        
        if block_type[0] == 0:
            
            if switch in ['S_noGP', 'S_G', 'S_GP']:
                
                UV0 = [0,0]

            elif switch in ['R_noGP', 'R_G', 'R_GP']:

                UV0 = 0

        else:
            
            UV0 = -A0_inv@E_singularity.T@eigen_coeff[:,n]

        if switch in ['S_noGP', 'S_G', 'S_GP']:

            #UV0 is different from matlab version, although its 1e-13 vs 1e-12. Should I change it to zero?
            coeff_cur = np.insert(eigen_coeff[:,n], [0,block_len[0][0]-1], UV0)

        elif switch in ['R_noGP', 'R_G', 'R_GP']:

            coeff_cur = np.insert(eigen_coeff[:,n], 0, UV0)
        
        scale = np.sqrt(coeff_cur@B_eqv_pressure@coeff_cur)*eigen[n]*2.0*np.pi
        
        # Eigenvector of U and V.
        U_eigen = []
        V_eigen = set_if_needed([], switch, ['S_noGP', 'S_G', 'S_GP']) 

        # Get relative radius of eigenvector. This is done by squeeze of x in each layer.
        xx = []
        count_blk_size = 0
        # Renormalization.
        for i in range(layers):
            
            U_eigen = np.hstack((U_eigen, coeff_cur[count_blk_size : count_blk_size + block_len[i][0]]))

            if switch in ['S_noGP', 'S_G', 'S_GP']:

                if block_type[i] == 0:

                    # Interpolation of liquid part V to the same degree as solid part V.
                    #V: Spheroidal mode’s tangential displacement. 
                    f = interp1d(   lib.sqzx(x_V[:, count_thick[i] : count_thick[i + 1]], thickness[i], order_V),
                                    coeff_cur[count_blk_size + block_len[i][0] : count_blk_size + block_len[i][0] + block_len[i][1]], 'cubic')
                    V_inter = f(    lib.sqzx(x[:, count_thick[i] : count_thick[i + 1]], thickness[i], order))
                    V_eigen = np.hstack((V_eigen, V_inter))

                else:

                    V_eigen = np.hstack((V_eigen, coeff_cur[count_blk_size + block_len[i][0] : count_blk_size + block_len[i][0] + block_len[i][1]]))

            xx = np.hstack((xx, lib.sqzx(x[:, count_thick[i] : count_thick[i + 1]], thickness[i], order)))
            count_blk_size = count_blk_size + np.sum(block_len[i])
                
        U_eigen = U_eigen/scale
        
        if switch in ['S_noGP', 'S_G', 'S_GP']:

            V_eigen = V_eigen/(np.sqrt(l*(l+1))*scale)
        
        if save:

            # Write eigenvalue.
            with open(path_eigenvalues, 'a') as f_out:

                f_out.write('{:>10d} {:>10d} {:>16.12f}\n'.format(n, l, eigen[n]))

            # Write eigenfunction. 
            file_eigenfunc = '{:>05d}_{:>05d}.npy'.format(n, l)
            path_eigenfunc = os.path.join(dir_eigenfunc, file_eigenfunc)
            if switch in ['S_noGP', 'S_G', 'S_GP']:
                
                # Create columns in output array for gradient and potential
                # which are calculated later.
                Up = np.zeros(U_eigen.shape)
                Vp = np.zeros(U_eigen.shape)
                P = np.zeros(U_eigen.shape)
                Pp = np.zeros(U_eigen.shape)

                out_arr = np.array([1000.0*xx, U_eigen, V_eigen, Up, Vp, P, Pp])

            elif switch in['R_noGP', 'R_G', 'R_GP']:

                out_arr = np.array([1000.0*xx, U_eigen])

            np.save(path_eigenfunc, out_arr)
    
    # Select values to return.
    if (l == 1) and (nmin < 2):
        
        n_min_r = 2

    else:

        n_min_r = nmin

    eigen = eigen[nmin : nmax + 1]

    return eigen, n_min_r 

# All modes. ------------------------------------------------------------------
def prep_fem(model_path, dir_output, num_elmt, switch): 

    # Set finite-element order for various parameters.
    # Some parameters are only required for certain cases, and they are 
    # set to None if not required.
    order = 2
    order_p = set_if_needed(3, switch, ['R_noGP', 'R_G', 'R_GP', 'S_noGP', 'S_G', 'S_GP'])
    order_V = set_if_needed(1, switch, ['R_GP', 'S_noGP', 'S_G', 'S_GP'])
    order_P = set_if_needed(2, switch, ['R_GP', 'S_GP'])

    # Load model data.
    model = load_model(model_path)
    # Unpack.
    r   = model['r']
    rho = model['rho']
    vp = model['v_p']
    vs = model['v_s']
    # Convert to units used internally by Ouroboros.
    r   = r/1.0E6   # Million meters.
    rho = rho/1.0E3 # g/cm3.
    vp = vp/1.0E3 # km/s.
    vs = vs/1.0E3 # km/s.
    # Calculate bulk and shear moduli (units of GPa).
    mu = rho*(vs**2.0)
    ka = rho*((vp**2.0) - (4.0/3.0)*(vs**2.0))
    
    # brk_num:      Records the position of solid-liquid boundary.
    # layers:       Number of 'layers' (continuous regions of solid or fluid).
    # Thickness:    Thickness of each layer.
    brk_num = [0]
    layers = 1
    thickness = []
    
    # Keep track of number of essential spectrum and block type
    essen       = set_if_needed(0,  switch, ['R_noGP', 'R_G', 'R_GP', 'S_noGP', 'S_G', 'S_GP'])
    block_type  = set_if_needed([], switch, ['R_noGP', 'R_G', 'R_GP', 'S_noGP', 'S_G', 'S_GP'])

    # Loop through the points in the input model.
    for i in range(model['n_layers'] - 1):

        # Find solid-fluid boundaries. 
        if vs[i]*vs[i+1]==0 and (vs[i]+vs[i+1])!=0:
            
            # Update the counter variables.
            brk_num.append(i+1)
            layers = layers+1
            temp_thick = num_elmt*(brk_num[-1]-brk_num[-2])/model['n_layers']
            thickness.append(round(temp_thick))

            # Get block type and number of essential spectrum. 
            if switch in ['R_noGP', 'R_G', 'R_GP', 'S_noGP', 'S_G', 'S_GP']:

                if vs[i]==0:
                    
                    block_type.append(0)

                    if switch in ['R_GP', 'S_GP']:

                        essen = essen + thickness[-1]*order_V

                    else:

                        essen = essen + thickness[-1]
                else:

                    block_type.append(1)

    thickness.append(num_elmt-int(np.sum(thickness)))

    # Get final block type and essen.
    if switch in ['R_noGP', 'R_G', 'R_GP', 'S_noGP', 'S_G', 'S_GP']:

        if vs[-1] == 0:

            block_type.append(0)

            if switch in ['R_GP', 'S_GP']:
                
                essen = essen + thickness[-1]*order_V

            else:

                essen = essen + thickness[-1]
                
        else:

            block_type.append(1)

    brk_num.append(model['n_layers'])
    #layers = len(brk_num)-1
    VX = []
    #radius of fluid-solid boundaries
    brk_radius = [0]
    #count_thick provide index of boundaries 
    count_thick = [0]
    for i in range(layers):
        brk_radius.append(r[brk_num[i+1]-1])
        count_thick.append(count_thick[-1]+thickness[i])
        temp_VX = lib.mantlePoint_equalEnd(r[brk_num[i]:brk_num[i+1]],thickness[i]+1,brk_radius[i],brk_radius[i+1])
        VX = np.hstack((VX,temp_VX))
    
    new_rho = lib.model_para_inv(r,rho,VX) #interpolate rho in nodal points
    new_mu = lib.model_para_inv(r,mu,VX) #interpolate mu in nodal points
    new_ka = lib.model_para_inv(r,ka,VX) #interpolate ka in nodal points
    new_alpha = new_ka-2/3*new_mu #alpha and beta are just parameters
    new_beta = 1/(new_ka+4/3*new_mu)

    if switch in ['R_G', 'R_GP', 'S_G', 'S_GP']:

        new_rho_p = lib.model_para_prime_inv(r,rho,VX) #interpolate prime of rho in nodal points
        #model.add_rho_p(new_rho_p)
    
    VX = lib.remDiscon(VX,0)
    
    # Read in Mesh
    #[Nv, K, EToV] = setup.GenElement(VX) #results used in setup.StartUp
    
    # Initialize solver and construct grid and metric
    va = VX[0:-1]
    vb = VX[1:]-VX[0:-1]
    x, J, rx, invV, Dr = setup.StartUp(order,va,vb)

    # p
    if switch in ['R_noGP', 'R_G', 'R_GP', 'S_noGP', 'S_G', 'S_GP']:

        invV_p,Dr_p,x_p = setup.StartUp4pressure(order_p,va,vb)
        #model.add_xp(x_p)

    else:

        invV_p  = None
        Dr_p    = None
        x_p     = None

    # V
    if switch in ['S_noGP', 'S_G', 'S_GP']:

        invV_V,Dr_V,x_V = setup.StartUp4V(order_V,va,vb)
        #model.add_xV(x_V)  

    else:

        invV_V  = None
        Dr_V    = None
        x_V     = None

    # P
    if switch in ['R_GP', 'S_GP']:

        invV_P, Dr_P, x_P = setup.StartUp4Perturbation(order_P,va,vb)
        #model.add_xP(x_P)

    else:

        invV_P  = None
        Dr_P    = None
        x_P     = None

    model = setup.model_para(new_mu,new_ka,new_rho,x,new_alpha,new_beta,J,rx)

    if switch in ['R_G', 'R_GP', 'S_G', 'S_GP']:

        model.add_rho_p(new_rho_p)

    # p
    if switch in ['R_noGP', 'R_G', 'R_GP', 'S_noGP', 'S_G', 'S_GP']:

        model.add_xp(x_p)

    # V
    if switch in ['S_noGP', 'S_G', 'S_GP']:

        model.add_xV(x_V)  

    # P
    if switch in ['R_GP', 'S_GP']:

        model.add_xP(x_P)

    mkdir_if_not_exist(dir_output)
    if switch != 'T':

        # Define eigenvalue output file and delete if it already exists.
        path_eigenvalues = os.path.join(dir_output, 'eigenvalues.txt')
        rm_file_if_exist(path_eigenvalues)

        # Create output directories if they don't exist.
        dir_eigenfunc = os.path.join(dir_output, 'eigenfunctions')
        mkdir_if_not_exist(dir_eigenfunc)

    else:
        
        path_eigenvalues_list = []
        dir_eigenfunc_list = []
        for i in range(layers):

            if vs[brk_num[i]] != 0:

                path_eigenvalues_i = os.path.join(dir_output, 'eigenvalues_{:03d}.txt'.format(i//2))
                rm_file_if_exist(path_eigenvalues_i)
                path_eigenvalues_list.append(path_eigenvalues_i)

                # Create output directories if they don't exist.
                dir_eigenfunc_i = os.path.join(dir_output, 'eigenfunctions_{:03d}'.format(i//2))
                mkdir_if_not_exist(dir_eigenfunc_i)
                dir_eigenfunc_list.append(dir_eigenfunc_i)

        # Rename the lists so they can be passed back to the main function in the
        # same way.
        path_eigenvalues = path_eigenvalues_list
        dir_eigenfunc = dir_eigenfunc_list

    return (model, vs, count_thick, thickness, essen,
            invV, invV_p, invV_V, invV_P,
            order, order_p, order_V, order_P,
            Dr, Dr_p, Dr_V, Dr_P,
            x, x_V, x_P, VX,
            rho, r,
            block_type, brk_radius, brk_num, layers,
            dir_eigenfunc, path_eigenvalues)

# Generic utilities. ----------------------------------------------------------
def rm_file_if_exist(path):

    if os.path.exists(path):

        os.remove(path)

def set_if_needed(value, switch, switch_list):

    if switch in switch_list:

        return value 

    else:

        return None

