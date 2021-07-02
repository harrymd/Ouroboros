import argparse
import os

# Temporary.
import matplotlib.pyplot as plt

import numpy as np
import scipy

from Ouroboros.common import (get_Ouroboros_out_dirs, read_Ouroboros_input_file)

def read_input_anelastic(path_input_anelastic):

    anelastic_params = dict()

    with open(path_input_anelastic, 'r') as in_id:

        anelastic_params["model_type"] = in_id.readline().split()[-1]
        anelastic_params["n_eigs"] = int(in_id.readline().split()[-1])
        anelastic_params["eig_start_mHz"] = float(in_id.readline().split()[-1])

        if anelastic_params["model_type"] == "maxwell_uniform":
            
            anelastic_params["nu1"] = float(in_id.readline().split()[-1])
        
        elif anelastic_params["model_type"] == "burgers_uniform":

            anelastic_params["nu1"] = float(in_id.readline().split()[-1])
            anelastic_params["nu2"] = float(in_id.readline().split()[-1])
            anelastic_params["mu2_factor"] = float(in_id.readline().split()[-1])

        else:
            
            raise ValueError

    return anelastic_params

def rank_revealing_decomp(A, n, Np, K, N):
    
    # Set tolerance for truncation of eigendecomposition.
    tol = 1.0E-10

    # Loop over elements.
    for i in range(K):

        # Get index of start of block for this element, and then
        # extract the block.
        i_n = (i * n)
        Ai = A[i, i_n : (i_n + Np), i_n : (i_n + Np)]

        # Find the eigenvalues (lam) and eigenvectors (Q) of the block.
        # It is a symmetric matrix so we use eigh instead of eig.
        lam, Q = np.linalg.eigh(Ai)
        
        # Find the inverse of the eigenvector matrix.
        Qinv = np.linalg.inv(Q)

        # Find the non-zero eigenvalues.
        # The number of non-zero eigenvalues is the rank of the matrix.
        j = np.where(np.abs(lam) > tol)[0]
        if i == 0:

            rank = len(j)
            assert (rank == n)

            # Prepare output arrays.
            L = np.zeros((K, N, rank))
            U = np.zeros((K, N, rank))

        else:

            rank_i = len(j)
            assert (rank_i == rank)

        # Truncate the eigenvectors and eigenvalues.
        lam     = lam[j]
        Q       = Q[:, j]
        Qinv    = Qinv[j, :]

        # Construct matrices L and U which are a full-column-rank decomposition
        # of the matrix A[i, ...], so that A[i, ...] = (L @ U.T)
        # This is based on the eigendecomposition
        # Ai = (Q @ np.diag(lam) @ Q_inv)
        # Arbitrarily, the eigenvalue terms are absorbed into L instead of U.
        li = (Q @ np.diag(lam))
        Li = np.zeros((N, rank))
        Li[i_n : (i_n + Np), :] = li
        # 
        ui = Qinv.T
        Ui = np.zeros((N, rank))
        Ui[i_n : (i_n + Np), :] = ui 

        # Check that the matrices L and U have full column rank. 
        column_rank_L = np.linalg.matrix_rank(Li, tol = tol)
        column_rank_U = np.linalg.matrix_rank(Ui, tol = tol)
        assert column_rank_L == Li.shape[1]
        assert column_rank_U == Ui.shape[1]
        
        # Check the quality of the reconstruction.
        A_reconstructed = (Li @ (Ui.T))
        recon_error = np.linalg.norm(A[i, ...] - A_reconstructed, ord = 2)
        print('Element {:>3d} of {:>3d}, reconstruction error: {:>.2e}'
                .format(i + 1, K, recon_error))
        assert recon_error < tol

        # Store in output arrays.
        L[i, ...] = Li
        U[i, ...] = Ui

    return L, U

def minimal_realisation_wrapper(mu, K, mu2_factor, nu1, nu2):

    # Prepare output arrays.
    ai  = np.zeros((K, 2, 1), dtype = np.complex)
    bi  = np.zeros((K, 2, 1))
    Ci  = np.zeros((K, 2, 2), dtype = np.complex)
    Di  = np.zeros((K, 2, 2))
    
    # Get mu2 parameter.
    mu2 = (mu2_factor * mu)

    # Loop over elements.
    for k in range(K):

        # Get expression for roots (zeros of eq. 2.32 in ref. [1]). 
        r1, r2 = -1.0j * np.array([0.0, -(mu2[k] / nu2)])

        # Get expression for poles (zeros of eq. 2.33 in ref. [1]).
        # Get expression for poles (zeros of eq. 2.33 in ref. [1]).
        # Use quadratic formula.
        a = (nu1 * nu2) / (mu[k] * mu2[k]) 
        b = (nu1 / mu[k]) + (nu1 / mu2[k]) + (nu2 / mu2[k])
        c = 1.0
        #
        det = (b ** 2.0) - (4.0 * a * c)
        sqrt_det = np.sqrt(det)
        #
        p1 = (-b - sqrt_det) / (2.0 * a)
        p2 = (-b + sqrt_det) / (2.0 * a)
        #
        p1 = (p1 * -1.0j)
        p2 = (p2 * -1.0j)

        # Calculate components of minimal realisation.
        #ai[k, :, :], bi[k, :, :], Ci[k, :, :], Di[k, :, :] = \
        aik, bik, Cik, Dik = \
            minimal_realisation(p1, p2, r1, r2)
        ai[k, :, :] = aik[:, np.newaxis]
        bi[k, :, :] = bik[:, np.newaxis]
        Ci[k, :, :] = Cik
        Di[k, :, :] = Dik

    return ai, bi, Ci, Di

def minimal_realisation(p1, p2, r1, r2):

    alpha   = (p1 + p2 - r1 - r2)
    beta    = (p1 * p2) + (r1 * r2)


    a = np.array([-beta, alpha])
    b = np.array([0.0, 1.0])

    C = np.array([[0.0, -1.0], [(p1 * p2), -1.0 * (p1 + p2)]])
    D = -1.0 * np.identity(2)

    return a, b, C, D

def plot_black_white_imshow(A):

    fig = plt.figure()
    ax  = plt.gca()
    
    i1, i2 = A.shape
    G = np.zeros((i1, i2, 3))
    G[np.abs(A) > 0.0] = [1.0, 1.0, 1.0]

    ax.imshow(G)
    plt.show()

    return

def build_kronecker_matrices(ai, bi, Ci, Di, Li, Ui):

    # Infer integers.
    rank = Li.shape[-1]
    K = Ci.shape[0]

    Ir = np.identity(rank)
    
    L_list = []
    U_list = []
    C_list = []
    D_list = []
    for k in range(K):
        
        I_kron_a = np.kron(Ir, ai[k, :])
        L_list.append(Li[k, :, :] @ I_kron_a.T)

        I_kron_b = np.kron(Ir, bi[k, :])
        U_list.append(Ui[k, :, :] @ I_kron_b.T)

        C_list.append(np.kron(Ir, Ci[k, ...]))

        D_list.append(np.kron(Ir, Di[k, ...]))


    L = np.concatenate(L_list, axis = 1)
    U = np.concatenate(U_list, axis = 1)

    C = scipy.linalg.block_diag(*C_list)
    D = scipy.linalg.block_diag(*D_list)
    
    return L, U, C, D

def build_linearised_matrices(B, E, L, U, C, D):

    N = E.shape[-1]

    E_sum = np.sum(E, axis = 0)
    
    zeros_NN = np.zeros((N, N))
    ident_NN = np.eye(N)
    zeros_Nm = np.zeros(L.shape)

    calA = np.block([   [ zeros_NN,     -E_sum,     L],
                        [-ident_NN,     zeros_NN,   zeros_Nm],
                        [ zeros_Nm.T,   U.T,        C]])

    calB = scipy.linalg.block_diag(*[B, -ident_NN, D])
    
    tt = np.linalg.det(B)
    print(tt)
    import sys
    sys.exit()
    t = np.linalg.det(calB)
    print(t)
    print(calA.shape)
    print(calB.shape)

    plot_black_white_imshow(calB)
    import sys
    sys.exit()

    
    return calA, calB

def main():

    # Parse input arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_input_file", help = "File path (relative or absolute) to Ouroboros input file.")
    #
    input_args = parser.parse_args()
    Ouroboros_input_file = input_args.path_to_input_file

    # Read the input files.
    Ouroboros_info = read_Ouroboros_input_file(Ouroboros_input_file)
    anelastic_info = read_input_anelastic(Ouroboros_info['path_atten'])

    # Unpack anelastic info.
    # Viscosities are convertred from SI units to Ouroboros units.
    mu2_factor  = anelastic_info['mu2_factor']
    nu1         = anelastic_info['nu1'] * 1.0E-9
    nu2         = anelastic_info['nu2'] * 1.0E-9

    # Find output directories.
    mode_type = 'T'
    i_toroidal = 0
    dir_model, dir_run, dir_g, dir_type = \
            get_Ouroboros_out_dirs(Ouroboros_info, mode_type)
    dir_numpy = os.path.join(dir_type, 'numpy_{:>03d}'.format(i_toroidal))

    # Load arrays.
    E  = np.load(os.path.join(dir_numpy, 'Mmu.npy'))
    mu = np.load(os.path.join(dir_numpy, 'mu.npy'))
    B  = np.load(os.path.join(dir_numpy, 'A2.npy'))

    # Set integer variables.
    # n     Polynomial order of elements.
    # Np    Number of nodal points per element, equal to (n + 1).
    # K     Nuber of elements.
    # N     Size of matrix problem, equal to (n  * K) + 1.
    # 
    #       
    n = 2
    Np = (n + 1)
    N = E.shape[-1]
    K = (N - 1) // 2

    # Calculate minimal realization.
    ai, bi, Ci, Di = minimal_realisation_wrapper(mu, K, mu2_factor, nu1, nu2)

    print('ai', ai.shape)
    print('bi', bi.shape)
    print('Ci', Ci.shape)
    print('Di', Di.shape)

    # Do rank-revealing decomposition.
    Li, Ui = rank_revealing_decomp(E, n, Np, K, N)

    print('Li', Li.shape)
    print('Ui', Ui.shape)

    # Build Kronecker representations.
    L, U, C, D = build_kronecker_matrices(ai, bi, Ci, Di, Li, Ui)

    print('L', L.shape)
    print('U', U.shape)
    print('C', C.shape)
    print('D', D.shape)

    # Build linearised matrices.
    calA, calB = build_linearised_matrices(B, E, L, U, C, D)

    return

if __name__ == '__main__':

    main()
