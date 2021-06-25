# References:
# [1] GMIG Report (2020) Vol. 5 Report 12, pp 263--289.
# [2] Yuen and Peltier (1982) 'Normal modes of the viscoelastic Earth'
#       Geophys. J. R. astr. Soc. 69, pp 495--526.
#using LinearAlgebra
using NonlinearEigenproblems
#using NPZ
using Printf
#using DelimitedFiles
#using PyCall
include("lib.jl")
include("common.jl")

order_V = 1
order = 2

function save_toroidal_eigvecs_eigvals(eigvals, xx, eigvecs, A2, nep_h, dir_output, i_toroidal, l, num_eigen)

    # Get number of samples.
    size_r = size(xx)[1]

    # Convert from rad/s to mHz.
    fre = (eigvals / (2.0 * pi)) * 1000.0
    
    # Save the eigenvalues.
    name_eigvals = @sprintf("eigenvalues_%03d_%05d.txt", i_toroidal, l)
    path_eigvals = joinpath(dir_output, name_eigvals)
    open(path_eigvals, "w") do f_eigval

        for i = 1 : num_eigen

            write(f_eigval, string(real(fre[i]), " ", imag(fre[i]), "\n"))

        end

    end

    # Save the eigenvectors.
    name_eigvecs = @sprintf("eigenfunctions_%03d", i_toroidal)
    dir_eigvecs = joinpath(dir_output, name_eigvecs)
    mkdir_if_not_exist(dir_eigvecs)
    k = sqrt(l * (l + 1.0))
    for i = 1 : num_eigen

        W_eigen = eigvecs[1 : size_r, i]

        # Normalise eigenvector.
        # GMIG report 2020, eq. 3.4.
        scale = sqrt(transpose(W_eigen) * A2 * W_eigen -
                     (1.0 / (2.0 * eigvals[i])) * transpose(W_eigen) *
                        compute_Mder(nep_h, eigvals[i], 1) * W_eigen)
        W_eigen = (W_eigen / scale)
        # 
        scaleW = (k * fre[i] * 2.0 * pi)
        W_eigen = (W_eigen / scaleW)

        # Get output path.
        n = i - 1
        name_eigvec = @sprintf("eigvec_%05d_%05d.txt", n, l)
        path_eigvec = joinpath(dir_eigvecs, name_eigvec)
        #
        open(path_eigvec, "w") do f_eigvec

            # Loop over node points.
            for j = 1 : size_r

                write(f_eigvec,
                      string(@sprintf("%3.15E", xx[j]), " ",
                             @sprintf("%3.15E", real(W_eigen[j])), " ",
                             @sprintf("%3.15E", imag(W_eigen[j])), "\n"))

            end

        end

    end

end

# Finds the toroidal modes in anelastic case by solving REP (rational
# eigenvalue problem).
function toroidal_rep(args)

    # Load anelastic parameters.
    path_input_anelastic = args[1]
    anelastic_params = read_input_anelastic(path_input_anelastic)

    # Read files generated by Python script, relating to matrices.
    dir_output = args[2]
    i_toroidal = parse(Int64, args[3])
    dir_numpy = joinpath(dir_output, @sprintf("numpy_%03d", i_toroidal))
    dir_julia = joinpath(dir_output, @sprintf("julia_%03d", i_toroidal))
    #
    lines = readlines(joinpath(dir_numpy, "parameter_T.txt"))
    l = parse.(Int64,(lines[1]))
    Ki = parse.(Int64,(lines[2]))
    dimension = parse.(Int64,(lines[3]))
    #
    #A   = npzread(joinpath(dir_numpy, "A.npy"))
    Mmu = npzread(joinpath(dir_numpy, "Mmu.npy"))
    mu  = npzread(joinpath(dir_numpy, "mu.npy"))
    A2  = npzread(joinpath(dir_numpy, "A2.npy"))
    xx  = npzread(joinpath(dir_numpy, "xx.npy"))
    #mu2 = 75*mu

    # constant term
    temp_A0 = zeros(dimension, dimension)

    # Unpack variables.
    # eig_start is converted from mHz to rad/s.
    # nu is converted from SI units to Ouroboros units.
    eig_start_rad_per_s = (anelastic_params["eig_start_mHz"] * 1.0E-3) * (2.0 * pi)
    @printf("eig_start_rad_per_s %.6f (%.3f mHz) \n", eig_start_rad_per_s,
            anelastic_params["eig_start_mHz"])
    nu1 = anelastic_params["nu1"] * 1.0E-9
    num_eigen = anelastic_params["n_eigs"] 

    if anelastic_params["model_type"] == "burgers_uniform"

        nu2 = anelastic_params["nu2"] * 1.0E-9
        mu2 = anelastic_params["mu2_factor"] * mu

    end

    # Define linear term in polynomial eigenvalue problem.
    # This is generally zero, however when the individual elements consist of
    # REPs, these contain (by default) a unit linear term, therefore
    # in those cases we must add Ki unit linear terms to cancel these
    # terms out.
    #
    # Elastic case (0) and Kelvin case (-2) have no REP terms.
    if (anelastic_params["model_type"] == 0) || (anelastic_params["model_type"] == -2)

        # This is a zero matrix.
        A1 = temp_A0

    # All other cases have REP terms.
    else

        A1 = (Ki * Matrix(1.0I, dimension, dimension))

    end

    xx = xx*1000
    size_r = size(xx)[1]

    nep = PEP([temp_A0, A1, A2])

    # Loop over elements.
    for k = 1:Ki

        # Get the mu matrix for this element.
        # This is zero except within the block:
        # 0 0 0
        # 0 M 0
        # 0 0 0
        @printf("Element %4d of %4d\n", k, Ki)
        temp_A1 = Mmu[k, :, :]

        # Elastic case.
        if anelastic_params["model_type"] == 0

            # The elastic case is a polynomial of order 0, in other words
            # a constant; the elastic case is not frequency-dependent.
            temp_rep = PEP([temp_A1])
            
        # Kelvin solid.
        elseif anelastic_params["model_type"] == -2
                
            # hrmd 2021-06-20: Need reference for this form.
            temp_rep = PEP([temp_A1/mu[k]*mu2[k], temp_A1/mu[k]*nu2*im])

        # The remaining rheologies can be expressed as REPs (rational
        # eigenvalue problems).
        else

            # Maxwell solid with uniform viscosity.
            #elseif body_type == -1
            if anelastic_params["model_type"] == "maxwell_uniform"
                
                # Get expression for roots (zeros of eq. 2.27 in ref. [1]).
                temp_roots = [0.0]

                # Get expression for poles (zeros of eq. 2.28 in ref. [1]).
                temp_poles = [mu[k] / nu1]

            # Standard linear solid.
            elseif anelastic_params["model_type"] == 1

                # Get expression for roots (zeros of numerator of
                # eq. 13c in ref. [2]).
                # hrmd 2021-06-21: Missing negative sign?
                temp_roots = [mu2[k] / nu2]

                # Get expression for poles (zeros of denominator of eq. 13c
                # in ref. [2]).
                temp_poles = [(mu[k] + mu2[k]) / nu2]

            # Burger's solid with uniform viscosities.
            #elseif anelastic_params["model_type"] == 2
            elseif anelastic_params["model_type"] == "burgers_uniform" 
                
                # Get expression for roots (zeros of eq. 2.32 in ref. [1]). 
                temp_roots = [0.0, (mu2[k] / nu2)]

                # Get expression for poles (zeros of eq. 2.33 in ref. [1]).
                b = (mu[k] / nu1) + (mu[k] / nu2) + (mu2[k] / nu2)
                ac = (mu[k] / nu1)^2.0 + (mu[k] / nu2)^2 + (mu2[k] / nu2)^2.0 +
                        2.0 * (mu[k]^2.0 + (mu[k] * mu2[k])) / (nu1 * nu2) + 
                        2.0 * (mu[k] * mu2[k]) / (nu2^2.0)
                x1 = (b + sqrt(ac)) / 2.0
                x2 = (b - sqrt(ac)) / 2.0
                temp_poles = [x1, x2]

            end

            # Create the rational eigenvalue problem for this element.
            # Not the factor of i (imaginary unit)
            temp_rep = REP([temp_A0, temp_A1],
                           im * temp_roots, im * temp_poles)

        end

        # Add the rational eigenvalue problem for this element to the overall
        # eigenvalue problem.
        nep = SumNEP(nep, temp_rep)

    end

    # Build operator ∫𝜑*H*rho*𝜑 from original nep ω^2*∫𝜑*rho*𝜑 - ∫𝜑*H*rho*𝜑 = 0
    # This is used for calculating derivative during normalisation.
    nep_h = SumNEP(nep, PEP([-temp_A0, -temp_A0, -A2]))

    # Solve non-linear eigenvalue problem.
    println("Trying to solve eigenvalue problem...")
    method = "iar"
    tol = 1.0E-13
    if method == "iar"

        eigvals, eigvecs = iar( nep, maxit = 100 , σ = eig_start_rad_per_s,
                                neigs = num_eigen,
                                logger = 1,
                                tol = tol)

    #elseif method == "tiar"

    #    eigvals, eigvecs = tiar( nep, maxit = 100 , σ = eig_start_rad_per_s,
    #                            neigs = num_eigen,
    #                            logger = 1,
    #                            tol = 1.0E-5)

    #elseif method == "jd_effenberger"

    #    eigvals, eigvecs = jd_effenberger(nep, maxit = 10,
    #                            neigs = num_eigen,
    #                            logger = 1)

    else
        
        error(@sprintf("method %s not recognised", method))

    end
    
    # Save the eigenvectors and eigenvalues.
    save_toroidal_eigvecs_eigvals(eigvals, xx, eigvecs, A2, nep_h, dir_julia,
                                  i_toroidal, l, num_eigen)

end

# Run with command-line arguments.
toroidal_rep(ARGS)
