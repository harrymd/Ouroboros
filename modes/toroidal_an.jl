#using LinearAlgebra
using NonlinearEigenproblems
#using NPZ
using Printf
#using DelimitedFiles
#using PyCall
include("lib.jl")

order_V = 1
order = 2

# Makes a directory if it doesn't already exist. ------------------------------
function mkdir_if_not_exist(path_)

    if isdir(path_) == false
        
        @printf("Making directory %s", path_)
        mkdir(path_)

    end

end

# Reads the anelastic input file specified in the modes input file.
function read_input_anelastic(path_input_anelastic)
    
    println(path_input_anelastic)
    anelastic_params = Dict()

    open(path_input_anelastic) do f
        
        anelastic_params["model_type"] = split(readline(f), ' ')[2]
        anelastic_params["n_eigs"] = parse(Int64, split(readline(f), ' ')[2])
        anelastic_params["eig_start_mHz"] = parse(Float64, split(readline(f), ' ')[2])

        if anelastic_params["model_type"] == "maxwell_uniform"
            
            anelastic_params["nu"] = parse(Float64, split(readline(f), ' ')[2])

        end

    end

    return anelastic_params

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
    nu1 = anelastic_params["nu"] * 1.0E-9
    num_eigen = anelastic_params["n_eigs"] 

    #Take care of the lambda*I term in rep
    if anelastic_params["model_type"] == 0

        A1 = temp_A0
        print("Body type: elastic\n")

    elseif anelastic_params["model_type"] == -2

        A1 = temp_A0
        print("Body type: kelvin\n")
        #A1 = sum(Ki)*Matrix(1.0I, dimension, dimension)

    else

        A1 = Ki*Matrix(1.0I, dimension, dimension)
        print("Body type: others\n")

    end

    xx = xx*1000
    size_r = size(xx)[1]

    nep = PEP([temp_A0, A1, A2])

    # Loop over elements.
    for k = 1:Ki

        @printf("Element %4d of %4d\n", k, Ki)
        temp_A1 = Mmu[k,:,:]
        if anelastic_params["model_type"] == 0

            #elastic case
            #temp_rep = REP([temp_A0,temp_A1],[0],[0])
            temp_rep = PEP([temp_A1])

        #elseif body_type == -1
        elseif anelastic_params["model_type"] == "maxwell_uniform"

            #maxwell solid
            temp_rep = REP([temp_A0, temp_A1], [0], [(im * mu[k]) / nu1])


        elseif anelastic_params["model_type"] == -2

            #kelvin solid
            temp_rep = PEP([temp_A1/mu[k]*mu2[k],temp_A1/mu[k]*nu2*im])

        elseif anelastic_params["model_type"] == 1

            #standard linear solid
            temp_pole = (mu[k]+mu2[k])/nu2
            temp_rep = REP([temp_A0,temp_A1],[im*mu2[k]/nu2],[im*temp_pole])

        elseif anelastic_params["model_type"] == 2

            #anelastic case, burger's body
            b = mu[k]/nu1+mu[k]/nu2+mu2[k]/nu2
            ac = (mu[k]/nu1)^2+(mu[k]/nu2)^2+(mu2[k]/nu2)^2+2*(mu[k]^2+mu[k]*mu2[k])/(nu1*nu2)+2*mu[k]*mu2[k]/(nu2^2)
            x1 = (b+sqrt(ac))/2
            x2 = (b-sqrt(ac))/2
            temp_rep = REP([temp_A0,temp_A1],[0,im*mu2[k]/nu2],[-x1/im,-x2/im])

        end

        nep = SumNEP(nep, temp_rep)

    end

    # Build operator ∫𝜑*H*rho*𝜑 from original nep ω^2*∫𝜑*rho*𝜑 - ∫𝜑*H*rho*𝜑 = 0
    nep_h = SumNEP(nep, PEP([-temp_A0, -temp_A0, -A2]))

    # Solve non-linear eigenvalue problem.
    println("Trying to solve eigenvalue problem...")
    method = "iar"
    if method == "iar"

        eigvals, eigvecs = iar( nep, maxit = 100 , σ = eig_start_rad_per_s,
                                neigs = num_eigen,
                                logger = 1,
                                tol = 1.0E-5)

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

    # Sort the eigenvalues by their real part#, in decreasing order.
    #p = sortperm(real(eigvals), rev = true)
    p = sortperm(real(eigvals))
    eigvals = eigvals[p]
    eigvecs = eigvecs[1 : end, p]

    # Convert from rad/s to mHz.
    fre = (eigvals / (2.0 * pi)) * 1000.0
    
    # Save the eigenvalues.
    name_eigvals = @sprintf("eigenvalues_%03d_%05d.txt", i_toroidal, l)
    path_eigvals = joinpath(dir_output, name_eigvals)
    open(path_eigvals, "w") do f_eigval

        for i = 1 : num_eigen

            write(f_eigval,string(l," ",real(fre[i])," ",imag(fre[i]),"    ",real(eigvals[i])," ",imag(eigvals[i]),"\n"))

        end

    end

    # Save the eigenvectors.
    name_eigvecs = @sprintf("eigenfunctions_%03d", i_toroidal)
    dir_eigvecs = joinpath(dir_output, name_eigvecs)
    mkdir_if_not_exist(dir_eigvecs)
    for i = 1 : num_eigen

        W_eigen = eigvecs[1 : size_r, i]

        # Normalise eigenvector.
        # GMIG report, eq. 3.4.
        scale = sqrt(transpose(W_eigen)*A2*W_eigen-0.5/eigvals[i]*transpose(W_eigen)*compute_Mder(nep_h,eigvals[i],1)*W_eigen)
        W_eigen = W_eigen/scale
        # 
        scaleW = sqrt(l*(l+1))*(fre[i] * 2.0 * pi)
        W_eigen = W_eigen/scaleW

        # Get output path.
        if l == 1

            n = i

        else

            n = i - 1

        end
        name_eigvec = @sprintf("eigvec_%05d_%05d.txt", n, l)
        path_eigvec = joinpath(dir_eigvecs, name_eigvec)
        #
        open(path_eigvec, "w") do f_eigvec

            # Loop over node points.
            for j = 1:size_r

                write(f_eigvec,
                      string(@sprintf("%3.15E", xx[j]), " ",
                             @sprintf("%3.15E", real(W_eigen[j])), " ",
                             @sprintf("%3.15E", imag(W_eigen[j])), "\n"))

            end

        end

    end

end

# Run with command-line arguments.
toroidal_rep(ARGS)
