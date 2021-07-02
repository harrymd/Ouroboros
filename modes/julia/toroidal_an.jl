# References:
# [1] GMIG Report (2020) Vol. 5 Report 12, pp 263--289.
# [2] Yuen and Peltier (1982) 'Normal modes of the viscoelastic Earth'
#       Geophys. J. R. astr. Soc. 69, pp 495--526.
using NonlinearEigenproblems
using Printf
include("lib.jl")
include("common.jl")

order_V = 1
order = 2

function save_toroidal_eigvecs_eigvals(eigvals, xx, eigvecs, A2, nep_h, dir_output, i_toroidal, l, num_eigen, poles, roots)

    # Get number of samples.
    size_r = size(xx)[1]

    # Convert from rad/s to mHz.
    rad_s_to_mHz = (1000.0 / (2.0 * pi))
    fre = (eigvals * rad_s_to_mHz)
    poles = (poles * rad_s_to_mHz)
    roots = (roots * rad_s_to_mHz)
    
    # Save the eigenvalues.
    name_eigvals = @sprintf("eigenvalues_%03d_%05d.txt", i_toroidal, l)
    path_eigvals = joinpath(dir_output, name_eigvals)
    open(path_eigvals, "w") do f_out

        for i = 1 : num_eigen

            write(f_out, @sprintf("%+19.12e %+19.12e\n", real(fre[i]), imag(fre[i])))

        end

    end

    # Save the roots.
    num_roots = length(roots)
    name_eigvals = @sprintf("roots_%03d_%05d.txt", i_toroidal, l)
    path_eigvals = joinpath(dir_output, name_eigvals)
    open(path_eigvals, "w") do f_out

        for i = 1 : num_roots

            write(f_out, @sprintf("%+19.12e %+19.12e\n", real(roots[i]), imag(roots[i])))

        end

    end
    
    # Save the poles.
    num_poles = length(poles)
    name_eigvals = @sprintf("poles_%03d_%05d.txt", i_toroidal, l)
    path_eigvals = joinpath(dir_output, name_eigvals)
    open(path_eigvals, "w") do f_out

        for i = 1 : num_poles

            write(f_out, @sprintf("%+19.12e %+19.12e\n", real(poles[i]), imag(poles[i])))

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

function read_numpy_files(dir_output, i_toroidal)

    dir_numpy = joinpath(dir_output, @sprintf("numpy_%03d", i_toroidal))
    dir_julia = joinpath(dir_output, @sprintf("julia_%03d", i_toroidal))
    #
    lines = readlines(joinpath(dir_numpy, "parameter_T.txt"))
    l = parse.(Int64,(lines[1]))
    Ki = parse.(Int64,(lines[2]))
    dimension = parse.(Int64,(lines[3]))
    #
    Mmu = npzread(joinpath(dir_numpy, "Mmu.npy"))
    mu  = npzread(joinpath(dir_numpy, "mu.npy"))
    A2  = npzread(joinpath(dir_numpy, "A2.npy"))
    xx  = npzread(joinpath(dir_numpy, "xx.npy"))

    return dir_julia, l, Ki, dimension, Mmu, mu, A2, xx
    
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
    dir_julia, l, Ki, dimension, Mmu, mu, A2, xx = read_numpy_files(dir_output,
                                                            i_toroidal)

    # Change units of radius array, and get number of points.
    xx  = (xx * 1000.0)
    size_r = size(xx)[1]

    # A0 is the first term in many eigenvalue formulations but for us it
    # is equal to zero.
    temp_A0 = zeros(dimension, dimension)

    # Unpack variables.
    # eig_start is converted from mHz to rad/s.
    eig_start_rad_per_s = (anelastic_params["eig_start_mHz"] * 1.0E-3) * (2.0 * pi)
    @printf("eig_start_rad_per_s %.6f (%.3f mHz) \n", eig_start_rad_per_s,
            anelastic_params["eig_start_mHz"])
    num_eigen = anelastic_params["n_eigs"] 
    
    # Convert viscosities from SI units to Ouroboros units.
    anelastic_params = change_anelastic_param_units(anelastic_params)

    # Prepare model dictionary.
    model = prepare_model_dictionary(mu, anelastic_params)

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

    # Create the NEP with the constant (non-frequency-dependent) parts of the
    # matrices. The coefficients are for terms A0 + A1 s + A2 s^2.
    nep = PEP([temp_A0, A1, A2])

    # Loop over elements.
    poles = Vector{ComplexF64}()
    roots = Vector{ComplexF64}()
    #
    for k = 1:Ki

        # Get the mu matrix for this element.
        # This is zero except within the block:
        # 0 0 0
        # 0 M 0
        # 0 0 0
        @printf("Element %4d of %4d\n", k, Ki)
        temp_A1 = Mmu[k, :, :]

        # Get the parameters for this element.
        ele_params = Dict()
        for key in keys(model)
            
            ele_params[key] = model[key][k]

        end

        # Get the eigenvalue problem for this element.
        ele_EP, root_pole_k = get_element_EP(anelastic_params, temp_A0,
                                             temp_A1, ele_params)

        # Store root and pole information (if used).
        roots, poles = update_root_pole_list(roots, poles,
                            anelastic_params["model_type"], root_pole_k)

        # Add the eigenvalue problem for this element to the overall
        # eigenvalue problem.
        nep = SumNEP(nep, ele_EP)

    end

    # Build operator ∫𝜑*H*rho*𝜑 from original nep ω^2*∫𝜑*rho*𝜑 - ∫𝜑*H*rho*𝜑 = 0
    # This is used for calculating derivative during normalisation.
    nep_h = SumNEP(nep, PEP([-temp_A0, -temp_A0, -A2]))

    # Solve non-linear eigenvalue problem.
    println("Trying to solve eigenvalue problem...")
    method = "iar"
    #tol = 1.0E-13 
    tol = 1.0E-10
    if method == "iar"

        eigvals, eigvecs = iar( nep, maxit = 100 , σ = eig_start_rad_per_s,
                                neigs = num_eigen,
                                logger = 1,
                                tol = tol)

    else

        error(@sprintf("method %s not recognised", method))

    end
    
    # Save the eigenvector, eigenvalues, poles and roots.
    save_toroidal_eigvecs_eigvals(eigvals, xx, eigvecs, A2, nep_h, dir_julia,
                                  i_toroidal, l, num_eigen, poles, roots)

end

# Run with command-line arguments.
toroidal_rep(ARGS)
