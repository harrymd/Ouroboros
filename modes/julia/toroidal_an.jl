# References:
# [1] GMIG Report (2020) Vol. 5 Report 12, pp 263--289.
# [2] Yuen and Peltier (1982) 'Normal modes of the viscoelastic Earth'
#       Geophys. J. R. astr. Soc. 69, pp 495--526.
using NonlinearEigenproblems
using Printf
include("lib.jl")
include("common.jl")

# Import the Python library for the Extended Burgers Model (EBM).
#ENV["PYTHON"] = "/anaconda/envs/stoneley_data/bin/python"
#using Pkg
#Pkg.build("PyCall")    # Can be necessary to rebuild PyCall if Python environment 
                        # has changed.
#using PyCall
#py_ebm = pyimport("anelasticity.ebm")
#py_ebm = pyimport("anelasticity.calculate_EBM_response_empirical")

#conditions = Dict()
#conditions["mineral_id"] = zeros(Int64, 1) .+ 0 # Integer ID of mineral.
#conditions["T"]     = zeros(1) .+ 900.0 .+ 273.0  # Temperature (K).
#conditions["P"]     = zeros(1) .+ 0.2E9          # Pressure (Pa).
#conditions["d"]     = zeros(1) .+ 13.4E-6        # Grain size (m).
#conditions["omega"] = zeros(1) .+ (5.0E-3 * 2.0 * pi) # Angular
#                                                    # frequency (rad/s).
#conditions["n_samples"] = length(conditions["mineral_id"]) 

function prepare_model_dictionary_toroidal(mu, anelastic_params, dir_numpy)

    if anelastic_params["model_type"] == "SLS_uniform"

        model = Dict("mu1" => mu,
                     "mu2" => mu * anelastic_params["mu2_factor"],
                     "nu2" => zeros(size(mu)) .+ anelastic_params["nu2"])

    elseif anelastic_params["model_type"] == "burgers_uniform"

        model = Dict("mu1" => mu,
                     "mu2" => mu * anelastic_params["mu2_factor"],
                     "nu1" => zeros(size(mu)) .+ anelastic_params["nu1"],
                     "nu2" => zeros(size(mu)) .+ anelastic_params["nu2"])

    elseif anelastic_params["model_type"] == "extended_burgers_uniform"
        
        mineral_id = py_ebm.mineral_name_to_int[anelastic_params["mineral"]]
        model = Dict(   "mu1"           => mu,
                        "mineral_id"    => zeros(Int8, size(mu)) .+ mineral_id,
                        "temp_K"        => zeros(size(mu)) .+ anelastic_params["temp_K"],
                        "pressure_GPa"  => zeros(size(mu)) .+ anelastic_params["pressure_GPa"],
                        "grain_size_m"  => zeros(size(mu)) .+ anelastic_params["grain_size_m"])

    elseif anelastic_params["model_type"] == "SLS"

        mu2  = npzread(joinpath(dir_numpy, "mu2.npy"))
        nu2 = npzread(joinpath(dir_numpy,  "eta2.npy"))

        model = Dict("mu1" => mu,
                     "mu2" => mu2,
                     "nu2" => nu2)

    else

        error("Not implemented.")

    end

    return model

end

function save_toroidal_eigvecs(eigvals, eigvecs, dir_output, dir_julia, l,
        j_search, i_toroidal, n_eigs, save_params)
    #function save_toroidal_eigvecs(eigvals, xx, eigvecs, A2, nep_h, dir_output, i_toroidal, l, num_eigen)
    
    # Unpack save parameter dictionary.
    xx      = save_params["xx"]
    B      = save_params["B"]
    nep_h   = save_params["nep_h"]

    # Get number of samples.
    size_r = size(xx)[1]

    # Convert from rad/s to mHz.
    rad_s_to_mHz = (1000.0 / (2.0 * pi))
    fre = (eigvals * rad_s_to_mHz)
    
    # Save the eigenvectors.
    name_eigvecs = @sprintf("eigenfunctions_%03d", i_toroidal)
    dir_eigvecs = joinpath(dir_julia, name_eigvecs)
    println('\n', dir_eigvecs)
    mkdir_if_not_exist(dir_eigvecs)
    k = sqrt(l * (l + 1.0))
    for i = 1 : n_eigs

        W_eigen = eigvecs[1 : size_r, i]

        # Normalise eigenvector.
        # GMIG report 2020, eq. 3.4.
        scale = sqrt(transpose(W_eigen) * B * W_eigen -
                     (1.0 / (2.0 * eigvals[i])) * transpose(W_eigen) *
                        compute_Mder(nep_h, eigvals[i], 1) * W_eigen)
        W_eigen = (W_eigen / scale)
        # 
        scaleW = (k * fre[i] * 2.0 * pi)
        W_eigen = (W_eigen / scaleW)

        # Get output path.
        n = i - 1
        name_eigvec = @sprintf("eigvec_%05d_%05d_%05d.txt", n, l, j_search)
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

function read_numpy_files_toroidal(dir_output, i_toroidal)
#function read_numpy_files_toroidal(dir_output, i_toroidal, model_type)

    dir_numpy = joinpath(dir_output, @sprintf("numpy_%03d", i_toroidal))
    dir_julia = joinpath(dir_output, @sprintf("julia_%03d", i_toroidal))
    #
    lines = readlines(joinpath(dir_numpy, "parameter_T.txt"))
    l = parse.(Int64,(lines[1]))
    Ki = parse.(Int64,(lines[2]))
    dimension = parse.(Int64,(lines[3]))
    #
    #Mmu = npzread(joinpath(dir_numpy, "Mmu.npy"))
    A = npzread(joinpath(dir_numpy, "A.npy"))
    B = npzread(joinpath(dir_numpy, "B.npy"))
    mu  = npzread(joinpath(dir_numpy, "mu.npy"))
    #A2  = npzread(joinpath(dir_numpy, "A2.npy"))
    xx  = npzread(joinpath(dir_numpy, "xx.npy"))
    
    #extra_params = read_extra_anelastic_params(dir_numpy, model_type, "")

    #return dir_julia, l, Ki, dimension, A, mu, B, xx, extra_params
    return dir_numpy, dir_julia, l, Ki, dimension, A, mu, B, xx
    
end

# Finds the toroidal modes in anelastic case by solving REP (rational
# eigenvalue problem).
function toroidal_rep(args)

    # Load anelastic parameters.
    path_input_anelastic = args[1]
    anelastic_params = read_input_anelastic(path_input_anelastic)
   
    # Read files generated by Python script, relating to matrices.
    # ? Put polynomials here.
    dir_output = args[2]
    i_toroidal = parse(Int64, args[3])
    #dir_julia, l, Ki, dimension, A, mu, B, xx, extra_params = 
    #    read_numpy_files_toroidal(dir_output, i_toroidal, anelastic_params["model_type"])
    dir_numpy, dir_julia, l, Ki, dimension, A, mu, B, xx = 
        read_numpy_files_toroidal(dir_output, i_toroidal)

    # Change units of radius array, and get number of points.
    xx  = (xx * 1000.0)
    size_r = size(xx)[1]

    # A0 is the first term in many eigenvalue formulations but for us it
    # is equal to zero.
    temp_A0 = zeros(dimension, dimension)

    ## Convert viscosities from SI units to Ouroboros units.
    #anelastic_params = change_anelastic_param_units(anelastic_params)

    # Prepare model dictionary.
    #model = prepare_model_dictionary_toroidal(mu, anelastic_params, extra_params)
    model = prepare_model_dictionary_toroidal(mu, anelastic_params, dir_numpy)

    # Define linear term in polynomial eigenvalue problem.
    # This is generally zero, however when the individual elements consist of
    # REPs, these contain (by default) a unit linear term, therefore
    # in those cases we must add Ki unit linear terms to cancel these
    # terms out.
    #
    # Elastic case (0) and Kelvin case (-2) have no REP terms.
    # Uniform extended Burgers has no REP terms.
    if anelastic_params["model_type"] in [0, -2, "extended_burgers_uniform"]

        # This is a zero matrix.
        A1 = temp_A0

    # All other cases have REP terms.
    elseif anelastic_params["model_type"] in ["maxwell_uniform", "SLS_uniform",
                                              "burgers_uniform", "SLS"]

        A1 = (Ki * Matrix(1.0I, dimension, dimension))

    else
        
        error_str = @sprintf("Model type %s not recognised.",
                             anelastic_params["model_type"])
        error(error_str)

    end

    # Create the NEP with the constant (non-frequency-dependent) parts of the
    # matrices. The coefficients are for terms A0 + A1 s + A2 s^2.
    nep = PEP([temp_A0, A1, B])

    # Loop over elements.
    poles = Vector{ComplexF64}()
    roots = Vector{ComplexF64}()
    #
    for k = 1 : Ki

        # Get the mu matrix for this element.
        # This is zero except within the block:
        # 0 0 0
        # 0 M 0
        # 0 0 0
        @printf("Element %4d of %4d\n", k, Ki)
        temp_A1 = A[k, :, :]

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
    # The negative signs are used to cancel out unwanted terms.
    nep_h = SumNEP(nep, PEP([-temp_A0, -temp_A0, -B]))

    # Make dictionary of values needed to save output.
    save_params = Dict()
    save_params["xx"]       = xx
    save_params["B"]       = B 
    save_params["nep_h"]    = nep_h

    # Solve non-linear eigenvalue problem.
    solve_NEP_wrapper(nep, anelastic_params, poles, roots, l, i_toroidal,
                      dir_output, dir_julia, save_params)

end

# Run with command-line arguments.
toroidal_rep(ARGS)
