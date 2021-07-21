using LinearAlgebra
using NonlinearEigenproblems
using NPZ
using Printf
using DelimitedFiles
include("lib.jl")
include("common.jl")

function save_spheroidal_eigvecs(eigvals, eigvecs, dir_julia, l, j_search,
                                    n_eigs, save_params)
    #function save_spheroidal_eigvecs(eigvals, eigvecs, A2, count_blk_size, 
    #        blk_type, blk_len, layers, x, xx, x_V, thickness,nep_h, dir_output, l,
    #        num_eigen, j_search)
    #
    
    # Unpack save params.
    A2      = save_params["A2"]
    x       = save_params["x"]
    x_V     = save_params["x_V"]
    xx      = save_params["xx"]
    nep_h   = save_params["nep_h"]

    size_r = size(xx)[1]
    
    # Convert from rad/s to mHz.
    fre = (eigvals / (2.0 * pi)) * 1000.0

    # Save the eigenvectors.
    name_eigvecs = "eigenfunctions"
    dir_eigvecs = joinpath(dir_output, name_eigvecs)
    mkdir_if_not_exist(dir_eigvecs)
    #
    for i = 1 : num_eigen

        # Prepare output arrays.
        U_eigen = ComplexF64[]
        V_eigen = ComplexF64[]

        # Interpolate the eigenvector.
        count_blk_size = 1
        for j=1:layers

            append!(U_eigen,eigvecs[count_blk_size:count_blk_size+blk_len[j][1]-1,i])
            if blk_type[j] == 0

                xx_V = sqzx(x_V[1:end,count_thick[j]: count_thick[j+1]-1],thickness[j],order_V)
                sitp = interp.interp1d(xx_V,eigvecs[count_blk_size+blk_len[j][1]: count_blk_size+blk_len[j][1]+blk_len[j][2]-1,i],"cubic")
                V_inter = sitp(sqzx(x[:,count_thick[j]: count_thick[j+1]-1],thickness[j],order))
                append!(V_eigen,V_inter)

            else

                append!(V_eigen,eigvecs[count_blk_size+blk_len[j][1]: count_blk_size+blk_len[j][1]+blk_len[j][2]-1,i])

            end

            count_blk_size = count_blk_size + sum(blk_len[j])

        end

        # Normalise eigenvector.
        # GMIG report 2020, eq. 3.4.
        scale = sqrt(transpose(eigvecs[1:end,i])*A2*eigvecs[1:end,i]-0.5/eigvals[i]*transpose(eigvecs[1:end,i])
                    *compute_Mder(nep_h,eigvals[i],1)*eigvecs[1:end,i])
        U_eigen = U_eigen/scale
        V_eigen = V_eigen/scale
        scaleU = fre[i]*2*pi
        scaleV = sqrt(l*(l+1))*scaleU
        U_eigen = U_eigen/scaleU
        V_eigen = V_eigen/scaleV
        
        # Get output path.
        #if l == 1

        #    n = i

        #else

        #    n = i - 1

        #end
        #
        name_eigvec = @sprintf("eigvec_%05d_%05d_%05d.txt", i - 1, l, j_search)
        path_eigvec = joinpath(dir_eigvecs, name_eigvec)
        #
        open(path_eigvec, "w") do f_eigvec

            # Loop over node points.
            for j = 1:size_r

                write(f_eigvec,string(  @sprintf("%3.15E",xx[j]), " ",
                                        @sprintf("%3.15E",real(U_eigen[j])), " ",
                                        @sprintf("%3.15E",imag(U_eigen[j])), " ",
                                        @sprintf("%3.15E",real(V_eigen[j])), " ",
                                        @sprintf("%3.15E",imag(V_eigen[j])),"\n"))

            end

        end

    end

end

function read_numpy_files_spheroidal(dir_output)

    dir_julia = joinpath(dir_output, "julia")
    dir_numpy = joinpath(dir_output, "numpy")
    #    
    lines = readlines(joinpath(dir_numpy, "parameter_S.txt"))
    size_K = 0
    Ki = Int64[]
    blk_type = Int16[]
    blk_len = Any[]
    l = parse.(Int64, (lines[1]))
    for i = 1 : (size(lines)[1] - 2)

        line = parse.(Int64, (split(lines[i + 1], " ")))
        append!(blk_type, line[1])
        push!(blk_len, line[3 : end])

        if line[1] == 1

            append!(Ki, [line[2]])
            size_K = (size_K + 1)

        end

    end
    #
    layers = size(blk_type)[1]
    dimensions = parse.(Int64, (split(lines[end], " ")))
    dimension = dimensions[end]
    #
    A0 = npzread(joinpath(dir_numpy, "A0.npy"))
    A2 = npzread(joinpath(dir_numpy, "A2.npy"))
    xx = npzread(joinpath(dir_numpy, "xx.npy"))
    x_V = npzread(joinpath(dir_numpy, "x_V.npy"))
    x   = npzread(joinpath(dir_numpy, "x.npy"))
    thickness = npzread(joinpath(dir_numpy, "thickness.npy"))

    return dir_numpy, dir_julia, A0, A2, xx, x, x_V, l, dimension, layers,
        blk_type, blk_len, Ki, size_K, thickness

end

function prepare_model_dictionary_spheroidal(anelastic_params, dir_numpy, layers, blk_type)

    if anelastic_params["model_type"] == "maxwell_uniform"

        mu1 = Array{Any}(nothing, layers)
        nu1 = Array{Any}(nothing, layers)

        for i = 1 : layers

            # Solid layers only.
            if blk_type[i] == 1
                
                mu1[i] = npzread(joinpath(dir_numpy, string( "mu", i - 1, ".npy")))
                nu1[i] = zeros(size(mu1[i])) .+ anelastic_params["nu1"]

            end

        end

        model = Dict("mu1" => mu1,
                     "nu1" => nu1)

    elseif anelastic_params["model_type"] == "SLS_uniform"

        mu1 = Array{Any}(nothing, layers)
        mu2 = Array{Any}(nothing, layers)
        nu2 = Array{Any}(nothing, layers)

        for i = 1 : layers

            # Solid layers only.
            if blk_type[i] == 1
                
                mu1[i] = npzread(joinpath(dir_numpy, string( "mu", i - 1, ".npy")))
                mu2[i] = mu1[i] * anelastic_params["mu2_factor"]
                nu2[i] = zeros(size(mu1[i])) .+ anelastic_params["nu2"]

            end

        end

        model = Dict("mu1" => mu1,
                     "mu2" => mu2,
                     "nu2" => nu2)

    elseif anelastic_params["model_type"] == "burgers_uniform"

        mu1 = Array{Any}(nothing, layers)
        mu2 = Array{Any}(nothing, layers)
        nu1 = Array{Any}(nothing, layers)
        nu2 = Array{Any}(nothing, layers)

        for i = 1 : layers

            # Solid layers only.
            if blk_type[i] == 1
                
                mu1[i] = npzread(joinpath(dir_numpy, string( "mu", i - 1, ".npy")))
                mu2[i] = mu1[i] * anelastic_params["mu2_factor"]
                nu1[i] = zeros(size(mu1[i])) .+ anelastic_params["nu1"]
                nu2[i] = zeros(size(mu1[i])) .+ anelastic_params["nu2"]

            end

        end

        model = Dict("mu1" => mu1,
                     "mu2" => mu2,
                     "nu1" => nu1,
                     "nu2" => nu2)

    else

        error("Not implemented.")

    end



    return model

end

function spheroidal_rep(args)
    #=
    Solve normal modes nonlinear eigenvalue problem in spheroidal case.
    parameter_S.txt gives parameters from the python code.
    blk_type gives type of each layer. 1: solid, 0:fluid
    blk_len gives constitution of each layer. Wheather it has U,V,p,P etc.
    layers is the number of layers.
    Ki is the number of Ki in each solid layer.
    size_K is the number of solid layers.
    dimensions gives result of where each element is.
    =#
    
    # Load anelastic parameters.
    path_input_anelastic = args[1]
    anelastic_params = read_input_anelastic(path_input_anelastic)
   
    # Read files generated by Python script, relating to matrices.
    # ? Put polynomials here.
    dir_output = args[2]
    dir_numpy, dir_julia, A0, A2, xx, x, x_V, l, dimension, layers, blk_type,
        blk_len, Ki, size_K, thickness = read_numpy_files_spheroidal(dir_output)

    # Change units of radius array, and get number of points.
    xx = xx*1000
    size_r = size(xx)[1]

    # A0 is the first term in many eigenvalue formulations but for us it
    # is equal to zero.
    temp_A0 = zeros(dimension, dimension)
    
    # Convert viscosities from SI units to Ouroboros units.
    anelastic_params = change_anelastic_param_units(anelastic_params)

    # Prepare model dictionary.
    model = prepare_model_dictionary_spheroidal(anelastic_params, dir_numpy,
                                                layers, blk_type)

    # Define linear term in polynomial eigenvalue problem.
    # This is generally zero, however when the individual elements consist of
    # REPs, these contain (by default) a unit linear term, therefore
    # in those cases we must add Ki unit linear terms to cancel these
    # terms out.
    #
    # Elastic case (0) and Kelvin case (-2) have no REP terms.
    if anelastic_params["model_type"] in [0, -2]

        # This is a zero matrix.
        A1 = temp_A0

    # All other cases have REP terms.
    elseif anelastic_params["model_type"] in ["maxwell_uniform", "SLS_uniform",
                                              "burgers_uniform"]

        A1 = (sum(Ki) * Matrix(1.0I, dimension, dimension))

    else
        
        error_str = @sprintf("Model type %s not recognised.",
                             anelastic_params["model_type"])
        error(error_str)

    end

    # Create the NEP with the constant (non-frequency-dependent) parts of the
    # matrices. The coefficients are for terms A0 + A1 s + A2 s^2.
    nep = PEP([A0, A1, A2])

    # Prepare loop variables.
    poles = Vector{ComplexF64}()
    roots = Vector{ComplexF64}()
    count_blk_size = 1
    num = 0
    count_thick = [1]

    # Loop over the layers in the model.
    for i = 1 : layers

        # Solid layers only.
        if blk_type[i] == 1

            Mmu = npzread(joinpath(dir_numpy, string("Mmu", i - 1, ".npy")))
            mu  = npzread(joinpath(dir_numpy, string( "mu", i - 1, ".npy")))

            # Loop over the elements of this layer.
            num = (num + 1)
            for k = 1 : Ki[num]

                # Get the mu matrix for this element.
                # This is zero except within the block:
                # 0 0 0
                # 0 M 0
                # 0 0 0
                temp_A1 = zeros(dimension, dimension)
                j1 = count_blk_size
                j2 = (count_blk_size + (2 * blk_len[i][1]) - 1)
                temp_A1[j1 : j2, j1 : j2] = Mmu[k, :, :]
                
                # Get the parameters for this element.
                ele_params = Dict()
                for key in keys(model)
                    
                    ele_params[key] = model[key][i][k]

                end

                # Get the eigenvalue problem for this element.
                ele_EP, root_pole_k = get_element_EP(anelastic_params, temp_A0,
                                                     temp_A1, ele_params)
                #
                # Store root and pole information (if used).
                roots, poles = update_root_pole_list(roots, poles,
                                    anelastic_params["model_type"], root_pole_k)

                # Add the eigenvalue problem for this element to the overall
                # eigenvalue problem.
                nep = SumNEP(nep, ele_EP)

            end

        end
        
        # Update counters.
        count_blk_size = count_blk_size + sum(blk_len[i])
        append!(count_thick, count_thick[end] + thickness[i])

    end

    # Build operator ∫𝜑*H*rho*𝜑 from original nep ω^2*∫𝜑*rho*𝜑 - ∫𝜑*H*rho*𝜑 = 0
    # This is used to calculate derivative for normalisation purposes.
    # The negative signs are used to cancel out unwanted terms.
    nep_h = SumNEP(nep, PEP([-temp_A0, -temp_A0, -A2]))

    # Make dictionary of values needed to save output.
    save_params = Dict()
    save_params["A2"]       = A2
    save_params["x"]        = x
    save_params["xx"]       = xx
    save_params["x_V"]      = x_V
    save_params["nep_h"]    = nep_h

    # Solve non-linear eigenvalue problem.
    solve_NEP_wrapper(nep, anelastic_params, poles, roots, l, nothing,
                      dir_julia, save_params)

end

# Run with command-line arguments.
spheroidal_rep(ARGS)
