order_V = 1
order = 2

# Makes a directory if it doesn't already exist. ------------------------------
function mkdir_if_not_exist(path_)

    if isdir(path_) == false
        
        @printf("Making directory %s\n", path_)
        mkdir(path_)

    end

end

# Reads the anelastic input file specified in the modes input file.
function read_input_anelastic(path_input_anelastic)
    
    anelastic_params = Dict()

    open(path_input_anelastic) do f
        
        anelastic_params["model_type"] = split(readline(f))[2]
        anelastic_params["control_file"] = split(readline(f))[2]
        #anelastic_params["n_eigs"] = parse(Int64, split(readline(f))[2])
        #anelastic_params["eig_start_mHz"] = parse(Float64, split(readline(f))[2])

        if anelastic_params["model_type"] == "maxwell_uniform"
            
            anelastic_params["nu1"] = parse(Float64, split(readline(f))[2])

        elseif anelastic_params["model_type"] == "SLS_uniform"

            anelastic_params["nu2"] = parse(Float64, split(readline(f))[2])
            anelastic_params["mu2_factor"] = parse(Float64, split(readline(f))[2])
        
        elseif anelastic_params["model_type"] == "burgers_uniform"

            anelastic_params["nu1"] = parse(Float64, split(readline(f))[2])
            anelastic_params["nu2"] = parse(Float64, split(readline(f))[2])
            anelastic_params["mu2_factor"] = parse(Float64, split(readline(f))[2])

        elseif anelastic_params["model_type"] == "extended_burgers_uniform"

            anelastic_params["mineral"] = split(readline(f))[2]
            anelastic_params["temp_K"] = parse(Float64, split(readline(f))[2])
            anelastic_params["pressure_GPa"] = parse(Float64, split(readline(f))[2])
            anelastic_params["grain_size_m"] = parse(Float64, split(readline(f))[2])

        elseif anelastic_params["model_type"] == "SLS"

            anelastic_params["param_file"] = split(readline(f))[2]

        else
            
            error_str = @sprintf("Model type %s not recognised.\n",
                                 anelastic_params["model_type"])
            error(error_str)

        end

    end

    # Read the control file.
    open(anelastic_params["control_file"]) do f

        # Read all lines.
        lines = readlines(f)

        # Prepare output arrays.
        n_searches = length(lines)
        eig_start_mHz   = zeros(Complex,    n_searches)
        n_eigs          = zeros(Int32,      n_searches)
        n_iters         = zeros(Int32,      n_searches)
        tol             = zeros(Float64,    n_searches)

        # Parse lines.
        for i = 1 : n_searches

            line = split(lines[i])
            eig_start_mHz_real = parse(Float64, line[1])
            eig_start_mHz_imag = parse(Float64, line[2])
            eig_start_mHz[i] = eig_start_mHz_real + (1.0im * eig_start_mHz_imag)

            n_eigs[i]   = parse(Int32,      line[3])
            n_iters[i]  = parse(Int32,      line[4])
            tol[i]      = parse(Float64,    line[5])

        # Store in dictionary.
        anelastic_params["n_searches"]          = n_searches
        anelastic_params["eig_start_mHz_list"]  = eig_start_mHz
        anelastic_params["n_eigs_list"]         = n_eigs
        anelastic_params["n_iters_list"]        = n_iters
        anelastic_params["tol_list"]            = tol

        end

    end

    return anelastic_params

end

function update_root_pole_list(roots, poles, model_type, root_pole_k)

    if model_type in ["maxwell_uniform", "SLS_uniform", "burgers_uniform",
                        "SLS"]
        
        # Store roots and poles.
        for temp_root in root_pole_k["roots"]
            
            push!(roots, temp_root)

        end

        for temp_pole in root_pole_k["poles"]

            push!(poles, temp_pole)

        end

    end

    return roots, poles

end

function read_extra_anelastic_params(dir_numpy, model_type, suffix)

    if model_type == "SLS"

        extra_params = Dict()
        extra_params["mu2"]  = npzread(joinpath(dir_numpy,
                                                join(["mu2", suffix, ".npy"])))
        extra_params["eta2"] = npzread(joinpath(dir_numpy,
                                                join(["eta2", suffix, ".npy"])))

    else

        extra_params = nothing

    end

    return extra_params

end

# Save eigenvalues, roots and poles.
function save_eigvals_poles_roots(dir_output, i_toroidal, l, j_search, num_eigen, eigvals, poles, roots)

    # Convert from rad/s to mHz.
    rad_s_to_mHz = (1000.0 / (2.0 * pi))
    fre = (eigvals * rad_s_to_mHz)
    if ~isnothing(poles)

        poles = (poles * rad_s_to_mHz)

    end

    if ~isnothing(roots)

        roots = (roots * rad_s_to_mHz)

    end
    
    # Get names of output files.
    if isnothing(i_toroidal)

        name_eigvals    = @sprintf("eigenvalues_%05d_%05d.txt", l, j_search)
        name_roots      = @sprintf("roots_%05d.txt", l)
        name_poles      = @sprintf("poles_%05d.txt", l)

    else

        name_eigvals    = @sprintf("eigenvalues_%03d_%05d_%05d.txt",
                                   i_toroidal, l, j_search)
        name_roots      = @sprintf("roots_%03d_%05d.txt", i_toroidal, l)
        name_poles      = @sprintf("poles_%03d_%05d.txt", i_toroidal, l)

    end

    # Save the eigenvalues.
    path_eigvals = joinpath(dir_output, name_eigvals)
    open(path_eigvals, "w") do f_out

        for i = 1 : num_eigen

            write(f_out, @sprintf("%+19.12e %+19.12e\n", real(fre[i]), imag(fre[i])))

        end

    end

    # Save the roots.
    if ~isnothing(roots)

        num_roots = length(roots)
        path_roots = joinpath(dir_output, name_roots)
        open(path_roots, "w") do f_out

            for i = 1 : num_roots

                write(f_out,
                      @sprintf("%+19.12e %+19.12e\n",
                               real(roots[i]), imag(roots[i])))

            end

        end

    end
    
    if ~isnothing(poles)

        # Save the poles.
        num_poles = length(poles)
        path_poles = joinpath(dir_output, name_poles)
        open(path_poles, "w") do f_out

            for i = 1 : num_poles

                write(f_out,
                      @sprintf("%+19.12e %+19.12e\n",
                               real(poles[i]), imag(poles[i])))

            end

        end

    end

end

# Convert viscosities from SI units to Ouroboros units.
function old_change_anelastic_param_units(anelastic_params)

    nu_SI_to_Ouroboros = 1.0E-9
    for nu in ["nu1", "nu2"]

        if nu in keys(anelastic_params)
            
            anelastic_params[nu] = (anelastic_params[nu] * nu_SI_to_Ouroboros)

        end

    end

    return anelastic_params

end

function prepare_model_dictionary(anelastic_params, dir_numpy, layers, blk_type)

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
                mu2[i] = mu1[i] * anelastic_params["mu2_factor"] * 1.0E9
                nu2[i] = zeros(size(mu1[i])) .+ anelastic_params["nu2"]

            end

        end

        model = Dict("mu1" => mu1,
                     "mu2" => mu2,
                     "nu2" => nu2)

    elseif anelastic_params["model_type"] == "SLS"

        mu1 = Array{Any}(nothing, layers)
        mu2 = Array{Any}(nothing, layers)
        nu2 = Array{Any}(nothing, layers)

        for i = 1 : layers

            # Solid layers only.
            if blk_type[i] == 1
                
                mu1[i]  = npzread(joinpath(dir_numpy, string( "mu",   i - 1, ".npy")))
                mu2[i]  = npzread(joinpath(dir_numpy, string( "mu2",  i - 1, ".npy")))
                nu2[i] = npzread(joinpath(dir_numpy, string( "eta2", i - 1, ".npy")))

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
                mu2[i] = mu1[i] * anelastic_params["mu2_factor"] * 1.0E9
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

function change_model_units(model, layers)
    
    # Elastic moduli are input in SI units (Pa) and Ouroboros uses units of GPa.
    # Note that we do not have to change the units of mu1, because this is
    # handled in the function prep_fem().
    mu_SI_to_Ouroboros = 1.0E-9
    # Viscosities are input in SI units (Pa s) and Ouroboros uses units of GPa s.
    nu_SI_to_Ouroboros = 1.0E-9

    unit_convert_dict = Dict("nu1" => nu_SI_to_Ouroboros,
                             "nu2" => nu_SI_to_Ouroboros,
                             "mu2" => mu_SI_to_Ouroboros)

    for (var, scaling) in unit_convert_dict    

        if var in keys(model)

            for i = 1 : layers

                model[var][i] = (model[var][i] * scaling)

            end

        end

    end

    return model

end

# Gives matrix function for a single element of a specified model.
function get_element_EP(anelastic_params, temp_A0, temp_A1, ele_params)
    
    # First we treat rheologies that can be expressed as PEPs (polynomial
    # eigenvalue problems).
    if (anelastic_params["model_type"] in [0, -2])

        # Elastic case.
        if anelastic_params["model_type"] == 0

            # The elastic case is a polynomial of order 0, in other words
            # a constant; the elastic case is not frequency dependent.
            temp_EP = PEP([temp_A1])
            
        # Kelvin solid.
        elseif anelastic_params["model_type"] == -2

            # Unpack.
            mu1 = ele_params["mu1"]
            mu2 = ele_params["mu2"]
            nu2 = ele_params["nu2"]
                
            # hrmd 2021-06-20: Need reference for this form.
            temp_EP = PEP([temp_A1/mu1*mu2, temp_A1/mu1*nu2*im])

        end

        # We do not store information about roots or poles for these
        # rheologies.
        root_pole_info = nothing

    # Next, we treat rheologies which can be expressed as REPs (rational
    # eigenvalue problems).
    elseif anelastic_params["model_type"] in ["maxwell_uniform", "SLS_uniform",
                                              "burgers_uniform", "SLS"]

        # Maxwell solid with uniform viscosity.
        #elseif body_type == -1
        if anelastic_params["model_type"] == "maxwell_uniform"
            
            # Unpack.
            mu1 = ele_params["mu1"]
            nu1 = ele_params["nu1"]
            
            # Get expression for roots (zeros of eq. 2.27 in ref. [1]).
            temp_roots = [0.0]

            # Get expression for poles (zeros of eq. 2.28 in ref. [1]).
            temp_poles = [-mu1 / nu1]

        # Standard linear solid.
        #elseif anelastic_params["model_type"] == 1
        elseif anelastic_params["model_type"] in ["SLS_uniform", "SLS"]
            
            # Unpack.
            mu1 = ele_params["mu1"]
            mu2 = ele_params["mu2"]
            nu2 = ele_params["nu2"]

            # Get expression for roots (zeros of numerator of
            # eq. 13c in ref. [2]).
            # Note that there is an error in the numerator of eq. 13c.
            # It should be
            # \mu (s) = \frac{\mu_{1}(s + \frac{\mu_{1}}{\nu_{2}})}{s +
            # \frac{1}{\nu_{2}}(\mu_{1} + \mu_{2})} 
            temp_roots = [-mu1 / nu2]

            # Get expression for poles (zeros of denominator of eq. 13c
            # in ref. [2]).
            temp_poles = [-(mu1 + mu2) / nu2]

        # Burger's solid with uniform viscosities.
        #elseif anelastic_params["model_type"] == 2
        elseif anelastic_params["model_type"] == "burgers_uniform" 

            # Unpack.
            mu1 = ele_params["mu1"]
            mu2 = ele_params["mu2"]
            nu1 = ele_params["nu1"]
            nu2 = ele_params["nu2"]
            
            # Get expression for roots (zeros of eq. 2.32 in ref. [1]). 
            temp_roots = [0.0, -(mu2 / nu2)]

            # Get expression for poles (zeros of eq. 2.33 in ref. [1]).
            # Use quadratic formula.
            a = (nu1 * nu2) / (mu1 * mu2) 
            b = (nu1 / mu1) + (nu1 / mu2) + (nu2 / mu2)
            c = 1.0
            #
            det = (b^2.0) - (4.0 * a * c)
            sqrt_det = sqrt(det)
            #
            p1 = (-b - sqrt_det) / (2.0 * a)
            p2 = (-b + sqrt_det) / (2.0 * a)
            #
            temp_poles = [p1, p2]

            @printf("%.1e %.1e %.1e %.1e", mu1, mu2, nu1, nu2)

        end

        # Create the rational eigenvalue problem for this element.
        # The Laplace variable s and the angular frequency omega are
        # related by
        #    s = i omega  <-->  omega = -i s
        # We are seeking solutions omega, so the roots and poles must
        # be converted from s to omega:
        temp_roots = -im * temp_roots 
        temp_poles = -im * temp_poles
        root_pole_info  = Dict("roots" => temp_roots, "poles" => temp_poles)
        # 
        temp_EP = REP([temp_A0, temp_A1], temp_roots, temp_poles)

    elseif (anelastic_params["model_type"] in ["extended_burgers_uniform"])

        mineral_params = py_ebm.define_mineral_params()

        function ebm_response(om_rad_per_s)

            om_rad_per_s = real(om_rad_per_s)

            conditions = Dict()

            conditions["mineral_id"] = zeros(Int64, 1) .+ ele_params["mineral_id"]
            conditions["T"]          = zeros(1) .+ ele_params["temp_K"]
            conditions["P"]          = zeros(1) .+ ele_params["pressure_GPa"] * 1.0E9
            conditions["d"]          = zeros(1) .+ ele_params["grain_size_m"]
            conditions["omega"]      = zeros(1) .+ om_rad_per_s
            conditions["n_samples"]  = 1

            relaxation_periods = py_ebm.calculate_relaxation_periods(
                                            mineral_params, conditions)

            J1, J2 = py_ebm.calculate_moduli_factors_loop(mineral_params,
                        relaxation_periods, conditions)

            J = J1[1] + (im * J2[1])

        end
        
        #J_test = ebm_response(1000.0)
        #println("\n", J_test)
        
        temp_A1_cplx = convert(Matrix{ComplexF64}, temp_A1)
        temp_EP_spmf = SPMF_NEP([temp_A1_cplx], [ebm_response],
                           check_consistency = false)
        #println(temp_EP_spmf)
        # See https://nep-pack.github.io/NonlinearEigenproblems.jl/
        # Section "Chebyshev interpolation"
        #temp_EP = ChebPEP(temp_EP_spmf, 10, a = 1.0E-3, b = 1.0E7,
        #                  cosine_formula_cutoff = 10)
        a = -1.0
        b =  1.0E7
        cosine_formula_cutoff = 8 
        println('A')
        temp_EP = ChebPEP(temp_EP_spmf, 20, a, b, cosine_formula_cutoff = 
                            cosine_formula_cutoff)

        root_pole_info = nothing

        #A0=[1 3; 4 5]; A1=[3 4; 5 6];
        #id_op=S -> one(S) 
        #exp_op=S -> exp(S)
        #nep=SPMF_NEP([A0,A1],[id_op,exp_op]);

    else

        error("Model type string is incorrect.")

    end

    return temp_EP, root_pole_info

end

# Solve NEP.
function solve_NEP_wrapper(nep, anelastic_params, poles, roots, l, i_toroidal,
    dir_output, dir_julia, save_params)

    for j = 1 : anelastic_params["n_searches"]

        # Unpack dictionary.
        eig_start_mHz   = anelastic_params["eig_start_mHz_list"][j]
        n_eigs          = anelastic_params["n_eigs_list"][j]
        maxit           = anelastic_params["n_iters_list"][j]
        tol             = anelastic_params["tol_list"][j]
        
        # eig_start is converted from mHz to rad/s.
        eig_start_rad_per_s = (eig_start_mHz * 1.0E-3) * (2.0 * pi)

        # Report.
        @printf("Trying to solve eigenvalue problem, search %3d of %3d \n",
                j, anelastic_params["n_searches"])
        @printf("eig_start_rad_per_s %.6f + %.6f i (%.3f + %.3f i mHz) \n",
                real(eig_start_rad_per_s),
                imag(eig_start_rad_per_s),
                real(eig_start_mHz),
                imag(eig_start_mHz))

        # Try to solve eigenvalue problem.
        eigvals, eigvecs = iar( nep,
                                maxit    = maxit,
                                σ        = eig_start_rad_per_s,
                                neigs    = n_eigs,
                                tol      = tol,
                                logger   = 1)

        # Save the eigenvector, eigenvalues, poles and roots.
        if j == 1

            poles_out = poles
            roots_out = roots

        else

            poles_out = nothing
            roots_out = nothing

        end

        save_eigvals_poles_roots(dir_julia, i_toroidal, l, j, n_eigs, eigvals,
                                 poles_out, roots_out)

        if isnothing(i_toroidal)
            
            save_spheroidal_eigvecs(eigvals, eigvecs, dir_julia, l, j, n_eigs,
                                    save_params)
            #save_spheroidal_eigvecs(eigvals, eigvecs, A2, count_blk_size, blk_type,
            #                        blk_len, layers, x, xx, x_V, thickness, nep_h,
            #                        dir_julia, l, n_eigs, j)

        else
            
            save_toroidal_eigvecs(eigvals, eigvecs, dir_output, dir_julia, l,
                                  j, i_toroidal, n_eigs, save_params)
            #save_toroidal_eigvecs(eigvals, xx, eigvecs, A2, nep_h, dir_julia,
            #                      i_toroidal, l, num_eigen)

        end

    end

end
