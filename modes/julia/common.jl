# Makes a directory if it doesn't already exist. ------------------------------
function mkdir_if_not_exist(path_)

    if isdir(path_) == false
        
        @printf("Making directory %s\n", path_)
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
            
            anelastic_params["nu1"] = parse(Float64, split(readline(f), ' ')[2])
        
        elseif anelastic_params["model_type"] == "burgers_uniform"

            anelastic_params["nu1"] = parse(Float64, split(readline(f), ' ')[2])
            anelastic_params["nu2"] = parse(Float64, split(readline(f), ' ')[2])
            anelastic_params["mu2_factor"] = parse(Float64, split(readline(f), ' ')[2])

        end

    end

    return anelastic_params

end

function update_root_pole_list(roots, poles, model_type, root_pole_k)

    if model_type in ["maxwell_uniform", 1, "burgers_uniform"]
        
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

# Convert viscosities from SI units to Ouroboros units.
function change_anelastic_param_units(anelastic_params)

    nu_SI_to_Ouroboros = 1.0E-9
    for nu in ["nu1", "nu2"]

        if nu in keys(anelastic_params)
            
            anelastic_params[nu] = (anelastic_params[nu] * nu_SI_to_Ouroboros)

        end

    end

    return anelastic_params

end

function prepare_model_dictionary(mu, anelastic_params)

    if anelastic_params["model_type"] == "burgers_uniform"

        model = Dict("mu1" => mu,
                     "mu2" => mu * anelastic_params["mu2_factor"],
                     "nu1" => zeros(size(mu)) .+ anelastic_params["nu1"],
                     "nu2" => zeros(size(mu)) .+ anelastic_params["nu2"])

    else

        error("Not implemented.")

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

    # The remaining rheologies can be expressed as REPs (rational
    # eigenvalue problems).
    elseif anelastic_params["model_type"] in ["maxwell_uniform", 1,
                                              "burgers_uniform"]

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
        elseif anelastic_params["model_type"] == 1
            
            # Unpack.
            mu1 = ele_params["mu1"]
            mu2 = ele_params["mu2"]
            nu1 = ele_params["nu1"]
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

    else

        error("Model type string is incorrect.")

    end

    return temp_EP, root_pole_info

end
