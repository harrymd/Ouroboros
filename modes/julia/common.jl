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
