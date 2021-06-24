using LinearAlgebra
using NonlinearEigenproblems
using NPZ
using Printf
using DelimitedFiles
#using Interpolations
#using PyCall
#interp = pyimport("scipy.interpolate")
#np = pyimport("numpy")
include("lib.jl")
include("common.jl")
order_V = 1
order = 2

function spheroidal_rep(args)
    #=
    Solve normal modes nonlinear eigenvalue problem in spheroidal case. parameter_S.txt gives parameters
    from the python code.
    blk_type gives type of each layer. 1: solid, 0:fluid
    blk_len gives constitution of each layer. Wheather it has U,V,p,P etc.
    layers is the number of layers.
    Ki is the number of Ki in each solid layer.
    size_K is the number of solid layers.
    dimensions gives result of where each element is.
    =#
    #=
    #yuen's paper
    nu1 = 1e22*1e-9
    nu2 = 1e17*1e-9
    #my test
    nu2 = 1e15*1e-9
    #body_type: 0 for elastic, -1 for maxwell, -2 for kelvin-voigt, 1 for standard linear solid, 2 for burgers body
    body_type = 2
    #wheather to calculate poles
    cal_pole = 0
    num_eigen = 4
    sigma = 0.035
    =#

    #input = readlines("input_an")
    #nu1 = parse.(Float64,(input[1]))*1e-9
    #nu2 = parse.(Float64,(input[2]))*1e-9
    #body_type = parse.(Int64,(input[3]))
    #line4 = parse.(Int64,(split(input[4]," ")))
    #cal_pole = line4[1]
    #if cal_pole == 1
    #    num_pole = line4[2]
    #    pole_option = line4[3]
    #end
    #num_eigen = parse.(Int64,(input[5]))
    #sigma = parse.(Float64,(input[6]))
    
    # Load anelastic parameters.
    path_input_anelastic = args[1]
    anelastic_params = read_input_anelastic(path_input_anelastic)
    cal_pole = 0
    #
    # Unpack variables.
    # eig_start is converted from mHz to rad/s.
    # nu is converted from SI units to Ouroboros units.
    eig_start_rad_per_s = (anelastic_params["eig_start_mHz"] * 1.0E-3) * (2.0 * pi)
    @printf("eig_start_rad_per_s %.6f (%.3f mHz) \n", eig_start_rad_per_s,
            anelastic_params["eig_start_mHz"])
    nu1 = anelastic_params["nu"] * 1.0E-9
    num_eigen = anelastic_params["n_eigs"] 

    # Read files generated by Python script, relating to matrices.
    dir_output = args[2]
    dir_numpy = joinpath(dir_output, "numpy")

    ##nu for kelvin-voigt solid
    #if body_type == -2
    #    nu2 = 1e9*1e-9
    #elseif body_type == -1
    #    #for maxwell solid
    #    nu1 = 1e15*1e-9
    #end

    lines = readlines(joinpath(dir_numpy, "parameter_S.txt"))
    size_K = 0
    Ki = Int64[]
    blk_type = Int16[]
    blk_len = Any[]
    l = parse.(Int64,(lines[1]))
    for i=1:size(lines)[1]-2
        line = parse.(Int64,(split(lines[i+1]," ")))
        append!(blk_type,line[1])
        push!(blk_len,line[3:end])
        #print(line)
        if line[1] == 1
            append!(Ki,[line[2]])
            size_K = size_K+1
        end
    end
    layers = size(blk_type)[1]
    dimensions = parse.(Int64,(split(lines[end]," ")))
    dimension = dimensions[end]

    A0 = npzread(joinpath(dir_numpy, "A0.npy"))
    temp_A0 = zeros(dimension,dimension)

    if anelastic_params["model_type"] == 0
        A1 = temp_A0
        print("Body type: elastic\n")
    elseif anelastic_params["model_type"] == -2
        A1 = temp_A0
        print("Body type: kelvin\n")
        #A1 = sum(Ki)*Matrix(1.0I, dimension, dimension)
    else
        A1 = sum(Ki)*Matrix(1.0I, dimension, dimension)
        print("Body type: others\n")
    end

    A2 = npzread(joinpath(dir_numpy, "A2.npy"))

    xx = npzread(joinpath(dir_numpy, "xx.npy"))
    xx = xx*1000
    size_r = size(xx)[1]
    x_V = npzread(joinpath(dir_numpy, "x_V.npy"))
    x   = npzread(joinpath(dir_numpy, "x.npy"))
    thickness = npzread(joinpath(dir_numpy, "thickness.npy"))
    
    #Starting nep
    nep = PEP([A0,A1,A2])

    poles = Float64[]
    poles1 = Float64[]
    poles2 = Float64[]
    count_blk_size = 1
    num = 0
    count_thick = [1]
    for i=1:layers

        if blk_type[i] == 1

            Mmu = npzread(joinpath(dir_numpy, string("Mmu",i-1,".npy")))
            mu = npzread(joinpath(dir_numpy, string("mu",i-1,".npy")))

            if anelastic_params["model_type"] == -2

                mu2 = mu

            else

                mu2 = 75*mu
                #mu2 = mu
                
            end

            num = num+1
            for k = 1:Ki[num]

                temp_A1 = zeros(dimension,dimension)
                temp_A1[count_blk_size:count_blk_size+2*blk_len[i][1]-1,
                        count_blk_size:count_blk_size+2*blk_len[i][1]-1] = Mmu[k,:,:]

                if cal_pole == 0

                    if anelastic_params["model_type"] == 0

                        #elastic case
                        #temp_rep = REP([temp_A0,temp_A1],[0],[0])
                        temp_rep = PEP([temp_A1])

                    elseif anelastic_params["model_type"] == "maxwell_uniform" 

                        #maxwell solid
                        temp_rep = REP([temp_A0, temp_A1], [0], [im*mu[k]/nu1])

                    elseif anelastic_params["model_type"] == -2

                        #kelvin solid
                        #temp_rep = REP([temp_A0,temp_A1*nu2*im/mu[k]],[0,im*mu2[k]/nu2],[0])
                        #print(mu[k],"\n")
                        temp_rep = PEP([temp_A1/mu[k]*mu2[k],temp_A1/mu[k]*nu2*im])

                    elseif anelastic_params["model_type"] == 1

                        #standard linear solid
                        temp_pole = (mu[k]+mu2[k])/nu2
                        temp_rep = REP([temp_A0,temp_A1],[im*mu2[k]/nu2],[im*temp_pole])

                    elseif anelastic_params["model_type"] == 2

                        #anelastic case, burgers body
                        b = mu[k]/nu1+mu[k]/nu2+mu2[k]/nu2
                        ac = (mu[k]/nu1)^2+(mu[k]/nu2)^2+(mu2[k]/nu2)^2+2*(mu[k]^2+mu[k]*mu2[k])/(nu1*nu2)+2*mu[k]*mu2[k]/(nu2^2)
                        x1 = (b+sqrt(ac))/2
                        x2 = (b-sqrt(ac))/2
                        temp_rep = REP([temp_A0,temp_A1],[0,im*mu2[k]/nu2],[-x1/im,-x2/im])
                    end

                elseif cal_pole == 1

                    if anelastic_params["model_type"] == 0

                        #elastic solid
                        temp_rep = PEP([temp_A1])

                    elseif anelastic_params["model_type"] == -2

                        #kelvin-voigt solid
                        temp_rep = PEP([temp_A1/mu[k]*mu2[k],temp_A1/mu[k]*nu2*im])

                    elseif anelastic_params["model_type"] == 1 || anelastic_params["model_type"] == "maxwell_uniform" 

                        if anelastic_params["model_type"] == 1

                            #standard linear solid
                            temp_pole = (mu[k]+mu2[k])/nu2
                            temp_rep = REP([temp_A0,temp_A1],[im*mu2[k]/nu2],[im*temp_pole])
                        else

                            #maxwell solid
                            temp_pole = mu[k]/nu1
                            #print(temp_pole,"\n")
                            temp_rep = REP([temp_A0,temp_A1],[0],[im*temp_pole])

                        end

                        #SLS and maxwell
                        flag = 0
                        for i = 1:size(poles)[1]

                            if abs(poles[i]-temp_pole)<poles[i]*1e-5# || temp_pole[1]<1e-10
                                flag = 1
                                break
                                
                            end

                        end

                        if flag == 0

                            append!(poles,temp_pole)

                        end

                    elseif anelastic_params["model_type"] == 2

                        #anelastic case, burgers body
                        b = mu[k]/nu1+mu[k]/nu2+mu2[k]/nu2
                        ac = (mu[k]/nu1)^2+(mu[k]/nu2)^2+(mu2[k]/nu2)^2+2*(mu[k]^2-mu[k]*mu2[k])/(nu1*nu2)+2*mu[k]*mu2[k]/(nu2^2)
                        x1 = (b+sqrt(ac))/2
                        x2 = (b-sqrt(ac))/2
                        temp_pole = [x1,x2]
                        temp_rep = REP([temp_A0,temp_A1],[0,im*mu2[k]/nu2],[-x1/im,-x2/im])
                        #print(temp_pole,"\n")
                        flag1 = 0
                        for i = 1:size(poles1)[1]

                            if abs(poles1[i]-temp_pole[1])<poles1[i]*1e-5# || temp_pole[1]<1e-10
                                flag1 = 1
                                break

                            end

                        end

                        if flag1 == 0

                            append!(poles1,temp_pole[1])

                        end

                        flag2 = 0
                        for i = 1:size(poles2)[1]

                            if abs(poles2[i]-temp_pole[2])<poles2[i]*1e-5# || temp_pole[2]<1e-10
                                flag2 = 1
                                break

                            end

                        end

                        if flag2 == 0

                            append!(poles2,temp_pole[2])

                        end

                    end

                end

                nep = SumNEP(nep,temp_rep)

            end
        end

        count_blk_size = count_blk_size + sum(blk_len[i])
        append!(count_thick,count_thick[end]+thickness[i])

    end
    #
    #get poles of burgers body ready
    if anelastic_params["model_type"] == 2 && cal_pole == 1

        append!(poles,poles1)
        append!(poles,poles2)
        print("Burgers' body\n")

    end

    print("Poles: ", poles, "\n")

    #build operator ∫𝜑*H*rho*𝜑 from original nep ω^2*∫𝜑*rho*𝜑 - ∫𝜑*H*rho*𝜑 = 0
    nep_h = SumNEP(nep,PEP([-temp_A0,-temp_A0,-A2]))

    # Solve non-linear eigenvalue problem.
    println("Trying to solve eigenvalue problem...")

    if cal_pole == 1

        #num_eigen = 4
        if anelastic_params["model_type"] == -2

            eigvals,eigvecs = iar(nep, maxit = 100, σ = im*0.1,
                                    neigs = num_eigen)

        elseif pole_option == 1

            eigvals,eigvecs = iar(nep, maxit = 100,
                                  σ = im*0.99*poles[num_pole],
                                  neigs = num_eigen)
        elseif pole_option == 2

            eigvals,eigvecs = iar(nep, maxit=100, σ = im*1.01*poles[num_pole],
                                  neigs = num_eigen)
        elseif pole_option == 3

            eigvals,eigvecs = iar(nep, maxit = 100, σ = im*sigma,
                                  neigs=num_eigen)

        end

    else

        eigvals,eigvecs = iar(nep, maxit = 100, σ = eig_start_rad_per_s,
                              neigs = num_eigen, logger = 1, tol = 1.0E-5)

    end

    # Sort the eigenvalues by their real part.
    #p = sortperm(real(eigvals),rev=true)
    p = sortperm(real(eigvals))
    eigvals = eigvals[p]
    eigvecs = eigvecs[1:end,p]
    
    # Convert from rad/s to mHz.
    fre = (eigvals / (2.0 * pi)) * 1000.0

    # Save the eigenvalues.
    name_eigvals = @sprintf("eigenvalues_%05d.txt", l)
    path_eigvals = joinpath(dir_output, name_eigvals)
    open(path_eigvals, "w") do f_eigval

        for i = 1 : num_eigen

            write(f_eigval, string(l, " ", real(fre[i]), " ", imag(fre[i]),
                                   "    ", real(eigvals[i]),
                                   " ",    imag(eigvals[i]), "\n"))

        end

    end

    # Save the poles.
    if cal_pole == 1

        name_poles = @sprintf("poles_%05d.txt", l)
        path_poles = joinpath(dir_output, name_poles)
        open(path_poles, "w") do f_eigval

            print(poles,"\n")
            for pole in poles

                write(f_pole, string((pole / (2.0 / pi)) * 1000.0))

            end

        end

    end
    
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
        if l == 1

            n = i

        else

            n = i - 1

        end
        #
        name_eigvec = @sprintf("eigvec_%05d_%05d.txt", n, l)
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

# Run with command-line arguments.
spheroidal_rep(ARGS)
