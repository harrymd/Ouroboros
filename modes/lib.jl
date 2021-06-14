# - Julia
# - Author: jiayuan han
# - Date: 2019-12-29
# us shell script in python to run this file and get the result.
using LinearAlgebra
using NonlinearEigenproblems
using NPZ
using Printf
using DelimitedFiles
#using Interpolations
using PyCall
#interp = pyimport("scipy.interpolate")
#np = pyimport("numpy")
order_V = 1
order = 2

#try to turn the rational eigenvalue problem to polynomial eigenvalue problem
#This test function could avoid poles in relaxation mdoes calculation
function yuen_pep()
    nu1 = 1e22*1e-9
    nu2 = 1e17*1e-9

    lines = readlines("parameter_S.txt")
    Ki0 = parse.(Int64,(split(lines[1]," ")))[2]
    Ki = Ki0
    dimensions = parse.(Int64,(split(lines[end]," ")))

    dimension = dimensions[end]
    dimension0 = dimensions[2]
    print(dimension," ",Ki," ",Ki0,"\n")
    Mmu0 = npzread("Mmu0.npy")
    mu0 = npzread("mu0.npy")
    mu2_0 =75*mu0

    A0 = npzread("A0.npy")
    temp_A0 = zeros(dimension,dimension)
    A1 = Ki*Matrix(1.0I, dimension, dimension)
    A2 = npzread("A2.npy")
    xx = npzread("xx.npy")
    xx = xx*1000
    size_r = size(xx)[1]
    #Starting nep
    nep = PEP([A0,A1,A2])

    poles = Float64[]
    poles1 = Float64[]
    poles2 = Float64[]

    for k = 1:Ki0
        temp_A1 = zeros(dimension,dimension)
        temp_A1[1:2*dimension0,1:2*dimension0] = Mmu0[k,:,:]
        #=
        #anelastic case, burger's body
        b = mu0[k]/nu1+mu0[k]/nu2+mu2_0[k]/nu2
        ac = (mu0[k]/nu1)^2+(mu0[k]/nu2)^2+(mu2_0[k]/nu2)^2+2*(mu0[k]^2+mu0[k]*mu2_0[k])/(nu1*nu2)+2*mu0[k]*mu2_0[k]/(nu2^2)
        x1 = (b+sqrt(ac))/2
        x2 = (b-sqrt(ac))/2
        temp_pole = [x1,x2]
        temp_rep = REP([temp_A0,temp_A1],[0,im*mu2_0[k]/nu2],[-x1/im,-x2/im])
        =#

        #standard linear solid
        temp_pole = (mu0[k]+mu2_0[k])/nu2
        temp_rep = REP([temp_A0,temp_A1],[im*mu2_0[k]/nu2],[im*temp_pole])
        #temp_rep = REP([temp_A0,temp_A1],[im*mu2_0[k]/nu2,0],[0])
        #SLS
        flag = 0
        for i = 1:size(poles)[1]
            if abs(poles[i]-temp_pole)<poles[i]*1e-5
                flag = 1
                break
            end
        end
        if flag == 0
            append!(poles,temp_pole)
        end

        #elastic case
        #temp_rep = REP([temp_A0,temp_A1],[0],[0])
        nep = SumNEP(nep,temp_rep)
    end
    #nep = shift_and_scale(nep,shift=-im*poles[1],scale=1)
    #append!(poles,poles1)
    #append!(poles,poles2)
    #print(poles,"\n")
    eigvals,eigvecs = iar(nep,maxit=100,σ=poles[1]*0.99*im,neigs=1)
    #eigvals,eigvecs = iar(nep,maxit=100,σ=poles[1]*0.99*im,tol=eps()*10,neigs=3)
    #eigvals,eigvecs = iar(nep,maxit=100,σ=0.01,neigs=2)
    #eigvals,eigvecs = tiar(nep,maxit=100,σ=0.0001*im,neigs=1)
    #eigvals,eigvecs = tiar(nep,maxit=100,σ=0.000,neigs=10)
    #sort result
    #print(imag(eigvals))
    p = sortperm(imag(eigvals),rev=true)
    eigvals = eigvals[p]
    eigvecs = eigvecs[1:end,p]

    #=
    print(norm(compute_Mlincomb(nep,eigvals[1],eigvecs[1:end,1])),"\n")
    print(norm(compute_Mlincomb(nep,eigvals[2],eigvecs[1:end,2])),"\n")
    print(norm(compute_Mlincomb(nep,eigvals[3],eigvecs[1:end,3])),"\n")
    print(norm(compute_Mlincomb(nep,eigvals[4],eigvecs[1:end,4])),"\n")
    print(norm(compute_Mlincomb(nep,eigvals[5],eigvecs[1:end,5])),"\n")
    =#

    fre = eigvals/2/pi*1000
    if isdir("output") == false
        mkdir("output")
        mkdir("output/eigenfunction")
    elseif isdir("output/eigenfunction") == false
        mkdir("output/eigenfunction")
    end

    f_pole = open("output/poles.txt","w")
    for pole in poles
        println(f_pole,string(pole/2/π*1000," ",pole))
    end
    close(f_pole)

    f_eigval = open("output/eigvals_S.txt","w")
    for i = 1:size(eigvals)[1]
        write(f_eigval,string(real(fre[i])," ",imag(fre[i]),"    ",real(eigvals[i])," ",imag(eigvals[i]),"\n"))
        f_eigvec = open(string("output/eigenfunction/eigvecs_",i,".txt"),"w")
        for j = 1:size_r
            write(f_eigvec,string(xx[j]," ",real(eigvecs[j,i])," ",imag(eigvecs[j,i])," ",
            real(eigvecs[j+size_r,i])," ",imag(eigvecs[j+size_r,i]),"\n"))
        end
        close(f_eigvec)
    end
    close(f_eigval)

end

function test_linear()
    #the result of using nep-pack is the same as python code in linear case
    #dimension = 129
    dimension = 388
    #dimension = 443
    A = npzread("A.npy")
    B = npzread("B.npy")
    A1 = zeros(dimension,dimension)
    nep = PEP([A,A1,-B])

    eigvals,eigvecs = iar(nep,maxit=100,σ=0.01,neigs=1)

#=
    out = compute_Mder(nep,eigvals[1])/1e5
    err = (A-B*eigvals[1]^2)/1e5
    npzwrite("out.npy", out)
    npzwrite("err.npy", err)
    print(det(out),'\n')
    print(det(err),'\n')
=#

    print(eigvals,"\n")
    print(eigvals/2/pi*1000)
end

function sqzx(xi,Ki,orderi)
    Npi = orderi+1
    #print(xi,"\n")
    xx = collect(Iterators.flatten(xi))
    #print(size(xi))
    posM = Npi:Npi:Npi*(Ki-1)
    #print(Npi," ",Ki,"\n")
    #print(size(posM))
    #print(xi,"\n")
    deleteat!(xx,posM)
    return xx
end
