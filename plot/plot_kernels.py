import matplotlib.pyplot as plt
import numpy as np

from common import load_model
from kernels import get_kernels_spheroidal, get_gravity_info
from plot.plot_common import get_r_fluid_solid_boundary, prep_RadialPNM_info
from post.read_output import load_eigenfreq_RadialPNM, load_eigenfunc_RadialPNM

def interp_n_parts(r, r_model, x_model, i_fluid_solid_boundary, i_fluid_solid_boundary_model):

    n_parts = len(i_fluid_solid_boundary) + 1
    assert n_parts == (len(i_fluid_solid_boundary_model) + 1)
    
    i_fluid_solid_boundary = list(i_fluid_solid_boundary)
    i_fluid_solid_boundary.insert(0, 0)
    i_fluid_solid_boundary.append(None)

    i_fluid_solid_boundary_model = list(i_fluid_solid_boundary_model)
    i_fluid_solid_boundary_model.insert(0, 0)
    i_fluid_solid_boundary_model.append(None)
    
    x_list = []
    for i in range(n_parts):

        i0 = i_fluid_solid_boundary[i]
        i1 = i_fluid_solid_boundary[i + 1]

        i0_model = i_fluid_solid_boundary_model[i]
        i1_model = i_fluid_solid_boundary_model[i + 1]

        x_i = np.interp(r[i0 : i1], r_model[i0_model : i1_model], x_model[i0_model : i1_model])
        x_list.append(x_i)

    x = np.concatenate(x_list)

    #print(i_solid_fluid_boundary)
    #print(r[0 : i_solid_fluid_boundary[0]])
    #print(r[i_solid_fluid_boundary[0] : i_solid_fluid_boundary[1]])
    #print(r[i_solid_fluid_boundary[1] : ])

    #print('\n')
    ## Interpolate the model parameters at the eigenfunction nodes.
    ##vs  = interp_3_parts(r, r_model, vs_model,  i_icb, i_cmb, i_icb_model, i_cmb_model)
    ##print(vs_model[0 : i_solid_fluid_boundary[0]])
    ##print(vs_model[i_solid_fluid_boundary[0]: i_solid_fluid_boundary[1]])
    ##print(vs_model[i_solid_fluid_boundary[1]:])

    #x_inner_core    = np.interp(r[0     : i_icb], r_model[0           : i_icb_model], x_model[0           : i_icb_model])
    #x_outer_core    = np.interp(r[i_icb : i_cmb], r_model[i_icb_model : i_cmb_model], x_model[i_icb_model : i_cmb_model])
    #x_mantle        = np.interp(r[i_cmb :      ], r_model[i_cmb_model :            ], x_model[i_cmb_model :])

    #x = np.concatenate([x_inner_core, x_outer_core, x_mantle])

    return x

# -----------------------------------------------------------------------------
def plot_kernel(RadialPNM_info, mode_type, n, l, i_toroidal = None, ax = None):

    # Get model information for axis limits, scaling and horizontal lines.
    model_data, shape, r_model, rho_model, vp_model, vs_model, mu_model, ka_model = load_model(RadialPNM_info['path_model'])
    # Convert to km.
    r_model = r_model*1.0E3
    
    # r_srf Radius of planet.
    # r_solid_fluid_boundary    List of radii of solid-fluid boundaries.
    r_srf = r_model[-1]
    if np.all(vs_model == 0.0):

        raise NotImplementedError

    elif np.all(vs_model > 0.0):

        raise NotImplementedError

    i_fluid_model, r_solid_fluid_boundary_model, i_fluid_solid_boundary_model =\
        get_r_fluid_solid_boundary(r_model, vs_model)
    
    # Get frequency information.
    f = load_eigenfreq_RadialPNM(RadialPNM_info, mode_type, n_q = n, l_q = l, i_toroidal = i_toroidal)

    # Get eigenfunction information.
    if mode_type == 'S': 

        r, U, V = load_eigenfunc_RadialPNM(RadialPNM_info, mode_type, n, l)

    elif mode_type == 'T':

        r, W = load_eigenfunc_RadialPNM(RadialPNM_info, mode_type, n, l, i_toroidal = i_toroidal)

    
    # Find indices of solid-fluid boundaries.
    i_fluid_solid_boundary = (np.where(np.diff(r) == 0.0))[0] + 1

    # Interpolate from the model grid to the output grid.
    rho = interp_n_parts(r, r_model, rho_model, i_fluid_solid_boundary, i_fluid_solid_boundary_model)
    vp  = interp_n_parts(r, r_model, vp_model,  i_fluid_solid_boundary, i_fluid_solid_boundary_model)
    vs  = interp_n_parts(r, r_model, vs_model,  i_fluid_solid_boundary, i_fluid_solid_boundary_model)

    fig = plt.figure()
    ax = plt.gca()

    ax.plot(r, rho)
    ax.plot(r, vp)
    ax.plot(r, vs)

    plt.show()

    #print(i_solid_fluid_boundary)
    #print(r[0 : i_solid_fluid_boundary[0]])
    #print(r[i_solid_fluid_boundary[0] : i_solid_fluid_boundary[1]])
    #print(r[i_solid_fluid_boundary[1] : ])

    #print('\n')
    ## Interpolate the model parameters at the eigenfunction nodes.
    ##vs  = interp_3_parts(r, r_model, vs_model,  i_icb, i_cmb, i_icb_model, i_cmb_model)
    ##print(vs_model[0 : i_solid_fluid_boundary[0]])
    ##print(vs_model[i_solid_fluid_boundary[0]: i_solid_fluid_boundary[1]])
    ##print(vs_model[i_solid_fluid_boundary[1]:])

    # Get fluid-solid boundaries for r.
    import sys
    sys.exit()

    # Convert to Mineos normalisation.
    ratio = 1.0E-3*(r_srf**2.0)
    if mode_type == 'S':

        U = U*ratio
        V = V*ratio

    elif mode_type == 'T':

        W= W*ratio

    # Calculate the kernels.
    if mode_type == 'S':

        g_model, g, P = get_gravity_info(r_model, rho_model, r, U, V, l, rho)
        K_ka, K_mu, K_rho, K_alpha, K_beta, K_rhop = \
            get_kernels_spheroidal(f, r, U, V, l, g, rho, vp, vs, P, i_fluid = i_fluid)

    # Create the title.
    title = '$_{{{:d}}}${:}$_{{{:d}}}$'.format(n, mode_type, l)

    return

# Main. -----------------------------------------------------------------------
def main():

    # Read the input file and command-line arguments.
    RadialPNM_info, mode_type, n, l, i_toroidal = prep_RadialPNM_info()

    # Plot the kernel.
    plot_kernel(RadialPNM_info, mode_type, n, l, i_toroidal = i_toroidal, ax = None)

if __name__ == '__main__':

    main()
