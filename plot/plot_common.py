import os
import sys
from itertools import groupby
from operator import itemgetter

import numpy as np

from common import read_RadialPNM_input_file, read_Mineos_input_file

def prep_run_info(args):

    # Rename input arguments.
    n = args.n
    l = args.l
    i_toroidal = args.layer_number
    use_mineos = args.mineos

    # Read the input file.
    if use_mineos:

        path_model, dir_output, g_switch, mode_type, l_min, l_max, n_min, n_max = \
        read_Mineos_input_file()

    else:

        path_model, dir_output, g_switch, mode_type, l_min, l_max, n_min, n_max, n_layers = \
        read_RadialPNM_input_file()

    if mode_type == 'T':

        g_switch = 0
        i_toroidal = layer_number

    # Get the name of the model from the file.
    name_model = os.path.splitext(os.path.basename(path_model))[0]

    # Store n- and l- limits as arrays.
    n_lims      = [n_min, n_max]
    l_lims      = [l_min, l_max]

    run_info = {        'use_mineos': use_mineos,
                        'dir_output': dir_output,
                        'model'     : name_model,
                        'n_lims'    : n_lims,
                        'l_lims'    : l_lims,
                        'g_switch'  : g_switch,
                        'path_model': path_model}

    if not use_mineos:

        run_info['n_layers'] = n_layers

    return run_info, mode_type, n, l, i_toroidal

def prep_RadialPNM_info_old():

    # Read the input file.
    path_model, dir_output, g_switch, mode_type, l_min, l_max, n_min, n_max, n_layers = \
        read_RadialPNM_input_file()

    if mode_type == 'T':

        g_switch = 0
        assert len(sys.argv) == 4, 'Usage for plotting toroidal modes: python3 script_name.py n l i, where i is the layer number.'

        # Get command-line argument.
        i_toroidal = int(sys.argv[3])

    else:
        
        assert len(sys.argv) == 3, 'Usage for plotting radial/spheroidal modes: python3 script_name.py n l'
        i_toroidal = None

    # Get command-line arguments.
    n = int(sys.argv[1])
    l = int(sys.argv[2])

    # Get the name of the model from the file.
    name_model = os.path.splitext(os.path.basename(path_model))[0]

    # Store n- and l- limits as arrays.
    n_lims      = [n_min, n_max]
    l_lims      = [l_min, l_max]

    RadialPNM_info = {  'dir_output': dir_output,
                        'model'     : name_model,
                        'n_layers'  : n_layers,
                        'n_lims'    : n_lims,
                        'l_lims'    : l_lims,
                        'g_switch'  : g_switch,
                        'path_model': path_model}

    return RadialPNM_info, mode_type, n, l, i_toroidal

def get_r_fluid_solid_boundary(radius, vs):

    n_layers = len(radius)
    i_fluid = np.where(vs == 0.0)[0]
    
    i_solid_fluid_boundary = []
    # This is based on https://docs.python.org/2.6/library/itertools.html#examples,
    # second example in second 9.7.2.
    for k, g in groupby(enumerate(i_fluid), lambda ix: ix[0] - ix[1]):

        i_fluid_subsection = list(map(itemgetter(1), g))
        if (i_fluid_subsection[0] != 0) and (i_fluid_subsection[0] != (n_layers - 1)):

            i_solid_fluid_boundary.append(i_fluid_subsection[0])

        if (i_fluid_subsection[-1] != 0) and (i_fluid_subsection[-1] != (n_layers - 1)):

            #i_solid_fluid_boundary.append(i_fluid_subsection[-1])
            i_solid_fluid_boundary.append(i_fluid_subsection[-1] + 1)

    r_solid_fluid_boundary = [radius[i] for i in i_solid_fluid_boundary]

    return i_fluid, r_solid_fluid_boundary, i_solid_fluid_boundary
