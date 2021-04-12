import os

from    geographiclib.geodesic  import  Geodesic
import numpy as np

from Ouroboros.misc.cmt_io import read_mineos_cmt

def gc_path_points_by_delta(lon0, lat0, azi, delta, geodesic_object = 'sphere_earth'):

    # Check input geodesic object.
    if geodesic_object == 'sphere_earth':

        # A hack--use a very small value of flattening because 0 flattening
        # (a sphere) is not supported.
        geodesic_object = Geodesic(6371.0E3, 1.0E-300)

    elif geodesic_object == 'wgs84':

        geodesic_object = Geodesic.WGS84

    else:

        assert isinstance(geodesic_object, Geodesic) 
    
    # Prepare output array.
    n_pts = len(delta)
    section = np.zeros((3, n_pts))
    
    # Generate the coordinates of the cross-section.
    # Create geodesic and great circle objects.
    gc      = geodesic_object.Line(lat0, lon0, azi)
    
    # Calculate arc length on surface.
    circumference = 2.0*np.pi*6371.0E3 # m
    arc_length = circumference*delta/(2.0*np.pi)

    # Find coordinates of each point in cross-section.
    for i in range(n_pts):
                
        # Calculate coordinates of this point of the section.
        point           = gc.Position(arc_length[i])
        
        # Assign outputs to array.
        # Convert distance to kilometres.
        section[0, i]     = arc_length[i]*1.0E-3 
        section[1, i]     = point['lon2']
        section[2, i]     = point['lat2']

    return section

def gc_path_points_through_antipode(lon0, lat0, azi, half_n = 100, geodesic_object = 'sphere_earth'):

    # Check input geodesic object.
    if geodesic_object == 'sphere_earth':

        # A hack--use a very small value of flattening because 0 flattening
        # (a sphere) is not supported.
        geodesic_object = Geodesic(6371.0E3, 1.0E-300)

    elif geodesic_object == 'wgs84':

        geodesic_object = Geodesic.WGS84

    else:

        assert isinstance(geodesic_object, Geodesic) 

    # Prepare output array.
    section_0 = np.zeros((3, half_n))
    
    # Generate the coordinates of the cross-section.
    # Create geodesic and great circle objects.
    gc      = geodesic_object.Line(lat0, lon0, azi)
    #
    # Get antipode coordinates.
    lon1 = lon0 + 180.0
    lat1 = -1.0*lat0 
    #
    gd      = geodesic_object.Inverse(lat1, lon1, lat0, lon0)

    # Find coordinates of each point in cross-section.
    for i in range(half_n):
                
        # Calculate distance of i_th point along arc in metres.
        distance        = (gd['s12']/(half_n - 1))*i
        
        # Calculate coordinates of this point of the section.
        point           = gc.Position(distance)
        
        # Assign outputs to array.
        # Convert distance to kilometres.
        section_0[0, i]     = distance*1.0E-3 
        section_0[1, i]     = point['lon2']
        section_0[2, i]     = point['lat2']

    #section_0[0, :] = gd['s12']*1.0E-3 - section_0[0, :]
    #section_0 = section_0[:, ::-1]

    # Repeat for opposite direction.
    # Prepare output array.
    section_1 = np.zeros((3, half_n))
   
    # Generate the coordinates of the cross-section.
    # Create geodesic and great circle objects.
    gc      = geodesic_object.Line(lat0, lon0, azi + 180.0)
    #
    # Find coordinates of each point in cross-section.
    for i in range(half_n):
                
        # Calculate distance of i_th point along arc in metres.
        distance        = (gd['s12']/(half_n - 1))*i
        
        # Calculate coordinates of this point of the section.
        point           = gc.Position(distance)
        
        # Assign outputs to array.
        # Convert distance to kilometres.
        section_1[0, i]     = distance*1.0E-3 
        section_1[1, i]     = point['lon2']
        section_1[2, i]     = point['lat2']

    #section_1[0, :] = gd['s12']*1.0E-3 + section_1[0, :]
    #section_1 = section_1[:, ::-1]

    section_1[0, :] = section_1[0, :]*-1.0

    # Merge the two.
    section_1 = section_1[:, 1 : -1]
    section_1 = section_1[:, ::-1]
    section = np.concatenate([section_0, section_1], axis = 1)

    return section

def read_radiation_pattern(path_in):
    
    # Note: Doesn't seem to recognise comments #.
    n_header = 10 
    data = np.loadtxt(path_in, skiprows = n_header, delimiter = '|')

    n_azi = 360
    n_obs, n_vars = data.shape
    n_freqs = n_obs//360

    freqs = np.zeros(n_freqs)

    radiation_info = dict()
    radiation_info['azi_span'] = data[0 : n_azi, 1]
    for i in range(n_freqs):
        
        j0 = n_azi*i
        j1 = n_azi*(i + 1)
        
        f = np.median(data[j0 : j1, 0])

        amp_rayl = data[j0 : j1, 2]
        phs_rayl = data[j0 : j1, 3] 
        amp_love = data[j0 : j1, 4]
        phs_love = data[j0 : j1, 5] 

        data_i = {'freq'      : f,
                'amp_rayl'  : amp_rayl,
                'phs_rayl'  : phs_rayl,
                'amp_love'  : amp_love,
                'phs_love'  : phs_love }

        name = 'band_{:>1d}'.format(i)
        radiation_info[name] = data_i

    return radiation_info

def make_grid_spokes():

    dir_input = '/Users/hrmd_work/Documents/research/stoneley/input/cmts_merged/'
    i = 0 # Identify which CMT soutce.

    # Read radiation pattern.
    path_radiation_pattern = os.path.join(dir_input, 'pattern_{:>05d}.txt'.format(i))
    radiation_info = read_radiation_pattern(path_radiation_pattern)

    az = radiation_info['azi_span']
    amp = radiation_info['band_0']['amp_rayl']

    i_max_amp = np.argmax(amp)
    az_max_amp = az[i_max_amp]

    # Read point-source moment tensor.
    path_point_source = os.path.join(dir_input, 'point_{:>05d}.txt'.format(i))
    point_cmt = read_mineos_cmt(path_point_source)
    # Get centroid coordinates.
    lon0 = point_cmt['lon_centroid']
    lat0 = point_cmt['lat_centroid']
    
    # Get great circle path points.
    points = gc_path_points_through_antipode(lon0, lat0, az_max_amp,
            half_n = 100,
            geodesic_object = 'sphere_earth')
    
    # Define depths.
    r_srf = 6371.0
    r_cmb = 3480.0
    r_icb = 1221.5
    #
    dr_mantle = r_srf - r_cmb
    dr_outer_core = r_cmb - r_icb
    dr_inner_core = r_icb
    #
    r_spacing_appx = 150.0
    #
    n_r_mantle = int(np.round(dr_mantle/r_spacing_appx)) + 1
    n_r_outer_core = int(np.round(dr_outer_core/r_spacing_appx)) + 1
    n_r_inner_core = int(np.round(dr_inner_core/r_spacing_appx)) + 1
    #
    r_mantle        = np.linspace(r_srf, r_cmb, num = n_r_mantle)
    r_outer_core    = np.linspace(r_cmb, r_icb, num = n_r_outer_core)
    r_inner_core    = np.linspace(r_icb, 0.0, num = n_r_inner_core)
    #
    # Apply small shift at boundaries to avoid interpolation ambiguities and
    # and avoid singularity at centre of Earth.
    buff = 1.0
    r_mantle[-1]     = r_mantle[-1] + 1.0
    r_outer_core[0]  = r_outer_core[0] - 1.0
    r_outer_core[-1] = r_outer_core[-1] + 1.0
    r_inner_core[0]  = r_inner_core[0] - 1.0
    r_inner_core[-1] = r_inner_core[-1] + 1.0
    #
    r_span = np.concatenate([r_mantle, r_outer_core, r_inner_core])
    z_span = -1.0*(r_srf - r_span)
    n_z = len(z_span)

    # Write a station file.
    fmt_station = '{:>05d} {:>+8.4f} {:>+9.4f} {:>10.8f} \'Station description\''
    line_comp = '@   Z  0.0  0.0  0.0'
    n_x = points.shape[1]
    #
    dir_output = '/Users/hrmd_work/Documents/research/stoneley/input/mineos/station_lists/'
    path_station_list = os.path.join(dir_output, 'great_circle_{:>03d}_station_list.txt'.format(i))
    print('Writing {:}'.format(path_station_list))
    p = 0
    with open(path_station_list, 'w') as out_id:

        for j in range(n_x):
            
            for k in range(n_z):

                
                station_string = fmt_station.format(p, points[2, j], points[1, j], 1.0E-3*z_span[k])

                out_id.write(station_string + '\n')
                out_id.write(line_comp + '\n')

                p = p + 1

    #
    path_x_profile = os.path.join(dir_output, 'great_circle_{:>03d}_x_profile.txt'.format(i))
    print('Writing {:}'.format(path_x_profile))
    with open(path_x_profile, 'w') as out_id:

        for j in range(n_x):

            out_id.write('{:>5d} {:>+8.4f} {:>9.4f} {:>16.8e}\n'.format(j, points[1, j], points[2, j], points[0, j]))

    path_z_profile = os.path.join(dir_output, 'great_circle_{:>03d}_z_profile.txt'.format(i))
    print('Writing {:}'.format(path_z_profile))
    with open(path_z_profile, 'w') as out_id:

        for k in range(n_z):

            out_id.write(('{:>5d} {:>16.8e}\n'.format(k, -1.0*z_span[k])))

    return

def make_grid_even():

    dir_input = '/Users/hrmd_work/Documents/research/stoneley/input/cmts_merged/'
    i = 0 # Identify which CMT soutce.

    # Read radiation pattern.
    path_radiation_pattern = os.path.join(dir_input, 'pattern_{:>05d}.txt'.format(i))
    radiation_info = read_radiation_pattern(path_radiation_pattern)

    az = radiation_info['azi_span']
    amp = radiation_info['band_0']['amp_rayl']

    i_max_amp = np.argmax(amp)
    az_max_amp = az[i_max_amp]

    # Read point-source moment tensor.
    path_point_source = os.path.join(dir_input, 'point_{:>05d}.txt'.format(i))
    point_cmt = read_mineos_cmt(path_point_source)
    # Get centroid coordinates.
    lon0 = point_cmt['lon_centroid']
    lat0 = point_cmt['lat_centroid']

    spacing = 100.0 # km
    #spacing = 500.0 # km
    
    # Define depths.
    r_srf = 6371.0
    r_cmb = 3480.0
    r_icb = 1221.5
    #
    dr_mantle = r_srf - r_cmb
    dr_outer_core = r_cmb - r_icb
    dr_inner_core = r_icb
    #
    r_spacing_appx = spacing
    #
    n_r_mantle = int(np.round(dr_mantle/r_spacing_appx)) + 1
    n_r_outer_core = int(np.round(dr_outer_core/r_spacing_appx)) + 1
    n_r_inner_core = int(np.round(dr_inner_core/r_spacing_appx)) + 1
    #
    r_mantle        = np.linspace(r_srf, r_cmb, num = n_r_mantle)
    r_outer_core    = np.linspace(r_cmb, r_icb, num = n_r_outer_core)
    r_inner_core    = np.linspace(r_icb, 0.0, num = n_r_inner_core)
    #
    # Apply small shift at boundaries to avoid interpolation ambiguities and
    # and avoid singularity at centre of Earth.
    buff = 1.0
    r_mantle[-1]     = r_mantle[-1] + 1.0
    r_outer_core[0]  = r_outer_core[0] - 1.0
    r_outer_core[-1] = r_outer_core[-1] + 1.0
    r_inner_core[0]  = r_inner_core[0] - 1.0
    #
    r_span = np.concatenate([r_mantle, r_outer_core, r_inner_core])

    n_r = len(r_span)

    half_or_full_circle = 'half'
    
    circumference = 2.0*np.pi*r_span
    if half_or_full_circle == 'full':

        n_pts_around = (np.round(circumference/r_spacing_appx)).astype(np.int)
        n_pts = np.sum(n_pts_around) + 1

    elif half_or_full_circle == 'half':

        n_pts_around = (np.round(0.5*circumference/r_spacing_appx)).astype(np.int) + 1
        n_pts = np.sum(n_pts_around)

    r = np.zeros(n_pts)
    delta = np.zeros(n_pts)
    j0 = 0
    for k in range(n_r - 1):
        
        j1 = j0 + n_pts_around[k] 

        if half_or_full_circle == 'full':

            delta[j0 : j1] = np.linspace(-np.pi, np.pi, num = n_pts_around[k] + 1)[:-1]

        elif half_or_full_circle == 'half':

            delta[j0 : j1] = np.linspace(0.0, np.pi, num = n_pts_around[k])

        r[j0 : j1] = r_span[k]

        j0 = j1

    show_points = False
    if show_points:

        x = r*np.cos(delta)
        y = r*np.sin(delta)

        import matplotlib.pyplot as plt
        fig = plt.figure(figsize = (10.0, 6.0))
        ax  = plt.gca()

        ax.scatter(x, y, alpha = 0.5)

        ax.set_aspect(1.0)

        plt.show()

    section = gc_path_points_by_delta(lon0, lat0, az_max_amp, delta)

    delta_m = section[0, :]
    lon = section[1, :]
    lat = section[2, :]
    
    z = -1.0*(6371.0 - r)

    # Write a station file.
    fmt_station = '{:>05d} {:>+8.4f} {:>+9.4f} {:>10.8f} \'Station description\''
    line_comp = '@   Z  0.0  0.0  0.0'
    #
    dir_output = '/Users/hrmd_work/Documents/research/stoneley/input/mineos/station_lists/'
    path_station_list = os.path.join(dir_output, 'great_circle_{:>03d}_station_list.txt'.format(i))
    print('Writing {:}'.format(path_station_list))
    with open(path_station_list, 'w') as out_id:

        for j in range(n_pts):

                station_string = fmt_station.format(j, lat[j], lon[j], 1.0E-3*z[j])

                out_id.write(station_string + '\n')
                out_id.write(line_comp + '\n')

    path_pts = os.path.join(dir_output, 'great_circle_{:>03d}_point_list.txt'.format(i))
    # lon_deg lat_deg delta_rad radius_km
    fmt_pts = '{:>5d} {:>+18.12e} {:>+18.12e} {:>+18.12e} {:+18.12e}\n'
    print('Writing {:}'.format(path_pts))
    with open(path_pts, 'w') as out_id:

        for j in range(n_pts):
        
            out_id.write(fmt_pts.format(j, lon[j], lat[j], delta[j], r[j]))

    return

def main():

    grid_type = 'even'
    assert grid_type in ['spokes', 'even']

    if grid_type == 'spokes':

        make_grid_spokes()

    elif grid_type == 'even':

        make_grid_even()

    return

if __name__ == '__main__':

    main()
