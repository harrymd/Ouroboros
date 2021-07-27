from Ouroboros.modes.compute_modes import process_eigen_anelastic

def main():

    #process_eigen_anelastic('/Users/hrmd_work/Documents/research/anelasticity/modes/output/model_homogenous_00050_full_an/00005_00003/T/', 'T', 1, i_toroidal = 0)
    #process_eigen_anelastic('/Users/hrmd_work/Documents/research/anelasticity/modes/output/model_homogenous_yuen_1982_00050_SLS_uniform/00005_00002/grav_1/S/', 'S', 2, 2, 2)
    process_eigen_anelastic('/Users/hrmd_work/Documents/research/anelasticity/modes/output/prem_noocean_00300_SLS/00005_00002/T/', 'T', 2, 2, 1, i_toroidal = 2)

if __name__ == '__main__':

    main()
