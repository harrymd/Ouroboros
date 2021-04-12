import os

from Ouroboros.misc.cmt_io import read_ndk, write_mineos_cmt

def main():

    path_catalog = '/Users/hrmd_work/Documents/research/stoneley/input/global_cmt_ndk_list.txt'
    dir_output = '/Users/hrmd_work/Documents/research/stoneley/input/global_cmt/'
    i_max = 10000 

    #scalar_moment_thresh = 1.0E28
    scalar_moment_thresh = 1.0E27
    
    i = 0
    path_tmp_ndk = 'tmp_cmt.ndk'
    with open(path_catalog, 'r') as in_id:

        n_chunk = 5 
        while i < i_max: 
            
            lines = []
            for j in range(n_chunk):
                
                line = in_id.readline()
                lines.append(line)

            with open(path_tmp_ndk, 'w') as out_id:

                for line in lines:

                    out_id.write(line)

            cmt = read_ndk(path_tmp_ndk)
            print(cmt['datetime_ref'])

            scalar_moment = cmt['scalar_moment']*(10.0**cmt['exponent'])

            if scalar_moment > scalar_moment_thresh:

                file_out = '{:>05d}.txt'.format(i)
                path_out = os.path.join(dir_output, file_out)
                print("Writing {:}".format(path_out))
                write_mineos_cmt(path_out, cmt)

                i = i + 1

    return

if __name__ == '__main__':

    main()

