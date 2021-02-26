from Ouroboros.misc.cmt_io import read_ndk

def main():

    path_catalog = '/Users/hrmd_work/Documents/research/stoneley/input/global_cmt_ndk_list.txt'
    print(path_catalog)

    scalar_moment_thresh = 1.0E28
    
    path_tmp_ndk = 'tmp_cmt.ndk'
    with open(path_catalog, 'r') as in_id:

        n_chunk = 5 
        while True: 
            
            lines = []
            for j in range(n_chunk):
                
                line = in_id.readline()
                lines.append(line)

            with open(path_tmp_ndk, 'w') as out_id:

                for line in lines:

                    out_id.write(line)

            cmt = read_ndk(path_tmp_ndk)

            scalar_moment = cmt['scalar_moment']*(10.0**cmt['exponent'])

            if scalar_moment > scalar_moment_thresh:

                print(cmt['datetime_ref'])
    
    return

if __name__ == '__main__':

    main()

