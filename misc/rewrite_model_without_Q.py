'''
Read planetary model, set Q to 0, and re-write.
'''

import argparse
import os

from Ouroboros.common import load_model_full, write_model

def main():

    # Parse input arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("path_model")
    #
    input_args = parser.parse_args()
    path_model_in = input_args.path_model

    # Read model.
    model = load_model_full(path_model_in)

    # Get name of output model.
    dir_, file_in = os.path.split(path_model_in)
    name_in, ext = os.path.splitext(file_in)
    # 
    name_out = '{:}_noq'.format(name_in)
    file_out = '{:}{:}'.format(name_out, ext)
    path_out = os.path.join(dir_, file_out)

    # Change Q values to zero.
    model['header'] = '{:} no Q'.format(model['header'])
    model['Q_ka'] = 0.0*model['Q_ka']
    model['Q_mu'] = 0.0*model['Q_mu']

    # Write new model.
    write_model(model, path_out, model['header'])

    return

if __name__ == '__main__':

    main()
