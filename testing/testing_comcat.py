# stdlib imports
from datetime import datetime
import io

# Third party imports
import matplotlib.pyplot as plt
import numpy as np
#from PIL import Image

# Local imports
from libcomcat.search import get_event_by_id, search

def main():
    '''
    https://github.com/usgs/libcomcat/blob/master/notebooks/Classes.ipynb
    '''

    earthquakes = search(   starttime = datetime(1900, 1, 1, 0, 0),
                            endtime =   datetime(2100, 1, 1, 0, 0),
                            minmagnitude = 8.0,
                            orderby = 'time', limit = 2)

    #print(earthquakes)
    #import sys
    #sys.exit()
    #print("%s occurred on %s with a magnitude of %s and depth of %s" %
    #     (earthquake.id, earthquake.time, earthquake.magnitude,
    #      earthquake.depth))

    ## Check for product
    #product = earthquake.hasProduct('shakemap')
    #print('Includes "shakemap" as a product: %s' % product)
    earthquake = earthquakes[0]
    detail = earthquake.getDetailEvent()
    finite_fault = detail.getProducts('finite-fault')[0]
    cmt_text = finite_fault.getContentBytes('CMTSOLUTION')[0]
    tmp_file = 'test.tmp'
    print(cmt_text)
    with open(tmp_file, 'wb') as out_id:

        out_id.write(cmt_text)

    #print(finite_fault.contents)

    #print(earthquake._jdict['properties']['types'])

    return

if __name__ == '__main__':

    main()
