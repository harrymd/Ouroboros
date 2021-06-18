'''
https://stackoverflow.com/a/13055843/6731244
Finding multiple roots of function.
Method based on chapter 4.3 from 'Numerical Methods in Engineering with Python'
by Jaan Kiusalaas.
'''

import math

def rootsearch(f, a, b, dx):

    x1 = a
    f1 = f(a)
    x2 = a + dx
    f2 = f(x2)
    while (f1 * f2) > 0.0:

        if x1 >= b:

            return None, None

        x1 = x2
        f1 = f2
        x2 = x1 + dx
        f2 = f(x2)

    return x1, x2

def bisect(f, x1, x2, switch = 0, epsilon = 1.0e-9):

    f1 = f(x1)
    if f1 == 0.0:

        return x1

    f2 = f(x2)
    if f2 == 0.0:

        return x2

    if (f1 * f2) > 0.0:

        print('Root is not bracketed')
        return None

    n = int(math.ceil(math.log(abs(x2 - x1)/epsilon)/math.log(2.0)))
    for i in range(n):

        x3 = 0.5*(x1 + x2)
        f3 = f(x3)
        if (switch == 1) and (abs(f3) > abs(f1)) and (abs(f3) > abs(f2)):

            return None

        if f3 == 0.0:

            return x3

        if (f2 * f3) < 0.0:

            x1 = x3
            f1 = f3

        else:

            x2 =x3
            f2 = f3

    return (x1 + x2)/2.0

def roots(f, a, b, eps = 1e-6, n_roots_max = 100, verbose = True):
    
    if verbose:

        print ('The roots on the interval [%f, %f] are:' % (a,b))

    roots = []

    while len(roots) < n_roots_max:

        x1, x2 = rootsearch(f, a, b, eps)
        if x1 != None:

            a = x2
            root = bisect(f, x1, x2, 1)
            if root != None:
                
                roots.append(root)
                if verbose:

                    print (round(root, -int(math.log(eps, 10))))

        else:
            
            if verbose:

                print ('\nSearched entire interval.')

            break

    return roots
