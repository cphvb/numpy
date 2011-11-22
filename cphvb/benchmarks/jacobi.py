import sys
import numpy as np
import time
import cphvbnumpy

def cphvb(ary):
    if DIST:
        cphvbnumpy.handle_array(ary)

def jacobi(A, B, tol=0.005, forcedIter=0):
    '''itteratively solving for matrix A with solution vector B
       tol = tolerance for dh/h
       init_val = array of initial values to use in the solver
    '''
    cphvb(A)
    cphvb(B)
    h = np.zeros(np.shape(B), float)
    cphvb(h)
    dmax = 1.0
    n = 0
    tmp0 = np.empty(np.shape(A), float)
    cphvb(tmp0)
    tmp1 = np.empty(np.shape(B), float)
    cphvb(tmp1)
    AD = np.diagonal(A)
    t1 = time.time()
    while (forcedIter and forcedIter > n) or \
          (forcedIter == 0 and dmax > tol):
        n += 1
        np.multiply(A,h,tmp0)
        #np.add.reduce(tmp0,1,out=tmp1)
        #cphvb(tmp1)
        tmp2 = AD
        np.subtract(B, tmp1, tmp1)
        np.divide(tmp1, tmp2, tmp1)
        hnew = h + tmp1
        np.subtract(hnew,h,tmp2)
        np.divide(tmp2,h,tmp1)
        np.absolute(tmp1,tmp1)
        #dmax = np.maximum.reduce(tmp1)
        dmax = 0
        h = hnew

    t2 = time.time()
    print 'Iter: ', n, ' size:', np.shape(A), " time:", t2-t1

    return h
DIST = int(sys.argv[1])
size = int(sys.argv[2])
iter = int(sys.argv[3])

#A = array([[4, -1, -1, 0], [-1, 4, 0, -1], [-1, 0, 4, -1], [0, -1, -1, 4]], float, dist=d)
#B = array([1,2,0,1], float, dist=d)

A = np.random.random((size,size))
B = np.random.random((size))

C = jacobi(A, B, forcedIter=iter)

