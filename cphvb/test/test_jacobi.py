from numpy import *
import dnumpytest

def jacobi(A, B, tol=0.005):
    '''itteratively solving for matrix A with solution vector B
       tol = tolerance for dh/h
       init_val = array of initial values to use in the solver
    '''
    h = zeros(shape(B), float, dist=A.dist())
    dmax = 1.0
    n = 0
    tmp0 = empty(shape(A), float, dist=A.dist())
    tmp1 = empty(shape(B), float, dist=A.dist())
    AD = diagonal(A)
    while dmax > tol:
        n += 1
        multiply(A,h,tmp0)
        add.reduce(tmp0,1,out=tmp1)
        tmp2 = AD
        subtract(B, tmp1, tmp1)
        divide(tmp1, tmp2, tmp1)
        hnew = h + tmp1
        subtract(hnew,h,tmp2)
        divide(tmp2,h,tmp1)
        absolute(tmp1,tmp1)
        dmax = maximum.reduce(tmp1)
        h = hnew
    return h

def run():
    A = load("%sJacobi_Amatrix.npy"%dnumpytest.DataSetDir, dist=True)
    B = load("%sJacobi_Bvector.npy"%dnumpytest.DataSetDir, dist=True)
    C = load("%sJacobi_Cvector.npy"%dnumpytest.DataSetDir, dist=True)
    result = jacobi(A,B)

    if not dnumpytest.array_equal(C,result):
        raise Exception("Uncorrect result vector\n")

if __name__ == "__main__":
    run()
