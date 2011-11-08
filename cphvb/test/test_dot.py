import numpy as np
import dnumpytest
import random

def run():
    niter = 6
    for m in range(2,niter+2):
        for n in range(2,niter+2):
            for k in range(2,niter+2):
                Asrc = dnumpytest.random_list([k,m])
                Bsrc = dnumpytest.random_list([m,k])
                Ad = np.array(Asrc, dtype=float, dist=True)
                Af = np.array(Asrc, dtype=float, dist=False)
                Bd = np.array(Bsrc, dtype=float, dist=True)
                Bf = np.array(Bsrc, dtype=float, dist=False)
                Cd = np.dot(Ad,Bd)
                Cf = np.dot(Af,Bf)
                if not dnumpytest.array_equal(Cd,Cf):
                    raise Exception("Uncorrect result matrix\n")

if __name__ == "__main__":
    run()
