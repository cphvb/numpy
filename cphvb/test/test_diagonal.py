import numpy as np
import dnumpytest
import random

def run():
    niters = 10
    for i in xrange(niters):
        for j in xrange(niters):
            src = dnumpytest.random_list([i+1,j+1])
            Ad = np.array(src, dtype=float, dist=True)
            Af = np.array(src, dtype=float, dist=False)
            Cd = Ad.diagonal()
            Cf = Af.diagonal()
            if not dnumpytest.array_equal(Cd,Cf):
                raise Exception("Uncorrect result matrix\n")

if __name__ == "__main__":
    run()
