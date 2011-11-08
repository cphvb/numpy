import numpy as np
import dnumpytest
import random

def run():
    max_ndim = 6
    for i in range(1,max_ndim+1):
        src = dnumpytest.random_list(range(2, i+2))
        Ad = np.array(src, dtype=float, dist=True)
        Af = np.array(src, dtype=float, dist=False)
        for j in range(len(Ad.shape)):
            Cd = np.add.reduce(Ad,j)
            Cf = np.add.reduce(Af,j)
            if not dnumpytest.array_equal(Cd,Cf):
                raise Exception("Uncorrect result array\n")
    return (False, "")

if __name__ == "__main__":
    run()
