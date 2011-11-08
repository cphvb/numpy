import numpy as np
import dnumpytest

def jacobi_sencil(H,W,Dist):
    full = np.zeros((H+2,W+2), dtype=np.double, dist=Dist)
    work = np.zeros((H,W), dtype=np.double, dist=Dist)
    diff = np.zeros((H,W), dtype=np.double, dist=Dist)
    tmpdelta = np.zeros((H), dtype=np.double, dist=Dist)

    cells = full[1:-1, 1:-1]
    up    = full[1:-1, 0:-2]
    left  = full[0:-2, 1:-1]
    right = full[2:  , 1:-1]
    down  = full[1:-1, 2:  ]

    full[:,0]  += -273.15
    full[:,-1] += -273.15
    full[0,:]  +=  40.0
    full[-1,:] += -273.13

    epsilon=W*H*0.002
    delta=epsilon+1
    i=0
    while epsilon<delta:
      i+=1
      work[:] = cells
      work += up
      work += left
      work += right
      work += down
      work *= 0.2
      np.subtract(cells,work,diff)
      diff = np.absolute(diff)
      np.add.reduce(diff, out=tmpdelta)
      delta = np.add.reduce(tmpdelta)
      cells[:] = work
    return cells

def run():
    Seq = jacobi_sencil(5,5,False)
    Par = jacobi_sencil(5,5,True)

    if not dnumpytest.array_equal(Seq,Par):
        raise Exception("Uncorrect result matrix\n")

if __name__ == "__main__":
    print run()
