import sys
import numpy as np
import time
import cphvbnumpy

def cphvb(ary):
    if DIST:
        cphvbnumpy.handle_array(ary)

DIST=int(sys.argv[1])
W = int(sys.argv[2])
H = int(sys.argv[2])
forcedIter = int(sys.argv[3])

full = np.zeros((W+2,H+2), dtype=np.double)
work = np.zeros((W,H), dtype=np.double)
diff = np.zeros((W,H), dtype=np.double)
tmpdelta = np.zeros((W), dtype=np.double)

cphvb(full)
cphvb(work)
cphvb(diff)
cphvb(tmpdelta)

cells = full[1:-1, 1:-1]
up    = full[1:-1, 0:-2]
left  = full[0:-2, 1:-1]
right = full[2:  , 1:-1]
down  = full[1:-1, 2:  ]

full[:,0]  += -273.15
full[:,-1] += -273.15
full[0,:]  +=  40.0
full[-1,:] += -273.13

cphvbnumpy.flush()
t1 = time.time()
epsilon=W*H*0.010
delta=epsilon+1
i=0
print "HEJ"
while (forcedIter and forcedIter > i) or \
      (forcedIter == 0 and epsilon<delta):
  i+=1
  work[:] = cells
  work += up
  work += left
  work += right
  work += down
  work *= 0.2
  np.subtract(cells,work,diff)
  np.absolute(diff, diff)
  np.add.reduce(diff,out=tmpdelta)
  delta = np.add.reduce(tmpdelta)
  cells[:] = work

cphvbnumpy.flush()
timing = time.time() - t1

print 'jacobi_stencil - Iter: ', i, ' size:', np.shape(work), " time:", timing
