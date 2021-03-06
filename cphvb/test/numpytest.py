#Test and demonstration of DistNumPy.
import numpy as np
import cphvbnumpy
import sys
import time
import subprocess
import os
import getopt
import random

#If not available some test may be ignored.
try:
    import scipy
    from scipy import linalg
except:
    pass
try:
    import zlib
except:
    pass
try:
    import pyHPC
except:
    pass

DataSetDir = os.path.join(os.path.join(\
             os.path.dirname(sys.argv[0]), "datasets"), "")

TmpSetDir = os.path.join("/",os.path.join("tmp", ""))

def array_equal(A,B,maxerror=0.0):
    if type(A) is not type(B):
        return False
    elif (not type(A) == type(np.array([]))) and (not type(A) == type([])):
        if A == B:
            return True
        else:
            return False
    A = A.flatten()
    B = B.flatten()
    if not len(A) == len(B):
        return False

    for i in range(len(A)):
        delta = abs(A[i] - B[i])
        if delta > maxerror:
            print "Delta error:",delta
            return False
    return True

def random_list(dims):
    (val,unique) = _random_list(dims)
    return val

def _random_list(dims, unique=1):
    if len(dims) == 0:
        return (unique, unique + 1)
    list = []
    for i in range(dims[-1]):
        (val,unique) = _random_list(dims[0:-1], unique + i)
        list.append(val)
    return (list, unique + dims[-1])


if __name__ == "__main__":
    pydebug = True
    seed = time.time()
    script_list = []
    exclude_list = []

    try:
        sys.gettotalrefcount()
    except AttributeError:
        pydebug = False

    try:
        opts, args = getopt.getopt(sys.argv[1:],"s:f:e:",["seed=", "file=", "exclude="])
    except getopt.GetoptError, err:
        print str(err)
        sys.exit(2)
    for o, a in opts:
        if o in ("-s", "--seed"):
            seed = int(a)
            random.seed(seed)
        elif o in ("-f", "--file"):
            script_list.append(a)
        elif o in ("-e", "--exclude"):
            exclude_list.append(a)
        else:
            assert False, "unhandled option"

    if len(script_list) == 0:
        script_list = os.listdir(\
                      os.path.dirname(os.path.abspath(__file__)))

    print "*"*100
    print "*"*31, "Testing Distributed Numerical Python", "*"*31
    cphvbnumpy.flush()
    for i in xrange(len(script_list)):
        f = script_list[i]
        if f.startswith("test_") and f.endswith("py")\
           and f not in exclude_list:
            m = f[:-3]#Remove ".py"
            m = __import__(m)
            print "*"*100
            print "Testing %s"%f
            err = False
            msg = ""
            r1 = 0; r2 = 0
            if pydebug:
                r1 = sys.gettotalrefcount()
            try:
                cphvbnumpy.flush()
                m.run()
                cphvbnumpy.flush()
            except:
                cphvbnumpy.flush()
                err = True
                msg = sys.exc_info()[1]
            if pydebug:
                r2 = sys.gettotalrefcount()
                if r2 != r1:
                    print "Memory leak - totrefcount: "\
                          "from %d to %d\n"%(r1,r2),
            if err:
                print "Error in %s! Random seed: %d - "\
                      "message: %s\n"%(f, seed, msg),
            else:
                print "Succes\n",
    cphvbnumpy.flush()
    print "*"*100
    print "*"*46, "Finish", "*"*46
    print "*"*100

