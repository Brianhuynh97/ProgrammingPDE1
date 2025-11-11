import numpy as np
import scipy.sparse as sprs # type: ignore
import matplotlib.pyplot as plt # type: ignore
from scipy.integrate import quad  # type: ignore

N = 100 # number of points in grid

xs = np.linspace(0,1,N+2) # grid
h = 1/(N+1) # distance between each point

def f(x):
    return 1;

def phi(i,x):
    if xs[i] <= x <= xs[i+1]:
        return (x-xs[i])/h
    elif xs[i+1] <  x <= xs[i+2]:
        return (xs[i+2]-x)/h
    else:
        return 0

stfMat_lil = sprs.lil_matrix((N,N),dtype=float)
for i in range(N-1):
    stfMat_lil[i,i] = 2
    stfMat_lil[i+1,i] = -1
    stfMat_lil[i,i+1] = -1
stfMat_lil[N-1,N-1] = 2
stfMat_lil /= h
stfMat = stfMat_lil.tocsc()

rhs = np.zeros(N,dtype=float)
for i in range(N):
    rhs[i], err = quad(lambda x: f(x)*phi(i,x), xs[i], xs[i+2])

coeffs = sprs.linalg.spsolve(stfMat,rhs)

def aprxSol(x):
    i = np.searchsorted(xs,x)-1
    if i == 0:
        return coeffs[i]*phi(i,x)
    elif i == N:
        return coeffs[i-1]*phi(i-1,x)
    else:
        return coeffs[i-1]*phi(i-1,x) + coeffs[i]*phi(i,x)

def trueSol(x):
    return -(x-1/2)*(x-1/2)/2 + 1/8

xss = np.linspace(0,1,1000)
ys_aprx = np.array([aprxSol(x) for x in xss])
ys_true = trueSol(xss)

errL2, err = np.sqrt(quad(lambda x: abs(aprxSol(x)-trueSol(x))**2, 0, 1))
print(errL2,err)
print(abs(aprxSol(0.5)-trueSol(0.5))**2)


plt.plot(xss,ys_aprx)
plt.plot(xss,ys_true)
plt.show()