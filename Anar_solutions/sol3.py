import time
import numpy as np
import scipy.sparse as sprs
import matplotlib.pyplot as plt
from scipy.integrate import quad

class Function:
    def __init__(self, continuous=None, singular=None):
        self.continuous = continuous
        self.singular = singular or []

        if self.continuous and self.singular:
            self.type = "mixed"
        elif self.continuous:
            self.type = "continuous"
        elif self.singular:
            self.type = "singular"
        else:
            self.type = "empty"

class PoissonSolver1D:
    def __init__(self, f = lambda x: 1, N = 9, a = 0, b = 1):
        self.f, self.a, self.b, self.N = f, a, b, N
        self.xs = np.linspace(a,b,N+2)
        self.h = (b-a)/(N+1)

    def update_grid(self):
        a, b, N = self.a, self.b, self.N
        self.xs = np.linspace(a,b,N+2)
        self.h = (b-a)/(N+1)

    def phi(self,i,x):
        xs, h = self.xs, self.h
        if xs[i] <= x <= xs[i+1]:
            return (x-xs[i])/h
        elif xs[i+1] <  x <= xs[i+2]:
            return (xs[i+2]-x)/h
        else:
            return 0

    def gen_stfMat(self):
        N, h = self.N, self.h
        stfMat_lil = sprs.lil_matrix((N,N),dtype=float)
        for i in range(N-1):
            stfMat_lil[i,i] = 2
            stfMat_lil[i+1,i] = -1
            stfMat_lil[i,i+1] = -1
        stfMat_lil[N-1,N-1] = 2
        stfMat_lil /= h
        self.stfMat = stfMat_lil.tocsc()

    def gen_rhs(self):
        f, N, phi, xs = self.f, self.N, self.phi, self.xs
        rhs = np.zeros(N,dtype=float)

        if f.type in ("continuous", "mixed"):
            f_cont = f.continuous
            for i in range(N):
                rhs[i], _ = quad(lambda x: f_cont(x)*phi(i,x), xs[i], xs[i+2])

        if f.type in ("singular", "mixed"):
            for f_sing in f.singular:
                x0, w = f_sing["point"], f_sing["weight"]
                i = np.searchsorted(xs, x0) - 1
                if i == 0:
                    rhs[i] += w*phi(i,x0)
                elif i == N:
                    rhs[i-1] += w*phi(i-1,x0)
                else:
                    rhs[i] += w*phi(i,x0)
                    rhs[i-1] += w*phi(i-1,x0)

        self.rhs = rhs

    def assemble(self):
        self.gen_stfMat()
        self.gen_rhs()

    def solve(self):
        stfMat, rhs = self.stfMat, self.rhs
        N, xs, phi = self.N, self.xs, self.phi
        coeffs = self.coeffs = sprs.linalg.spsolve(stfMat,rhs)
        def aprxSol(x):
            i = np.searchsorted(xs,x) - 1
            if i == 0:
                return coeffs[i]*phi(i,x)
            elif i == N:
                return coeffs[i-1]*phi(i-1,x)
            else:
                return coeffs[i-1]*phi(i-1,x) + coeffs[i]*phi(i,x)
        self.solution = aprxSol

    def cal_errL2(self,trueSol):
        aprxSol, a, b = self.solution, self.a, self.b
        val, _ = quad(lambda x: abs(aprxSol(x) - trueSol(x))**2, a, b)
        self.errL2 = np.sqrt(val)
        return self.errL2

    def cal_errL1(self, trueSol):
        aprxSol, a, b = self.solution, self.a, self.b
        self.errL1, _ = quad(lambda x: abs(aprxSol(x) - trueSol(x)), a, b)
        return self.errL1

    def cal_errLinf(self, trueSol, n = 1000):
        aprxSol, a, b = self.solution, self.a, self.b
        xs = np.linspace(a,b,n)
        self.errLinf = np.max(np.abs([aprxSol(x) - trueSol(x) for x in xs]))
        return self.errLinf

    def cal_errs(self, trueSol, n=1000):
        return [self.cal_errL2(trueSol), self.cal_errL1(trueSol), 
                self.cal_errLinf(trueSol, n)]

def isEven(num):
    return num%2==0

### FIRST FUNCTION ###
## FIRST FUNCTION ##
f_cont = lambda x: 1
f_sing = None

## FIRST TRUE SOLUTION ##
def trueSol(x):
    return -(x-1/2)*(x-1/2)/2 + 1/8

### SECOND FUNCTION ###
## SECOND FUNCTION ##
# f_cont = lambda x: np.sin(np.pi*x)
# f_sing = None

## SECOND TRUE SOLUTION ##
# def trueSol(x):
#     return np.sin(np.pi*x)/(np.pi*np.pi)

### THIRD FUNCTION ###
## THIRD FUNCTION ##
# f_cont = None
# f_sing = [{"point": 1/2, "weight": 2}]

## THIRD TRUE SOLUTION ##
#def trueSol(x):
#    if x <= 1/2:
#        return x
#    else:
#        return 1-x


f = Function(f_cont, f_sing)

### Solve just a problem ###
# start = time.perf_counter()
# solver = PoissonSolver1D(f, N = 1000)
# solver.assemble()
# solver.solve()
# aprxSol = solver.solution
# end = time.perf_counter()
# print(f"Solve time:\t{end - start:.6f} seconds")
# 
# errs = solver.cal_errs(trueSol)
# print(f"L2 error:\t{errs[0]}\nL1 error:\t{errs[1]}\nLinf error:\t{errs[2]}")
# 
# xss = np.linspace(0,1,1000)
# ys_aprx = np.array([aprxSol(x) for x in xss])
# ys_true = np.array([trueSol(x) for x in xss])
# 
# plt.plot(xss,ys_aprx)
# plt.plot(xss,ys_true)
# plt.show()

### Plot errors ###
errL2s = np.zeros(100)
errL1s = np.zeros(100)
errLinfs = np.zeros(100)
nss_rand = np.logspace(1,5,50,dtype=int)
nss_even = 2*nss_rand
nss_odd = nss_even-1
nss = np.concatenate((nss_even,nss_odd))
nss = np.sort(nss)

for i in range(100):
    solver = PoissonSolver1D(f, N=nss[i])
    solver.assemble()
    solver.solve()
    errs = solver.cal_errs(trueSol)
    errL2s[i],errL1s[i],errLinfs[i] = errs[0],errs[1],errs[2]
    print(f"iter {i}")
    print(f"L2 error:\t{errs[0]}\nL1 error:\t{errs[1]}\nLinf error:\t{errs[2]}")
    print("\n\n")

plt.loglog(nss,errL2s)
plt.loglog(nss,errL1s)
plt.loglog(nss,errLinfs)
plt.legend(["L2","L1","Linf"])
plt.show()
