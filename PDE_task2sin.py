import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

def grid(N, ansatz='linear', nonuniform=False):
    """Generate uniform or non-uniform 1D grid."""
    node_count = N + 1 if ansatz == 'linear' else 3*N - (N - 1)
    ## each element has 2 nodes for linear ansatz. so the total nodes are N+1. Else is for quadratic ansatz (each element has 3 nodes, but the middle nodes are shared))
    vec = np.linspace(0, 1, node_count)
    if nonuniform:
        vec = vec**2  # non-uniform mapping
    return vec

def assembleMatrix(lattice):
    n = len(lattice)
    main = np.zeros(n)
    upper = np.zeros(n-1)
    for i in range(n-1):
        h = lattice[i+1] - lattice[i]
        main[i] += 1/h
        main[i+1] += 1/h
        upper[i] -= 1/h
    return diags([main, upper, upper], [0, -1, 1], format='csc')

def rhsSinPi(lattice):
    """RHS vector for f(x) = sin(pi x)"""
    n = len(lattice)
    b = np.zeros(n)
    for i in range(n-1):
        h = lattice[i+1] - lattice[i]
        x_mid = 0.5*(lattice[i]+lattice[i+1])
        f_mid = np.sin(np.pi * x_mid)
        b[i] += f_mid * h/2
        b[i+1] += f_mid * h/2
    return b

def FEM1DSinPi(N):
    G = grid(N)
    A = assembleMatrix(G)
    b = rhsSinPi(G)
    u = np.zeros(len(G))
    u_interior = spsolve(A[1:-1,1:-1], b[1:-1])
    u[1:-1] = u_interior
    u_analytical = np.sin(np.pi * G)/np.pi**2
    return G, u, u_analytical

def interpolate(x, lattice, u):
    u_interp = np.zeros_like(x)
    for k, xi in enumerate(x):
        i = np.searchsorted(lattice, xi) - 1
        i = np.clip(i, 0, len(lattice)-2)
        h = lattice[i+1]-lattice[i]
        phi_left = (lattice[i+1]-xi)/h
        phi_right = (xi-lattice[i])/h
        u_interp[k] = phi_left*u[i] + phi_right*u[i+1]
    return u_interp

def plotResult(x, u, X, U):
    u_int = interpolate(X, x, u)
    plt.figure(figsize=(8,5))
    plt.plot(X, u_int, '--g', label='FEM (interpolated)')
    plt.plot(x, u, 'og', label='FEM nodes')
    plt.plot(X, U, '-b', label='Analytical')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title('FEM 1D for f = sin(pi x)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    x, u, u_analytical = FEM1DSinPi(10) 
    X = np.linspace(0, 1, 1000)
    U = np.sin(np.pi * X)/np.pi**2
    plotResult(x, u, X, U)
