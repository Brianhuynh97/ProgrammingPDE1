import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

## Grid generation
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
    ## main diagonal
    upper = np.zeros(n-1)
    ## upper diagonal
    for i in range(n-1):
        h = lattice[i+1] - lattice[i]
        main[i] += 1/h
        main[i+1] += 1/h
        upper[i] -= 1/h

    return diags([main, upper, upper], [0, -1, 1], format='csc') ## Right-hand side for delta function

def rhsDelta(lattice):
    """RHS vector for delta(x-0.5)."""
    n = len(lattice)
    b = np.zeros(n)
    for i in range(n-1):
        if lattice[i] <= 0.5 <= lattice[i+1]:
            ##Checks if the delta source at x = 0.5 lies inside this element.
            ##Only the element containing x=0.5 gets a non-zero contribution.
            h = lattice[i+1] - lattice[i]
            ##h is the length of the element containing the delta.
            b[i] += 2 * (lattice[i+1] - 0.5)/h
            ##RHS value for the left node of the element.
            b[i+1] += 2 * (0.5 - lattice[i])/h
            ##RHS value for the right node of the element.
    return b

## FEM iplementation for delta function
def FEM1DDelta(N):
    G = grid(N)
    A = assembleMatrix(G)
    b = rhsDelta(G)
    u = np.zeros(len(G))
    u_interior = spsolve(A[1:-1,1:-1], b[1:-1]) 
    u[1:-1] = u_interior
    u_analytical = np.where(G <= 0.5, 2*G, 2*(1-G))
    #This line defines the analytical solution for the PDE:
    #-u''(x) = 2δ(x - 0.5), u(0) = u(1) = 0
    # G is the vector of FEM nodes (grid points)
    # u_analytical will store the exact solution at each node
    # For x <= 0.5, the solution is u(x) = 2x
    # For x > 0.5, the solution is u(x) = 2(1 - x) 
    # This creates a piecewise linear function peaking at x=0.5
    return G, u, u_analytical

## Interpolation and Plotting
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
    plt.title('FEM 1D for 2*delta(x-0.5)')
    plt.legend()
    plt.grid(True)
    plt.show()

## Implementation test 
if __name__ == "__main__":
    x, u, u_analytical = FEM1DDelta(10) 
    X = np.linspace(0, 1, 1000)
    U = np.where(X <= 0.5, 2*X, 2*(1-X))       
    plotResult(x, u, X, U)


