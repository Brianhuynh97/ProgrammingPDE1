import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, csc_matrix
from scipy.sparse.linalg import spsolve

# 1. Grid Generation
def grid(N, ansatz='linear', nonuniform=False):
    """Generate uniform or non-uniform 1D grid."""
    node_count = N + 1 if ansatz == 'linear' else 3*N - (N - 1)
    ## each element has 2 nodes for linear ansatz. so the total nodes are N+1. Else is for quadratic ansatz (each element has 3 nodes, but the middle nodes are shared))
    vec = np.linspace(0, 1, node_count)
    if nonuniform:
        vec = vec**2  # non-uniform mapping
    return vec


# 2. FEM Assembly (Analytical Form)
def assembleMatrix(lattice):
    """Assemble sparse stiffness matrix using analytical element matrices."""
    n = len(lattice)
    ## lattice is the grid points. n is the number of nodes. 
    main = np.zeros(n)
    ## values for the main diagonal of the stiffness matrix
    upper = np.zeros(n - 1)
    ## values for the main diagonal and upper diagonal of the stiffness matrix
    for i in range(n - 1):
        h = lattice[i + 1] - lattice[i]
        ##computes the element length 
        main[i] += 1 / h
        ## each element contributes 1/h to both nodes
        main[i + 1] += 1 / h
        upper[i] -= 1 / h
    A = diags([main, upper, upper], [0, -1, 1], format='csc')
    return A

## Right-hand side assembly for f = 1
def rhsConstant(lattice):
    """Assemble right-hand side vector for f = 1."""
    n = len(lattice)
    b = np.zeros(n)
    ## initialize the right-hand side vector
    for i in range(n - 1):
        h = lattice[i + 1] - lattice[i]
        ##compute the element length between nodes i and i+1
        b[i] += h / 2
        ## each element contributes h/2 to both nodes
        b[i + 1] += h / 2
        ## so we add h/2 to both b[i] and b[i+1]
    return b

## Task 2 with delta distribution (ignored it for task 1 ^^)
def rhsDelta(lattice):
    """Assemble right-hand side for a delta function at x = 0.5."""
    n = len(lattice)
    b = np.zeros(n)
    for i in range(n - 1):
        if lattice[i] <= 0.5 <= lattice[i + 1]:
            h = lattice[i + 1] - lattice[i]
            # Distribute delta contribution proportionally to linear shape functions
            b[i] += 2 * (lattice[i + 1] - 0.5) / h
            b[i + 1] += 2 * (0.5 - lattice[i]) / h
    return b

# 3. Solve Systems
def FEM1DConstant(N):
    """Solve -u'' = 1 with Dirichlet BC u(0)=u(1)=0."""
    G = grid(N)
    A = assembleMatrix(G)
    ## Stiffness matrix
    b = rhsConstant(G)

    # Solve for interior nodes
    u = np.zeros(len(G))
    u_interior = spsolve(A[1:-1, 1:-1], b[1:-1])
    ## Solve the linear system for interior nodes only (excluding boundary nodes)
    u[1:-1] = u_interior
    ## Assign the computed interior values back to the full solution vector

    # Analytical solution
    u_analytical = -0.5 * (G - 0.5) ** 2 + 1 / 8
    return G, u, u_analytical


def FEM1DDelta(N):
    """Solve -u'' = delta(x-0.5) with Dirichlet BC u(0)=u(1)=0."""
    G = grid(N)
    A = assembleMatrix(G)
    b = rhsDelta(G)

    # Solve for interior nodes
    u = np.zeros(len(G))
    u_interior = spsolve(A[1:-1, 1:-1], b[1:-1])
    ##This slices the matrix to remove the first and last rows and columns. 
    u[1:-1] = u_interior

    # Analytical solution
    u_analytical = np.where(G <= 0.5, G, 1 - G)
    return G, u, u_analytical


# 4. Interpolation and Plotting
def interpolate(x, lattice, u):
    """ Interpolation on piecewise-linear grid."""
    if np.isscalar(x):
        x = np.array([x])
    x = np.clip(x, 0, 1)
    ## ensure x is within the domain [0, 1]

    u_interp = np.zeros_like(x)
    for k, xi in enumerate(x):
        i = np.searchsorted(lattice, xi) - 1
        ## find the element index containing xi
        i = np.clip(i, 0, len(lattice) - 2)
        ## ensure i is within valid range
        h = lattice[i + 1] - lattice[i]
        ## element length
        phi_left = (lattice[i + 1] - xi) / h
        ## linear shape function for left node
        phi_right = (xi - lattice[i]) / h
        ## linear shape function for right node
        u_interp[k] = phi_left * u[i] + phi_right * u[i + 1]
        ## interpolate using shape functions
    return u_interp


def plotResult(x, u, X, U, title='FEM 1D', color='r'):
    """Compare FEM and analytical results."""
    u_int = interpolate(X, x, u)
    plt.figure(figsize=(8, 5))
    plt.plot(X, u_int, f'--{color}', label='FEM (interpolated)')
    plt.plot(x, u, f'o{color}', label='FEM nodes')
    plt.plot(X, U, '-b', label='Analytical')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# 5. Example Tests
if __name__ == "__main__":
    # Constant RHS
    x, u, ua = FEM1DConstant(11)
    X = np.linspace(0, 1, 1000)
    U = -0.5 * (X - 0.5) ** 2 + 1 / 8
    plotResult(x, u, X, U, title='FEM 1D for f = 1', color='r')

    # Delta RHS (for task 2)
    #x, u, ua = FEM1DDelta(40)
    #X = np.linspace(0, 1, 1000)
    #U = np.where(X <= 0.5, X, 1 - X)
    #plotResult(x, u, X, U, title='FEM 1D for delta(x-0.5)', color='g')
