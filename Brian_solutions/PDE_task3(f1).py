import numpy as np
import matplotlib.pyplot as plt

def u_f(x):
    """Analytical solution for f = 1"""
    return -0.5*(x - 0.5)**2 + 1/8

def u_a_ij(i, j, h):
    """Stiffness matrix entry for linear hat functions"""
    if i == j:
        return 2 / h
    elif abs(i - j) == 1:
        return -1 / h
    else:
        return 0

def u_b(i, h):
    """RHS entry for linear hat functions with f=1"""
    return h

def solve_fem(N):
    h = 1 / N
    a = np.array([[u_a_ij(i,j,h) for j in range(1,N)] for i in range(1,N)])
    b = np.array([u_b(i,h) for i in range(1,N)])
    x = np.linalg.solve(a,b)
    # pad with boundary zeros
    u_num = np.pad(x, 1, 'constant', constant_values=0)
    x_grid = np.linspace(0,1,N+1)
    return x_grid, u_num

def interp_fine(x_grid, u_num, fine_x):
    return np.interp(fine_x, x_grid, u_num)

def l1_err(u_exact, u_num):
    return np.trapezoid(np.abs(u_exact - u_num), dx=u_exact[1]-u_exact[0])

def l2_err(u_exact, u_num):
    return np.sqrt(np.trapezoid((u_exact - u_num)**2, dx=u_exact[1]-u_exact[0]))

def linf_err(u_exact, u_num):
    return np.max(np.abs(u_exact - u_num))

if __name__ == "__main__":
    # Single FEM plot for N=10
    N_plot = 10
    x_grid, u_num = solve_fem(N_plot)
    fine_x = np.linspace(0,1,1000)
    u_exact = u_f(fine_x)
    u_num_fine = interp_fine(x_grid, u_num, fine_x)

    plt.figure(figsize=(8,5))
    plt.plot(fine_x, u_num_fine, '--r', label='FEM (interpolated)')
    plt.plot(x_grid, u_num, 'ro', label='FEM nodes')
    plt.plot(fine_x, u_exact, '-b', label='Analytical')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title(f'FEM vs Analytical Solution, N={N_plot}')
    plt.legend()
    plt.grid(True)
    plt.show()

    cumulative_error = np.sqrt(np.cumsum((u_exact - u_num_fine)**2) * (fine_x[1]-fine_x[0]))
    plt.figure(figsize=(8,4))
    plt.plot(fine_x, cumulative_error, 'r', label='Cumulative error')
    plt.xlabel('x')
    plt.ylabel('Cumulative L2 error')
    plt.title('Cumulative Error over Domain')
    plt.grid(True)
    plt.legend()
    plt.show()

    Ns = [4,8,16,32,64,128]
    l1_errors, l2_errors, linf_errors = [],[],[]

    for N in Ns:
        x_grid, u_num = solve_fem(N)
        u_num_fine = interp_fine(x_grid, u_num, fine_x)
        u_exact_fine = u_f(fine_x)

        l1_errors.append(l1_err(u_exact_fine, u_num_fine))
        l2_errors.append(l2_err(u_exact_fine, u_num_fine))
        linf_errors.append(linf_err(u_exact_fine, u_num_fine))

    print("N      L1 Error      L2 Error      Linf Error")
    for i,N in enumerate(Ns):
        if i==0:
            rate_l1 = rate_l2 = rate_linf = '-'
        else:
            rate_l1 = np.log(l1_errors[i-1]/l1_errors[i])/np.log(2)
            rate_l2 = np.log(l2_errors[i-1]/l2_errors[i])/np.log(2)
            rate_linf = np.log(linf_errors[i-1]/linf_errors[i])/np.log(2)
        print(f"{N:3d}   {l1_errors[i]:.3e}   {l2_errors[i]:.3e}   {linf_errors[i]:.3e}   "
              f"{rate_l1} {rate_l2} {rate_linf}")

    plt.figure(figsize=(7,5))
    plt.loglog(Ns, l1_errors,'o-', label='L1 Error')
    plt.loglog(Ns, l2_errors,'s-', label='L2 Error')
    plt.loglog(Ns, linf_errors,'^-', label='Linf Error')
    plt.xlabel('Number of elements N')
    plt.ylabel('Error')
    plt.grid(True, which='both', ls='--')
    plt.title('FEM Convergence for f=1')
    plt.legend()
    plt.show()

