import numpy as np
import scipy as sp

# Define standard basic hat function
def basis(x, i: int, h):
    x_i = h * i # center of the basis function

    y = np.maximum(-np.abs(x - x_i) / h + 1, 0) #function is 0 outside the support [x_i - h, x_i + h] 
    # within the support, it linearly increases from 0 to 1 at x_i and then decreases back to 0
    return y

# Implement the stiffness matrix entries
def a_ij(i, j, h):
    if i == j: # diagonal entry
        return 2 / h # diagonal entries
    elif np.abs(i - j) == 1:
        return -1 / h # off-diagonal entries
    else:
        return 0


# numerically get b_j for any function, will introduce instability
def b_j(j, h, func, a, b):
    def f(x):
        return func(x) * basis(x, j, h) # integrand for b_j 

    y, abs_err = sp.integrate.quad(f, a, b) # numerical integration over [a, b]
    return y


def b_j_1(j, h): 
    return h


def b_j_2(j, h):
    return 2 * basis(1 / 2, j, h) # evaluate basis function at x = 0.5


def b_j_3(j, h):

    return (
        -np.sin(h * np.pi * (j - 1)) # left neighbor contribution
        - np.sin(h * np.pi * (j + 1)) # right neighbor contribution
        + 2 * np.sin(h * np.pi * j) # center contribution
    ) / (h * np.pi**2) # divide by pi^2 to get the integral over [0,1]


def b_j_4(j, h):
    return 0
