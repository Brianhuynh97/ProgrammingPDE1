import numpy as np
import scipy as sp #type: ignore


def basis(x, i: int, h):
    x_i = i * h / 2 # center of the basis function
    if i % 2 == 0: # even index -> linear basis function
        y = np.maximum(-np.abs(x - x_i) / h + 1, 0) # function is 0 outside the support [x_i - h, x_i + h]
        # within the support, it linearly increases from 0 to 1 at x_i and then decreases back to 0
    else:
        y = np.maximum(-4 / h**2 * (x - x_i) ** 2 + 1, 0) # peak at x_i, 0 at x_i - h/2 and x_i + h/2
    return y


def a_ij(i, j, h):
    i_even = i % 2 == 0 # check if i is even
    j_even = j % 2 == 0 # check if j is even
    diff = i - j

    if i == j:
        if i_even:
            return 2 / h # diagonal entries for linear basis functions
        else:
            return 16 / 3 / h # diagonal entries for quadratic basis functions
    elif np.abs(diff) == 2 and i_even: # both i and j are even and neighbors
        return -1 / h
    # if ((diff == -1) and (i_even)) or ((diff == 1) and (j_even)):
    #     return 2 / h
    # if ((diff == 1) and (i_even)) or ((diff == -1) and (j_even)):
    #     return -2 / h
    return 0


# numerically get b_j for any function, will introduce instability
def b_j(j, h, func, a, b):
    def f(x):
        return func(x) * basis(x, j, h)

    y, abs_err = sp.integrate.quad(f, a, b)
    return y


def b_j_1(j, h):
    if j % 2 == 0: # even index -> linear basis function
        return h 
        return h
    else:
        return 2 * h / 3 # odd index -> quadratic basis function


def b_j_2(j, h):
    return 2 * basis(1 / 2, j, h)


def b_j_3(j, h):

    if j % 2 == 0:
        return (
            -np.sin(h * np.pi * (j / 2 - 1))
            - np.sin(h * np.pi * (j / 2 + 1))
            + 2 * np.sin(h * np.pi * j / 2)
        ) / (h * np.pi**2)
    else:

        return (
            -4
            / (h**2 * np.pi**3) # coefficient from integrating quadratic basis function
            * (
                2 * (np.cos(np.pi * h * (j + 1) / 2) - np.cos(np.pi * h * (j - 1) / 2)) # center contribution
                + h
                * np.pi
                * (np.sin(np.pi * h * (j + 1) / 2) + np.sin(np.pi * h * (j - 1) / 2))
            )
        )
