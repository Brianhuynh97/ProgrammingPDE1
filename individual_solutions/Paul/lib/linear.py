import numpy as np
import scipy as sp


def basis(x, i: int, h):
    x_i = h * i

    y = np.maximum(-np.abs(x - x_i) / h + 1, 0)
    return y


def a_ij(i, j, h):
    if i == j:
        return 2 / h
    elif np.abs(i - j) == 1:
        return -1 / h
    else:
        return 0


# numerically get b_j for any function, will introduce instability
def b_j(j, h, func, a, b):
    def f(x):
        return func(x) * basis(x, j, h)

    y, abs_err = sp.integrate.quad(f, a, b)
    return y


def b_j_1(j, h):
    return h


def b_j_2(j, h):
    return 2 * basis(1 / 2, j, h)


def b_j_3(j, h):

    return (
        -np.sin(h * np.pi * (j - 1))
        - np.sin(h * np.pi * (j + 1))
        + 2 * np.sin(h * np.pi * j)
    ) / (h * np.pi**2)
