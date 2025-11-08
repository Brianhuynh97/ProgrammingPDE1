import numpy as np
import scipy as sp


def basis(x, i: int, h):
    x_i = i * h / 2
    if i % 2 == 0:
        y = np.maximum(-np.abs(x - x_i) / h + 1, 0)
    else:
        y = np.maximum(-4 / h**2 * (x - x_i) ** 2 + 1, 0)
    return y


def a_ij(i, j, h):
    i_even = i % 2 == 0
    j_even = j % 2 == 0
    diff = i - j
    # if i == j:
    #     if i_even:
    #         return 2 / h
    #     else:
    #         return 4 * h**3 / 5
    # if i_even:
    #     if np.abs(diff) == 2:
    #         return -1 / h
    # if ((diff == -1) and (i_even)) or ((diff == 1) and (j_even)):
    #     return 2 * h**2
    # if ((diff == 1) and (i_even)) or ((diff == -1) and (i_even)):
    #     return -2 * h**2
    # return 0
    if i == j:
        if i_even:
            return 2 / h
        else:
            return 16 / 3 / h
    if i_even:
        if np.abs(diff) == 2:
            return -1 / h
    if ((diff == -1) and (i_even)) or ((diff == 1) and (j_even)):
        return 2 * h
    if ((diff == 1) and (i_even)) or ((diff == -1) and (j_even)):
        return -2 * h
    return 0


# numerically get b_j for any function, will introduce instability
def b_j(j, h, func, a, b):
    def f(x):
        return func(x) * basis(x, j, h)

    y, abs_err = sp.integrate.quad(f, a, b)
    return y


def b_j_1(j, h):
    if j % 2 == 0:
        return h
    else:
        return 2 * h / 3


def b_j_2(j, h):
    return


def b_j_3(j, h):

    return
