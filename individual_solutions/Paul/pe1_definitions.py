import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8")


############################################
# For basis = hat functions
############################################


def hat(x, i: int, h):
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
        return func(x) * hat(x, j, h)

    y, abs_err = sp.integrate.quad(f, a, b)
    return y


# construct the approximate function from coefficients in higher resolution
def constr_f(vec, x, h, func=hat):
    y = np.zeros_like(x)
    for i, val in enumerate(vec):
        y += val * func(x, i + 1, h)

    return y


############### f1 #################


def f1(x):
    return -0.5 * (x - 0.5) ** 2 + 0.125


def b_j_1(j, h):
    return h


# minus second derivative of u
def f1pp(x):
    return np.ones_like(x)


############### f2 #################


def f2(x):
    return 1 - np.abs(2 * (x - 0.5))


def b_j_2(j, h):
    return 2 * hat(1 / 2, j, h)


def f2pp(x):
    y = np.zeros_like(x)
    y[np.abs(x - 0.5) < 1e-2] = 1e50
    return y


############### f3 #################


def f3(x):
    return (np.sin(np.pi * x)) / np.pi**2


def f3pp(x):
    return -np.sin(np.pi * x)


def b_j_3(j, h):

    return (
        -np.sin(h * np.pi * (j - 1))
        - np.sin(h * np.pi * (j + 1))
        + 2 * np.sin(h * np.pi * j)
    ) / (h * np.pi**2)


# dictionary for easy access to the corresponding analytical functions, -second derivative and b coefficients
funcs = {"f1": (f1, f1pp, b_j_1), "f2": (f2, f2pp, b_j_2), "f3": (f3, f3pp, b_j_3)}


#########################################################
# error calculation
#########################################################


# approx integral by sum manually
def l2_err(a, b, h):
    return np.sqrt(np.sum((a - b) ** 2) * h)


def l2_err_func(func, y_vals, x_vals):
    def f(x):
        # np.interp works here because hat functions are used.
        # this results in u_N being a linear interpolation between the u(x_k)
        return np.square(func(x) - np.interp(x, x_vals, y_vals))

    err2, _ = sp.integrate.quad(f, x_vals[0], x_vals[-1])
    return np.sqrt(err2)


# approx integral by sum manually
def l1_err(a, b, h):
    return np.sum(np.abs(a - b)) * h


def l1_err_func(func, y_vals, x_vals):
    def f(x):
        return func(x) - np.interp(x, x_vals, y_vals)

    err, _ = sp.integrate.quad(f, x_vals[0], x_vals[-1])
    return err


def linf_err(a, b, h):
    return np.max(np.abs(a - b))


def linf_err_func(func, y_vals, x_vals):
    def f(x):
        return -np.abs(func(x) - np.interp(x, x_vals, y_vals))

    # find maximum (or minimum of -error) numerically
    # (might be overkill as generally just sampling it and getting the max m
    # is probably quicker)
    res = sp.optimize.minimize_scalar(f, bounds=(x_vals[0], x_vals[-1]))
    return -res.fun
