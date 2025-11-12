import numpy as np
import scipy as sp
import lib.linear
import lib.quadratic

############################################
# For basis = hat functions
############################################


# construct the approximate function from coefficients in higher resolution
def constr_f(vec, x, h, func=lib.linear.basis):
    y = np.zeros_like(x)
    for i, val in enumerate(vec):
        y += val * func(x, i + 1, h)

    return y


############### f1 #################

# analytical function for f=1
def f1(x):
    return -0.5 * (x - 0.5) ** 2 + 0.125


# minus second derivative of u
def f1pp(x):
    return np.ones_like(x)


############### f2 #################


def f2(x):
    return 0.5 * (1 - np.abs(2 * (x - 0.5)))


def f2pp(x):
    y = np.zeros_like(x)
    y[np.abs(x - 0.5) < 1e-2] = 1e50
    return y


############### f3 #################


def f3(x):
    return (np.sin(np.pi * x)) / np.pi**2


def f3pp(x):
    return -np.sin(np.pi * x)


# dictionary for easy access to the corresponding analytical functions, -second derivative and b coefficients
funcs = {
    "f1_lin": (f1, f1pp, lib.linear.b_j_1, lib.linear.a_ij),
    "f2_lin": (f2, f2pp, lib.linear.b_j_2, lib.linear.a_ij),
    "f3_lin": (f3, f3pp, lib.linear.b_j_3, lib.linear.a_ij),
    "f1_quad": (f1, f1pp, lib.quadratic.b_j_1, lib.quadratic.a_ij),
    "f2_quad": (f2, f2pp, lib.quadratic.b_j_2, lib.quadratic.a_ij),
    "f3_quad": (f3, f3pp, lib.quadratic.b_j_3, lib.quadratic.a_ij),
}


#########################################################
# error calculation
#########################################################


# approx integral by sum manually
def l2_err(a, b, h):
    return np.sqrt(np.sum((a - b) ** 2) * h)


def l2_err_func(func, y_vals, x_vals, order=1):
    spline = sp.interpolate.make_interp_spline(x_vals, y_vals, k=order)
    # construct spline interpolation of the approximate solution. Use k=1 for linear (hat) basis, k=2 for quadratic elements. 
    # #This reconstructs 𝑢ℎ(𝑥) inside each element

    def f(x):
        # np.interp works here because hat functions are used.
        # this results in u_N being a linear interpolation between the u(x_k)
        return np.square(func(x) - spline(x))  # np.interp(x, x_vals, y_vals))

    err2, _ = sp.integrate.quad(f, x_vals[0], x_vals[-1]) # integrate the squared error over the domain
    return np.sqrt(err2)


# approx integral by sum manually
def l1_err(a, b, h):
    return np.sum(np.abs(a - b)) * h


def l1_err_func(func, y_vals, x_vals, order=1):
    spline = sp.interpolate.make_interp_spline(x_vals, y_vals, k=order) 

    def f(x):
        return np.abs(func(x) - spline(x))  # np.interp(x, x_vals, y_vals)

    err, _ = sp.integrate.quad(f, x_vals[0], x_vals[-1])
    return err


def linf_err(a, b, h):
    return np.max(np.abs(a - b))


def linf_err_func(func, y_vals, x_vals, order=1):
    spline = sp.interpolate.make_interp_spline(x_vals, y_vals, k=order)

    def f(x):
        return -np.abs(func(x) - spline(x))  # np.interp(x, x_vals, y_vals))

    # find maximum (or minimum of -error) numerically
    # (might be overkill as generally just sampling it and getting the max m
    # is probably quicker)
    res = sp.optimize.minimize_scalar(f, bounds=(x_vals[0], x_vals[-1]))
    return -res.fun
