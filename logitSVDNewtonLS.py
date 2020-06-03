# ------------------------------------------------------------------------------
# Newton's method with a line search algorithm for the step size
#
# The step size is calculated to satisfy the strong Wolfe conditions and follows
# Wright and Nocedal, 'Numerical Optimization', 1999, pg. 59-60
#
# Author: Christian Wagner, May 2020
# ------------------------------------------------------------------------------

"""
Function call
    x = NewtonLS(x0,f_obj, f_grad, f_hesse, maxit = 100, tol=1e-10, print = "warn", indent = "", argv=()):

Parameters
    x0     : ndarray, initial solution
    f_obj  : callable f(x,*args), objective function
    f_grad : callable f'(x,*args), gradient of the objective function
    f_hesse: callable f'(x,*args), Hessian of the objective function
    argv   : arguments additional to x0 for f_obj, f_grad, f_hesse
    maxit  : maximum number of iterations
    tol    : stopping error tolerance
    verbose: ("none" | "warn" | "all"), print warnings and convergence progress. Default is "warn"
    indent : indent for printing convergence results

Returns
    x_k    : ndarray, optimal solution after k iterations
    f_k    : value of the objective function for the solution after k iteration
    k      : number of iterations

"""

import numpy as np

def NewtonLS_Wolfe_cubicmin(a, fa, fpa, b, fb, c, fc):
    """
    Finds the minimizer for a cubic polynomial that goes through the
    points (a,fa), (b,fb), and (c,fc) with derivative at a of fpa.

    If no minimizer can be found return None

    """
    # f(x) = A *(x-a)^3 + B*(x-a)^2 + C*(x-a) + D

    with np.errstate(divide='raise', over='raise', invalid='raise'):
        try:
            C = fpa
            db = b - a
            dc = c - a
            denom = (db * dc) ** 2 * (db - dc)
            d1 = np.empty((2, 2))
            d1[0, 0] = dc ** 2
            d1[0, 1] = -db ** 2
            d1[1, 0] = -dc ** 3
            d1[1, 1] = db ** 3
            [A, B] = np.dot(d1, np.asarray([fb - fa - C * db,
                                            fc - fa - C * dc]).flatten())
            A /= denom
            B /= denom
            radical = B * B - 3 * A * C
            xmin = a + (-B + np.sqrt(radical)) / (3 * A)
        except ArithmeticError:
            return None
    if not np.isfinite(xmin):
        return None
    return xmin


def NewtonLS_Wolfe_quadmin(a, fa, fpa, b, fb):
    """
    Finds the minimizer for a quadratic polynomial that goes through
    the points (a,fa), (b,fb) with derivative at a of fpa,
    """
    # f(x) = B*(x-a)^2 + C*(x-a) + D
    with np.errstate(divide='raise', over='raise', invalid='raise'):
        try:
            D = fa
            C = fpa
            db = b - a * 1.0
            B = (fb - D - C * db) / (db * db)
            xmin = a - C / (2.0 * B)
        except ArithmeticError:
            return None
    if not np.isfinite(xmin):
        return None
    return xmin


def NewtonLS_Wolfe_zoom(a_lo, a_hi, phi_lo, phi_hi, dphi_lo,
          phi, dphi, phi0, dphi0, c1, c2):
    """
    "zoom" procedure from  Wright and Nocedal, 'Numerical Optimization', 1999, pg. 59-60
    """

    maxiter = 10
    i = 0
    delta1 = 0.1  # cubic interpolant check
    delta2 = 0.1  # quadratic interpolant check
    phi_rec = phi0
    a_rec = 0

    a_star = None
    val_star = None
    grad_star = None

    for i in range(0,maxiter):
        # interpolate to find a trial step length between a_lo and  a_hi
        dalpha = a_hi - a_lo
        if dalpha < 0:
            a, b = a_hi, a_lo
        else:
            a, b = a_lo, a_hi

        if np.isnan(phi_lo) | np.isnan(phi_hi): # special case where phi(alpha) cannot be calculated
            a_j = a_lo + 0.5 * dalpha
        else:
            if (i > 0):
                cchk = abs(delta1 * dalpha)
                a_j = NewtonLS_Wolfe_cubicmin(a_lo, phi_lo, dphi_lo, a_hi, phi_hi,
                                a_rec, phi_rec)
            if (i == 0) or (a_j is None) or (a_j > b - cchk) or (a_j < a + cchk):
                qchk = abs(delta2 * dalpha)
                a_j = NewtonLS_Wolfe_quadmin(a_lo, phi_lo, dphi_lo, a_hi, phi_hi)
                if (a_j is None) or (a_j > b-qchk) or (a_j < a+qchk):
                    a_j = a_lo + 0.5*dalpha

        # Check new value of a_j
        phi_aj = phi(a_j)
        # only very special cases
        if np.isnan(phi_aj):
            if np.isnan(phi_hi):
                phi_rec = phi_hi
                a_rec = a_hi
                a_hi = a_j
                phi_hi = phi_aj
                continue;
            else:
                phi_rec = phi_lo
                a_rec = a_lo
                a_lo = a_j
                phi_lo = phi_aj
                continue;
        if (phi_aj > phi0 + c1*a_j*dphi0) or (phi_aj >= phi_lo):
            phi_rec = phi_hi
            a_rec = a_hi
            a_hi = a_j
            phi_hi = phi_aj
        else:
            dphi_aj = dphi(a_j)
            if abs(dphi_aj) <= -c2*dphi0:
                a_star = a_j
                val_star = phi_aj
                grad_star = dphi_aj
                break
            if dphi_aj*(a_hi - a_lo) >= 0:
                phi_rec = phi_hi
                a_rec = a_hi
                a_hi = a_lo
                phi_hi = phi_lo
            else:
                phi_rec = phi_lo
                a_rec = a_lo

            a_lo = a_j
            phi_lo = phi_aj
            dphi_lo = dphi_aj

    return a_star, val_star, grad_star


def NewtonLS_Wolfe_scalar(phi, dphi=None, phi0=None,  dphi0=None,
                         c1=1e-4, c2=0.9, amax=None, maxiter=10, verbose="warn"):
    """
    step size selection from  Wright and Nocedal, 'Numerical Optimization', 1999, pg. 59-60
    """

    if phi0 is None:
        phi0 = phi(0.0)

    if dphi0 is None and dphi is not None:
        dphi0 = dphi(0.0)

    # maximum number of iterations reached
    alpha_star = None
    phi_star = None
    dphi_star = None

    # notation: alpha_{i-1} is alpha_i1
    alpha_i1 = 0
    alpha_i = 1.0

    phi_ai = phi(alpha_i)
    phi_ai1 = phi0
    dphi_ai1 = dphi0

    # notation: alpha_{i-1} is
    for i in range(maxiter):

        if (phi_ai > phi0 + c1 * alpha_i * dphi0) or \
           ((phi_ai >= phi_ai1) and (i > 1)) or \
                np.isnan(phi_ai) or np.isnan(phi_ai1):
            alpha_star, phi_star, dphi_star = \
                        NewtonLS_Wolfe_zoom(alpha_i1, alpha_i, phi_ai1,
                              phi_ai, dphi_ai1, phi, dphi,
                              phi0, dphi0, c1, c2)
            break

        dphi_ai = dphi(alpha_i)
        if (abs(dphi_ai) <= -c2*dphi0):
            alpha_star = alpha_i
            phi_star = phi_ai
            dphi_star = dphi_ai
            break

        if (dphi_ai >= 0):
            alpha_star, phi_star, dphi_star = \
                        NewtonLS_Wolfe_zoom(alpha_i, alpha_i1, phi_ai,
                              phi_ai1, dphi_ai, phi, dphi,
                              phi0, dphi0, c1, c2)
            break

        # prepare nex iteration step
        alpha_i1 = alpha_i
        phi_ai1 = phi_ai
        dphi_ai1 = dphi_ai

        # choose new alpha
        alpha_i = 2 * alpha_i
        if amax is not None:
            alpha_i = min(alpha_i, amax)
            i = maxiter-2    #todo: test if stop works
        phi_ai = phi(alpha_i)

    if alpha_star == None:
        if verbose in ("warn","all"): print('Warning: Optimal alpha not found.')

    return alpha_star, phi_star, dphi_star


def NewtonLS_Wolfe(f, fprime, x_k, p_k, grad_k, f_k,
                   args=(), c1=1e-4, c2=0.9, amax=None, maxiter=10, verbose = "warn"):

    grad_alpha = [None]

    def phi(alpha):
        return f(x_k + alpha * p_k, *args)

    def dphi(alpha):
        grad_alpha[0] = fprime(x_k + alpha * p_k, *args)
        return np.dot(grad_alpha[0], p_k)

    dphi0_k = np.dot(grad_k, p_k)

    if dphi0_k > 0:
        if verbose in ("warn","all"): print('Warning: The Hessian is not positive definite.')
        return None, f_k, grad_k

    alpha_star, f_star, dphi_star = NewtonLS_Wolfe_scalar(
            phi, dphi, f_k, dphi0_k, c1, c2, amax, maxiter=maxiter, verbose=verbose)

    if (alpha_star is None) | (f_star is None) | (dphi_star is None) :
        if verbose in ("warn","all"): print('Warning: The line search algorithm did not converge.')
        return None, f_k, grad_k
    else:
        # dphi_star is a number (dphi) -- so use the most recently
        # calculated gradient used in computing it dphi = gfk*pk
        grad_star = grad_alpha[0]

    return alpha_star, f_star, grad_star


def NewtonLS(x0,f_obj, f_grad, f_hesse, maxit = 100, tol=1e-10, verbose = "warn", indent = "", argv=()):

    x_k = x0.copy()
    f_k = f_obj(x_k, *argv)
    grad_k = f_grad(x_k, *argv)
    gnorm = np.linalg.norm(grad_k)
    if verbose == "all": print(indent, 0, f_k, gnorm)
    gnormold = gnorm

    for k in range(0,maxit):
        hess = f_hesse(x_k, *argv)
        p_k = -np.linalg.solve(hess,grad_k)
        alpha, f_k, grad_k = NewtonLS_Wolfe(f_obj, f_grad, x_k, p_k, grad_k=grad_k, f_k=f_k,
                       args=argv, c1=1e-4, c2=0.9, amax=None, maxiter=10, verbose=verbose)
        mult = 4
        while (alpha is None) & (mult < 10000):
            if (mult < 5) & (verbose in ("warn","all")):
                print('Warning: Now increasing the stabilization of the Hessian.')
            if verbose == "all": print(indent, "Regularization multiplier ", mult)
            parlist = list(argv)
            parlist[len(parlist)-1] = mult * parlist[len(parlist)-1]
            params = tuple(parlist)
            hess = f_hesse(x_k, *params)
            p_k = -np.linalg.solve(hess, grad_k) # if alpha is None, NewtonLS_Wolfe returns the input values
            alpha, f_k, grad_k = NewtonLS_Wolfe(f_obj, f_grad, x_k, p_k, grad_k=grad_k, f_k=f_k,
                                                        args=argv, c1=1e-4, c2=0.9, amax=None, maxiter=10, verbose=verbose)
            mult = mult * 4

        if (mult > 10000): break
        x_k = x_k + alpha*p_k
        gnorm = np.linalg.norm(grad_k)
        if verbose == "all": print(indent, k, alpha, f_k, gnorm , gnorm/gnormold)
        gnormold = gnorm

        if gnorm < tol: break

    return x_k, f_k, k+1
