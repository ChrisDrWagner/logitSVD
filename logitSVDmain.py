# ------------------------------------------------------------------------------
# logitSVD: A SVD and user feature based method to approximate
#           binary and multinomial user-item-matrix
#
#
# Author: Christian Wagner, May 2020
# ------------------------------------------------------------------------------

"""
Function call
    P, C, Z, E, Q, t, z_score, p_value  = logitSVD(X, R, depth, la = 0.0, E = None, Q = None, t=None, method ="alternating",
             tol = 1e-4, maxit = 20, tolNewton = None, maxitNewton = 100, verbose = "warn")

Parameters
    X      : ndarray[nuser,nfeature], user feature vectors
    R      : ndarray[nuser,nitem], user-item-matrix (target)
    depth  : int, model parameter, depth of the embeddings = number of different models
    la     : float, regularization paramter
    E      : ndarray[nfeature,depth], initial solution for the feature weights (embeddings)
    Q      : ndarray[depth,nitem], initial solution for the item embeddings (model combination parameter)
    t      : ndarray[max(R)], initial solution for intercepts (only for multinomial case)
    method : string, binary: alternating [alter], fullNewton [full], alter_full, i.e. first alter, then fullNewton
                multinomial: alternating2 [alter2], 2 alternating steps  1. Q,t and 2. E,t
                             alternating3 [alter3], 3 alternating steps  1. Q, 2. t, 3. E
    tol    : float, alternating methods stop if the reduction of the log-likelihood is smaller than tol
    maxit  : int, maximum number of iterations of the alternating methods
    tolNewton: float, Newton's method stops if the 2-norm of the gradient becomes smaller than tolNewton
    maxitNewton: int, maximum number of iterations of Newton's method
    verbose: string, ("none" | "warn" | "all"), print warnings and convergence progress. Default is "warn"

Returns
    P      : binary: ndarray[nuser,nitem], P[u,i] is the probability that R[u,i] = 1
             multinomial: ndarray[max(R),nuser,nitem], P[k,u,i] is the probability that R[u,i] = k
    C      : ndarray[nuser,nitem], most likely class - None for binary
    Z      : ndarray[nuser,nitem], Z = X E Q
    E      : ndarray[nfeature,depth], solution for the parameter E
    Q      : ndarray[depth,nitem], solution for the parameter Q
    t      : ndarray[max(R)], solution for the intercepts t - None for binary
    z-score: ndarray[nfeature,depth], z-score from Wald test for the parameter E
    p-value: ndarray[nfeature,depth], p-values from Wald test for the parameter E

"""

import numpy as np
import scipy.stats as stat
from logitSVDNewtonLS import NewtonLS
from logitSVDderivatives import binlogitSVD_objE, binlogitSVD_gradE, binlogitSVD_hesseE, \
                                   binlogitSVD_objQ, binlogitSVD_gradQ, binlogitSVD_hesseQ, \
                                   binlogitSVD_objEQ, binlogitSVD_gradEQ, binlogitSVD_hesseEQ,\
                                   logitSVD_objTE, logitSVD_gradTE, logitSVD_hesseTE,\
                                   logitSVD_objT, logitSVD_gradT, logitSVD_hesseT, \
                                   logitSVD_objE, logitSVD_gradE, logitSVD_hesseE, \
                                   logitSVD_objQ, logitSVD_gradQ, logitSVD_hesseQ, \
                                   logitSVD_objTQ, logitSVD_gradTQ, logitSVD_hesseTQ, \
                                   logitSVD_objTEQ, logistic
#from statistics import NormalDist
import math

def checkAscending(te,nintercept):
    asc = True
    for i in range(0, nintercept - 1):
        if te[i] > te[i + 1]:
            te[i + 1] = te[i] + np.maximum(1e-4, 0.01 * np.abs(te[i]))
            asc =  False
    return asc, te

def sf(x):
    'Cumulative distribution function for the standard normal distribution'
    return 1-(1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def waldTestOrdered2(t, E, Q, X, R, la):

    nuser = X.shape[0]
    nfeature = X.shape[1]
    ndepth = Q.shape[0]
    nitem = Q.shape[1]
    nintercept = t.size

    e = E[:, 0:ndepth].transpose().reshape(nfeature*ndepth)
    te = np.append(t,e)
    F = logitSVD_hesseTE(te, Q, X, R, la)
    Finv = np.linalg.inv(F)
    sigma = np.sqrt(np.diagonal(Finv))
    sigma.reshape(nintercept+nfeature*ndepth)

    z_score = te/sigma
   # p_value_list = ([stat.norm.sf(abs(x))*2 for x in z_score])
    p_value_list = ([sf(abs(x))*2 for x in z_score])
    p_value = np.array(p_value_list)

    z_score = z_score[nintercept:].reshape([ndepth, nfeature]).transpose()
    p_value = p_value[nintercept:].reshape([ndepth, nfeature]).transpose()

    return z_score, p_value


def waldTestOrdered3(t, E, Q, X, R, la):

    nuser = X.shape[0]
    nfeature = X.shape[1]
    ndepth = Q.shape[0]
    nitem = Q.shape[1]
    nintercept = t.size

    e = E[:, 0:ndepth].transpose().reshape(nfeature*ndepth)
    F = logitSVD_hesseE(e, t, Q, X, R, la)
    Finv = np.linalg.inv(F)
    sigma = np.sqrt(np.diagonal(Finv))
    sigma.reshape(nfeature*ndepth)

    z_score = e/sigma
   # p_value_list = ([stat.norm.sf(abs(x))*2 for x in z_score])
    p_value_list = ([sf(abs(x))*2 for x in z_score])
    p_value = np.array(p_value_list)

    z_score = z_score.reshape([ndepth, nfeature]).transpose()
    p_value = p_value.reshape([ndepth, nfeature]).transpose()

    return z_score, p_value


def waldTestBin(E, Q, X, R, la):

    nuser = X.shape[0]
    nfeature = X.shape[1]
    ndepth = Q.shape[0]
    nitem = Q.shape[1]

    e = E[:,0:ndepth].transpose().reshape(nfeature*ndepth)
    F = binlogitSVD_hesseE(e, Q, X, R, la)
    Finv = np.linalg.inv(F)
    sigma = np.sqrt(np.diagonal(Finv))
    sigma.reshape(nfeature*ndepth)

    z_score = e/sigma
   # p_value_list = ([stat.norm.sf(abs(x))*2 for x in z_score])
    p_value_list = ([sf(abs(x))*2 for x in z_score])
    p_value = np.array(p_value_list)

    z_score = z_score.reshape([ndepth, nfeature]).transpose()
    p_value = p_value.reshape([ndepth, nfeature]).transpose()

    return z_score, p_value

def orderedPDs(t,E,Q,X):

    nuser = X.shape[0]
    nfeature = X.shape[1]
    ndepth = Q.shape[0]
    nitem = Q.shape[1]
    nintercept = t.size

    t = np.insert(t,0,-np.inf)
    t = np.append(t,np.inf)

    Z = np.matmul(np.matmul(X, E), Q)
    P = np.zeros([nintercept+1, nuser, nitem])
    Pmax = np.zeros([nuser, nitem])
    C = np.zeros([nuser, nitem])

    for k in range(0,nintercept+1):
        Zr = Z.copy()
        Zl = Z.copy()
        Zl += t[k]
        Zr += t[k+1]
        zr = Zr.reshape(nuser*nitem)
        zl = Zl.reshape(nuser*nitem)
        pr = logistic(zr)
        pl = logistic(zl)
        pk = pr - pl
        Pk = pk.reshape([nuser,nitem])
        C[Pk > Pmax] = k
        Pmax = np.maximum(Pmax,Pk)
        P[k,:,:] = Pk

    return C, P, Z

def logitSVDbin(X, R, depth, la, E = None, Q = None, method ="alternating",
                tol = 1e-4, maxit = 20, tolNewton = None, maxitNewton = 100, verbose = "warn"):

    if X.shape[0] != R.shape[0]:
        print("Error: X and R must have the same number of rows !")
        return None
    nuser = X.shape[0]
    nitem = R.shape[1]
    nfeature = X.shape[1]
    ndepth = depth
    niter = 0
    nnewton = 0
    if Q is None:
    # initialize Q, i.e. assume which items are described by the same model
        Q = np.zeros([ndepth,nitem])
        cm = np.ceil(nitem/ndepth-0.01)
        fm = np.floor(nitem/ndepth+0.01)
        num = np.floor((nitem/ndepth-fm)*ndepth+0.01).astype(int)
        for d in range(0,num):
            for i in range(0,nitem):
                if ((i >= d * cm) & (i < (d + 1) * cm)): Q[d,i] = 1
        offset = num*cm
        for d in range(num,ndepth):
            for i in range(0,nitem):
                if ((i >= offset+(d-num)*fm) & (i < offset+(d-num+1)*fm)): Q[d,i] = 1
    else:
        if Q.shape[0] != ndepth:
            print("Error: Q must have ndepth rows!")
            return None
        if Q.shape[1] != nitem:
            print("Error: Q and R must have the same number of columns!")
            return None

    if E is None:
    # initialize E: intercept represents the average number of events
    #               the other parameters are initialized with 0
        Rinfty = R.copy()
        Rinfty[np.isnan(Rinfty)] = -np.inf # to avoid runtime warning from greater
        prob = np.sum(Rinfty > 0, axis=0) / np.sum(np.isnan(R) == False, axis=0)
        del Rinfty
        prob = np.maximum(np.minimum(prob, 1 - 1e-6), 1e-6)
        lprob = -np.log((1.0 / prob) - 1)
        m = np.matmul(Q,Q.T)
        rhs = np.matmul(Q,lprob)
        try:
            sol = np.linalg.solve(m,rhs)
            E = np.zeros([nfeature,ndepth])
            E[0, :] = sol
        except np.linalg.LinAlgError as err:
            if 'Singular matrix' in str(err):
                print("Error: Q must have maximal rank!")
                return None
            else:
                raise
    else:
        if E.shape[0] != nfeature:
            print("Error: Number of rows of E must be equal to the number of colmns of X!")
            return None
        if E.shape[1] != ndepth:
            print("Error: E must have depth columns!")
            return None

    if tolNewton is None:
        tolNewton = ndepth*(nfeature+nitem)*1e-10

    e = E[:, 0:ndepth].transpose().reshape(nfeature*ndepth)
    q = Q.reshape(ndepth*nitem)

    # Optimize feature embeddings for given Q
    if verbose == "all":
        print("Initial optimization of item embeddings:")
    e, err_prev, k = NewtonLS(e, binlogitSVD_objE, binlogitSVD_gradE, binlogitSVD_hesseE, maxit = maxitNewton,
                 tol = tolNewton, verbose = verbose, argv = (Q, X, R, la))
    E = e.reshape(ndepth, nfeature).transpose()
    nnewton += k

    if method not in ("fullNewton","full"):
        if verbose == "all": print("Alternating parameter estimation:")
        for it in range(1,maxit+1):
            if verbose == "all": print(it, " Optimizing item embeddings:")
            q, err, k = NewtonLS(q, binlogitSVD_objQ, binlogitSVD_gradQ, binlogitSVD_hesseQ, maxit = maxitNewton,
                 tol = tolNewton, verbose = verbose, indent = "        ", argv = (E, X, R, la))
            Q = q.reshape([ndepth,nitem])
            nnewton += k
            if verbose == "all": print(it, " Optimizing feature embeddings:")
            e, err, k = NewtonLS(e, binlogitSVD_objE, binlogitSVD_gradE, binlogitSVD_hesseE, maxit = maxitNewton,
                 tol = tolNewton, verbose = verbose, indent = "        ",argv = (Q, X, R, la))
            E = e.reshape(ndepth, nfeature).transpose()
            nnewton += k
            if verbose == "all": print("Alternating convergence: ", it, err, err/err_prev)
            if err/err_prev > (1.0-tol): break
            err_prev = err
        z_score, p_value = waldTestBin(E, Q, X, R, la)
        niter += it
    if method in ("fullNewton","full","alter_full"):
        if verbose == "all": print("Parameter estimation with full Newton's method:")
        x = np.append(e, q)
        x, err, k = NewtonLS(x, binlogitSVD_objEQ, binlogitSVD_gradEQ, binlogitSVD_hesseEQ, maxit = maxitNewton,
            tol = tolNewton, verbose = verbose, argv = (X, R, la))
        niter += 1
        nnewton += k
        e = x[:nfeature * ndepth]
        E = e.reshape(ndepth, nfeature).transpose()
        q = x[nfeature * ndepth: nfeature * ndepth + ndepth * nitem]
        Q = q.reshape(ndepth, nitem)

    print("logitSVD: log-likelihood after ", niter, "iterations with ", nnewton, " Newton steps: ", err)
    Z = np.matmul(np.matmul(X, E), Q)
    P = logistic(Z.reshape(nuser*nitem)).reshape([nuser,nitem])

    return P, Z, E, Q, z_score, p_value


def logitSVDordered(X, R, depth, la, E = None, Q = None, t = None, method ="alternating",
                    tol = 1e-4, maxit = 20, tolNewton = None, maxitNewton = 100, verbose = "warn"):

    if X.shape[0] != R.shape[0]:
        print("Error: X and R must have the same number of rows !")
        return None

    nuser = X.shape[0]
    nitem = R.shape[1]
    nfeature = X.shape[1]
    ndepth = depth
    nintercept = np.nanmax(R).astype(int)
    niter = 0
    nnewton = 0

    if Q is None:
    # initialize Q, i.e. assume which items are described by the same model
        Q = np.zeros([ndepth,nitem])
        cm = np.ceil(nitem/ndepth-0.01)
        fm = np.floor(nitem/ndepth+0.01)
        num = np.floor((nitem/ndepth-fm)*ndepth+0.01).astype(int)
        for d in range(0,num):
            for i in range(0,nitem):
                if ((i >= d * cm) & (i < (d + 1) * cm)): Q[d,i] = 1
        offset = num*cm
        for d in range(num,ndepth):
            for i in range(0,nitem):
                if ((i >= offset+(d-num)*fm) & (i < offset+(d-num+1)*fm)): Q[d,i] = 1
    else:
        if Q.shape[0] != ndepth+1:
            print("Error: Q must have ndepth+1 rows!")
            return None
        if Q.shape[1] != nitem:
            print("Error: Q and R must have the same number of columns!")
            return None
    q = Q.reshape(ndepth*nitem)

    # prepare creating the start solution for Q and T
    if (E is None) & (t is None):
        Rinfty = R.copy()
        Rinfty[np.isnan(Rinfty)] = np.inf # to avoid runtime warning from greater
        prob = np.zeros([nitem, nintercept])
        for k in range(0, nintercept):
            prob[:, k] = np.sum(Rinfty < (k + 1), axis=0) / np.sum(np.isnan(R) == False, axis=0)
        del Rinfty
        prob = np.maximum(np.minimum(prob, 1 - 1e-6), 1e-6)
        lprob = -np.log((1.0 / prob) - 1)
        Qprob = np.matmul(Q, lprob)
        rhs = np.zeros(ndepth + nintercept+1)
        rhs[0:nintercept] = np.sum(lprob, axis=0)
        rhs[nintercept:nintercept + ndepth] = np.sum(Qprob, axis=1)
        m11 = nitem*np.identity(nintercept)
        Qsum = np.sum(Q,axis = 1)
        m12 = np.matmul(np.ones([nintercept,1]),Qsum.reshape([1,ndepth]))
        m22 = nintercept*np.matmul(Q,Q.T)
        m21 = (Qsum*np.ones([ndepth,nintercept]).T).T
        m = np.zeros([ndepth+nintercept+1,ndepth+nintercept])
        m[0:nintercept,0:nintercept] = m11
        m[0:nintercept,nintercept:nintercept+ndepth] = m12
        m[nintercept:nintercept+ndepth,0:nintercept] = m21
        m[nintercept:nintercept+ndepth,nintercept:nintercept + ndepth] = m22

        # without the additional row the linear system would be singular, but actually no problem as
        # the rhs is orthogonal to the kernel of the matrix
        m[ndepth+nintercept,0] = 1

        mm = np.matmul(m.T,m)
        mrhs = np.matmul(m.T,rhs)

        try:
            sol = np.linalg.solve(mm, mrhs)
            t = np.reshape(sol[0:nintercept], [1, nintercept])
            eInit = sol[nintercept:nintercept + ndepth]
            E = np.zeros([nfeature, ndepth])
            E[0, :] = eInit
        except np.linalg.LinAlgError as err:
            if 'Singular matrix' in str(err):
                print("Error: Q must have maximal rank!")
                return None
            else:
                raise

    if E is None:
        E = np.zeros([nfeature,ndepth])
    else:
        if E.shape[0] != nfeature:
            print("Error: Number of rows of E must be equal to the number of colmns of X!")
            return None
        if E.shape[1] != ndepth:
            print("Error: E must have depth columns!")
            return None
    e = E.reshape(nfeature*depth)

    if t is None:
        t = np.zeros(nintercept)
        for i in range(0, nintercept):
            t[i] = 1.01 * i
    else:
        if t.size != nintercept:
            print("Error: Size of t must equal the largest entry in R !")
            return None
        asc, t = checkAscending(t.reshape(nintercept), nintercept)
        if asc == False: print("Warning: Not strictly increasing t has been corrected.")

    if tolNewton is None:
        tolNewton = ndepth*(nfeature+nitem)*1e-12

    # Optimize feature embeddings for given Q
    if verbose == "all":
        print("Initial optimization of item embeddings:")
    x = np.append(t,e)
    x, err_prev, k = NewtonLS(x, logitSVD_objTE, logitSVD_gradTE, logitSVD_hesseTE, maxit = maxitNewton,
                 tol = tolNewton, verbose = verbose, argv = (Q, X, R, la))
    nnewton += k
    t = x[:nintercept]
    e = x[nintercept:]
    E = e.reshape(ndepth, nfeature).transpose()

    if method in ("alter3","alternating3"):
        if verbose == "all": print("3-step alternating parameter estimation:")
        for it in range(1,maxit+1):
            if verbose == "all": print(it, " Optimizing item embeddings:")
            # todo: check if t is ascending and create start vector
            q, err, k = NewtonLS(q, logitSVD_objQ, logitSVD_gradQ, logitSVD_hesseQ, maxit=maxitNewton,
                         tol=tolNewton, verbose=verbose, indent="        ", argv=(t, E, X, R, la))
            Q = q.reshape([ndepth, nitem])
            nnewton += k
            if verbose == "all": print(it, " Optimizing intercepts:")
            t, err, k = NewtonLS(t, logitSVD_objT, logitSVD_gradT, logitSVD_hesseT, maxit=maxitNewton,
                         tol=tolNewton, verbose=verbose, indent="        ", argv=(E, Q, X, R, la))
            nnewton += k
            if verbose == "all": print(it, " Optimizing feature embeddings:")
            # todo: check if t is ascending and create start vector
            e, err, k = NewtonLS(e, logitSVD_objE, logitSVD_gradE, logitSVD_hesseE, maxit=maxitNewton,
                         tol=tolNewton, verbose=verbose, indent="        ", argv=(t, Q, X, R, la))
            E = e.reshape(ndepth, nfeature).transpose()
            nnewton += k

            if verbose == "all": print("Alternating convergence: ", it, err, err / err_prev)
            if err / err_prev > (1.0 - tol): break
            err_prev = err
        z_score, p_value = waldTestOrdered3(t, E, Q, X, R, la)
        niter += it
    else:
        if verbose == "all": print("2-step alternating parameter estimation:")
        for it in range(1,maxit+1):
            if verbose == "all": print(it, " Optimizing item embeddings:")
            x = np.append(t,q)
            x, err, k = NewtonLS(x, logitSVD_objTQ, logitSVD_gradTQ, logitSVD_hesseTQ, maxit = maxitNewton,
                 tol = tolNewton, verbose = verbose, indent = "        ", argv = (E, X, R, la))
            nnewton += k
            t = x[:nintercept]
            q = x[nintercept:]
            Q = q.reshape([ndepth,nitem])
            if verbose == "all": print(it, " Optimizing feature embeddings:")
            x = np.append(t,e)
            x, err, k = NewtonLS(x, logitSVD_objTE, logitSVD_gradTE, logitSVD_hesseTE, maxit = maxitNewton,
                 tol = tolNewton, verbose = verbose, indent = "        ",argv = (Q, X, R, la))
            nnewton += k
            t = x[:nintercept]
            e = x[nintercept:]
            E = e.reshape(ndepth, nfeature).transpose()

            if verbose == "all": print("Alternating convergence: ", it, err, err/err_prev)
            if err/err_prev > (1.0-tol): break
            err_prev = err
        z_score, p_value = waldTestOrdered2(t, E, Q, X, R, la)
        niter += it

    print("logitSVD: log-likelihood after ", niter, "iterations with ", nnewton, " Newton steps: ", err)
    C, P, Z = orderedPDs(t, E, Q, X)



    return P, C, Z, E, Q, t, z_score, p_value

def logitSVD(X, R, depth, la = 0.0, E = None, Q = None, t=None, method ="alternating",
             tol = 1e-4, maxit = 20, tolNewton = None, maxitNewton = 100, verbose = "warn"):

    try:
        idx = (np.isnan(R) == False)
        R[idx] = R[idx].astype(int)
    except ValueError as verr:
        print("Error: R needs to be an integer matrix !")
        return None
    except Exception as ex:
        print("Error: R needs to be an integer matrix !")

    Rmax = np.nanmax(R).astype(int)
    Rmin = np.nanmin(R).astype(int)

    if Rmin != 0:
        print("Error: Minimum value of R needs to be 0 !")
        return None

    if Rmax == 0:
        print("Error: Maximum value of R needs to be at least 1 !")
        return None

    if Rmax == 1:
        P, Z, E, Q, z_score, p_value = logitSVDbin(X, R, depth, la, E=E, Q=Q, method=method, tol=tol, maxit=maxit, tolNewton=tolNewton, maxitNewton=maxitNewton, verbose=verbose)
        t = None
        C = None
    else:
        P, C, Z, E, Q, t, z_score, p_value = logitSVDordered(X, R, depth, la, E=E, Q=Q, t=t, method=method, tol=tol, maxit=maxit, tolNewton=tolNewton, maxitNewton=maxitNewton, verbose=verbose)

    return P, C, Z, E, Q, t, z_score, p_value

