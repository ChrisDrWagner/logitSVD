# ------------------------------------------------------------------------------
# Objective functions and derivatives required for logitSVD:
#
#
# Author: Christian Wagner, May 2020
# ------------------------------------------------------------------------------

import numpy as np

def logistic(t):
    # stable calculation of the logistic function, returns 1 / (1 + exp(-t))
    idx = t > 0
    out = np.empty(t.size, dtype=np.float)
    out[idx] = 1. / (1 + np.exp(-t[idx]))
    exp_t = np.exp(t[~idx])
    out[~idx] = exp_t / (1. + exp_t)
    return out

def logisticR(t,r):
    # stable calculation of the logistic function, returns 1 / (1 + exp(-t))
    pidx = (t > 0) & (np.isnan(r) == False)
    nidx = (t <= 0) & (np.isnan(r) == False)
    out = np.zeros(t.size)
    out[pidx] = 1. / (1 + np.exp(-t[pidx]))
    exp_t = np.exp(t[nidx])
    out[nidx] = exp_t / (1. + exp_t)
    return out

def log_logistic(t):
    # stable calculation of the logistic loss function, returns log(1 / (1 + exp(-t)))
    idx = t > 0
    out = np.zeros_like(t)
    out[idx] = np.log(1 + np.exp(-t[idx]))
    out[~idx] = (-t[~idx] + np.log(1 + np.exp(t[~idx])))
    return out

def logitSVD_objE(x0, t, Q, X, R, la):

    nuser = X.shape[0]
    nfeature = X.shape[1]
    ndepth = Q.shape[0]
    nitem = Q.shape[1]
    nintercept = t.size

    t = np.insert(t,0,-np.inf)
    t = np.append(t,np.inf)
    e = x0[0:nfeature*ndepth]
    E = e.reshape(ndepth, nfeature).transpose()
    Z = np.matmul(np.matmul(X, E), Q)

    Zr = Z.copy()
    Zl = Z.copy()

    tt = np.maximum(0,R.astype(int))
    Zl += t[tt]
    Zr += t[tt+1]

    zr = Zr.reshape(nuser*nitem)
    zl = Zl.reshape(nuser*nitem)
    r = R.reshape(nuser*nitem)
    pr = logisticR(zr,r)
    pl = logisticR(zl,r)

    log_prl = np.zeros(r.size)
    idx = (np.isnan(r) == False)
    log_prl[idx] = np.log(pr[idx] - pl[idx])

    logLike = log_prl[idx].sum()
    enorm = np.linalg.norm(x0)
    enorm = np.linalg.norm(E,ord='fro')
    qnorm = np.linalg.norm(Q,ord='fro')
    tnorm = np.linalg.norm(t[1:nintercept+1])

    return -logLike + 0.5*la*(enorm*enorm+qnorm*qnorm+tnorm*tnorm)


def logitSVD_objT(x0, E, Q, X, R, la):

    nuser = X.shape[0]
    nfeature = X.shape[1]
    ndepth = Q.shape[0]
    nitem = Q.shape[1]
    nintercept = x0.size

    t = x0[:nintercept]
    t = np.insert(t,0,-np.inf)
    t = np.append(t,np.inf)

    Z = np.matmul(np.matmul(X, E), Q)
    Zr = Z.copy()
    Zl = Z.copy()

    tt = np.maximum(0,R.astype(int))
    Zl += t[tt]
    Zr += t[tt+1]

    zr = Zr.reshape(nuser*nitem)
    zl = Zl.reshape(nuser*nitem)
    r = R.reshape(nuser*nitem)
    pr = logisticR(zr,r)
    pl = logisticR(zl,r)

    log_prl = np.zeros(r.size)
    idx = (np.isnan(r) == False)
    log_prl[idx] = np.log(pr[idx] - pl[idx])

    logLike = log_prl[idx].sum()
    enorm = np.linalg.norm(E,ord='fro')
    qnorm = np.linalg.norm(Q,ord='fro')
    tnorm = np.linalg.norm(t[1:nintercept+1])

    return -logLike + 0.5 *la*(enorm * enorm + qnorm * qnorm + tnorm * tnorm)


def logitSVD_objQ(x0, t, E, X, R, la):

    nuser = X.shape[0]
    nfeature = X.shape[1]
    ndepth = E.shape[1]
    nitem = R.shape[1]
    nintercept = t.size

    t = np.insert(t,0,-np.inf)
    t = np.append(t,np.inf)
    q = x0[0: ndepth * nitem]
    Q = q.reshape(ndepth, nitem)
    Z = np.matmul(np.matmul(X, E), Q)
    Zr = Z.copy()
    Zl = Z.copy()

    tt = np.maximum(0,R.astype(int))
    Zl += t[tt]
    Zr += t[tt+1]

    zr = Zr.reshape(nuser*nitem)
    zl = Zl.reshape(nuser*nitem)
    r = R.reshape(nuser*nitem)
    pr = logisticR(zr,r)
    pl = logisticR(zl,r)

    log_prl = np.zeros(r.size)
    idx = (np.isnan(r) == False)
    pdiff = pr[idx] - pl[idx]
    pdiff = np.maximum(1e-24,pdiff)
    log_prl[idx] = np.log(pdiff)

    logLike = log_prl[idx].sum()
    enorm = np.linalg.norm(E, ord='fro')
    qnorm = np.linalg.norm(Q, ord='fro')
    tnorm = np.linalg.norm(t[1:nintercept+1])

    return -logLike + 0.5 * la * (enorm * enorm + qnorm * qnorm + tnorm * tnorm)

def logitSVD_gradT(x0, E, Q, X, R, la):

    nuser = X.shape[0]
    nfeature = X.shape[1]
    ndepth = Q.shape[0]
    nitem = Q.shape[1]
    nintercept = x0.size

    t = x0[:nintercept]
    t = np.insert(t,0,-np.inf)
    t = np.append(t,np.inf)
    Z = np.matmul(np.matmul(X, E), Q)
    Zr = Z.copy()
    Zl = Z.copy()

    tt = np.maximum(0, R.astype(int))
    Zl += t[tt]
    Zr += t[tt + 1]
    zr = Zr.reshape(nuser*nitem)
    zl = Zl.reshape(nuser*nitem)
    r = R.reshape(nuser*nitem)

    pr = logisticR(zr,r)
    pl = logisticR(zl,r)
    Pr = pr.reshape([nuser, nitem])
    Pl = pl.reshape([nuser, nitem])

    grad = np.zeros(nintercept)
    for k in range(0,nintercept):
        idx = (R == k)
        grad[k] -= (Pr[idx]*(1-Pr[idx])/(Pr[idx] - Pl[idx])).sum()
        idx = (R == k+1)
        grad[k] += (Pl[idx]*(1-Pl[idx])/(Pr[idx] - Pl[idx])).sum()

    grad = grad +la*t[1:nintercept+1]

    return grad


def logitSVD_gradE(x0, t, Q, X, R, la):

    nuser = X.shape[0]
    nfeature = X.shape[1]
    ndepth = Q.shape[0]
    nitem = Q.shape[1]

    t = np.insert(t,0,-np.inf)
    t = np.append(t,np.inf)
    e = x0[0:nfeature*ndepth]
    E = e.reshape(ndepth, nfeature).transpose()
    Z = np.matmul(np.matmul(X, E), Q)
    Zr = Z.copy()
    Zl = Z.copy()

    tt = np.maximum(0, R.astype(int))
    Zl += t[tt]
    Zr += t[tt + 1]
    zr = Zr.reshape(nuser*nitem)
    zl = Zl.reshape(nuser*nitem)
    r = R.reshape(nuser*nitem)

    pr = logisticR(zr,r)
    pl = logisticR(zl,r)
    Pr = pr.reshape([nuser, nitem])
    Pl = pl.reshape([nuser, nitem])

    PP = (1 - Pr - Pl)
    PP[np.isnan(R)] = 0.0

    mat = -np.matmul(np.matmul(X.transpose(),PP),Q.transpose())
    grad = mat.transpose().reshape(nfeature*ndepth)
    grad = grad +la*e

    return grad


def logitSVD_gradQ(x0, t, E, X, R, la):

    nuser = X.shape[0]
    nfeature = X.shape[1]
    ndepth = E.shape[1]
    nitem = R.shape[1]

    t = np.insert(t,0,-np.inf)
    t = np.append(t,np.inf)
    q = x0[: ndepth * nitem]
    Q = q.reshape(ndepth, nitem)
    Z = np.matmul(np.matmul(X, E), Q)
    Zr = Z.copy()
    Zl = Z.copy()

    tt = np.maximum(0, R.astype(int))
    Zl += t[tt]
    Zr += t[tt + 1]
    zr = Zr.reshape(nuser*nitem)
    zl = Zl.reshape(nuser*nitem)
    r = R.reshape(nuser*nitem)

    pr = logisticR(zr,r)
    pl = logisticR(zl,r)
    Pr = pr.reshape([nuser, nitem])
    Pl = pl.reshape([nuser, nitem])

    PP = (1 - Pr - Pl)
    PP[np.isnan(R)] = 0.0

    dZdQ = -np.matmul(np.matmul(X,E).transpose(),PP)
    grad = dZdQ.reshape(ndepth*nitem)
    grad = grad +la*q

    return grad

def logitSVD_hesseT(x0, E, Q, X, R, la):

    nuser = X.shape[0]
    nfeature = X.shape[1]
    ndepth = Q.shape[0]
    nitem = Q.shape[1]
    nintercept = np.nanmax(R).astype(int)

    t = x0[:nintercept]
    t = np.insert(t,0,-np.inf)
    t = np.append(t,np.inf)
    Z = np.matmul(np.matmul(X, E), Q)
    Zr = Z.copy()
    Zl = Z.copy()

    tt = np.maximum(0, R.astype(int))
    Zl += t[tt]
    Zr += t[tt + 1]
    zr = Zr.reshape(nuser*nitem)
    zl = Zl.reshape(nuser*nitem)
    r = R.reshape(nuser*nitem)

    pr = logisticR(zr,r)
    pl = logisticR(zl,r)
    Pr = pr.reshape([nuser, nitem])
    Pl = pl.reshape([nuser, nitem])

    P3 = Pr*(1-Pr) + Pl*(1-Pl)
    P3[np.isnan(R)] = 0.0

    length = nintercept
    hesse = np.zeros([length,length])

    for k in range(0,nintercept):
        idx = (R == k)
        Pdiff = (Pr[idx] - Pl[idx])
        dgdp = 1+(Pl[idx]*(1-Pl[idx]))/(Pdiff * Pdiff)
        hesse[k,k] += (dgdp* Pr[idx] * (1 - Pr[idx])).sum()
        if k > 0:
            hesse[k,k-1] = -((Pr[idx] * (1 - Pr[idx])*Pl[idx]*(1-Pl[idx]))/(Pdiff * Pdiff)).sum()
        idx = (R == k+1)
        Pdiff = (Pr[idx] - Pl[idx])
        dgdp = 1+(Pr[idx]*(1-Pr[idx]))/(Pdiff * Pdiff)
        hesse[k,k] += (dgdp*Pl[idx]*(1-Pl[idx])).sum()
        if k < nintercept-1:
            hesse[k,k+1] = -((Pr[idx] * (1 - Pr[idx])*Pl[idx]*(1-Pl[idx]))/(Pdiff * Pdiff)).sum()

    hesse = hesse + la * np.identity(length)

    return hesse


def logitSVD_hesseE(x0, t, Q, X, R, la):

    nuser = X.shape[0]
    nfeature = X.shape[1]
    ndepth = Q.shape[0]
    nitem = Q.shape[1]

    t = np.insert(t,0,-np.inf)
    t = np.append(t,np.inf)
    e = x0[0:nfeature*ndepth]
    E = e.reshape(ndepth, nfeature).transpose()
    Z = np.matmul(np.matmul(X, E), Q)
    Zr = Z.copy()
    Zl = Z.copy()

    tt = np.maximum(0, R.astype(int))
    Zl += t[tt]
    Zr += t[tt + 1]
    zr = Zr.reshape(nuser*nitem)
    zl = Zl.reshape(nuser*nitem)
    r = R.reshape(nuser*nitem)

    pr = logisticR(zr,r)
    pl = logisticR(zl,r)
    Pr = pr.reshape([nuser, nitem])
    Pl = pl.reshape([nuser, nitem])

    P3 = Pr*(1-Pr) + Pl*(1-Pl)
    P3[np.isnan(R)] = 0.0

    length = nfeature*ndepth
    hesse = np.zeros([length,length])

    for d1 in range(0, ndepth):
        for d2 in range(0, ndepth):
            offset_row = nfeature * d1
            offset_col = nfeature * d2
            q1 = Q[d1, :]
            q2 = Q[d2, :]
            v = np.matmul(P3, q1 * q2)
            K = (v * X.T).T
            h = np.matmul(X.transpose(), K)
            hesse[offset_row: offset_row + nfeature, offset_col: offset_col + nfeature] = h

    hesse = hesse + la * np.identity(length)

    return hesse

def logitSVD_hesseQ(x0, t, E, X, R, la):

    nuser = X.shape[0]
    nfeature = X.shape[1]
    ndepth = E.shape[1]
    nitem = R.shape[1]

    t = np.insert(t,0,-np.inf)
    t = np.append(t,np.inf)
    q = x0[: ndepth * nitem]
    Q = q.reshape(ndepth, nitem)
    Z = np.matmul(np.matmul(X, E), Q)
    Zr = Z.copy()
    Zl = Z.copy()

    tt = np.maximum(0, R.astype(int))
    Zl += t[tt]
    Zr += t[tt + 1]
    zr = Zr.reshape(nuser*nitem)
    zl = Zl.reshape(nuser*nitem)
    r = R.reshape(nuser*nitem)

    pr = logisticR(zr,r)
    pl = logisticR(zl,r)
    Pr = pr.reshape([nuser, nitem])
    Pl = pl.reshape([nuser, nitem])

    P3 = Pr*(1-Pr) + Pl*(1-Pl)
    P3[np.isnan(R)] = 0.0

    length = nitem * ndepth
    hesse = np.zeros([length,length])

    for d1 in range(0,ndepth):
        for d2 in range(0,ndepth):
            offset_row = nitem*d1;
            offset_col = nitem*d2;
            u1 = np.matmul(X,E[:,d1])
            u2 = np.matmul(X,E[:,d2])
            v = np.matmul(P3.transpose(),u1*u2)
            h = np.diag(v)
            hesse[offset_row: offset_row+nitem, offset_col: offset_col+nitem] = h

    hesse = hesse + la * np.identity(length)

    return hesse


def logitSVD_objTEQ(x, X, R, la):

    nuser = R.shape[0]
    nitem = R.shape[1]
    nfeature = X.shape[1]
    nintercept = np.nanmax(R).astype(int)
    ndepth = np.floor((x.shape[0]-nintercept+nfeature)/(nfeature+nitem)+0.01).astype(int)

    t = x[:nintercept]
    t = np.insert(t,0,-np.inf)
    t = np.append(t,np.inf)
    e = x[nintercept:nintercept+nfeature*ndepth]
    E = e.reshape(ndepth, nfeature).transpose()
    q = x[nintercept + nfeature*ndepth:nintercept + nfeature*ndepth + nitem*ndepth]
    Q = q.reshape([ndepth,nitem])
    Z = np.matmul(np.matmul(X, E), Q)

    Zr = Z.copy()
    Zl = Z.copy()

    tt = np.maximum(0,R.astype(int))
    Zl += t[tt]
    Zr += t[tt+1]

    zr = Zr.reshape(nuser*nitem)
    zl = Zl.reshape(nuser*nitem)
    r = R.reshape(nuser*nitem)
    pr = logisticR(zr,r)
    pl = logisticR(zl,r)

    log_prl = np.zeros(r.size)
    idx = (np.isnan(r) == False)
    log_prl[idx] = np.log(pr[idx] - pl[idx])

    logLike = log_prl[idx].sum()

    enorm = np.linalg.norm(E, ord='fro')
    qnorm = np.linalg.norm(Q, ord='fro')
    tnorm = np.linalg.norm(t[1:nintercept+1])

    return -logLike + 0.5 * la * (enorm * enorm + qnorm * qnorm + tnorm * tnorm)


def logitSVD_objTE(x0, Q, X, R, la):

    nuser = X.shape[0]
    nfeature = X.shape[1]
    ndepth = Q.shape[0]
    nitem = Q.shape[1]
    nintercept = np.nanmax(R).astype(int)

    t = x0[:nintercept]
    t = np.insert(t,0,-np.inf)
    t = np.append(t,np.inf)
    e = x0[nintercept:nintercept + nfeature*ndepth]
    E = e.reshape(ndepth, nfeature).transpose()
    Z = np.matmul(np.matmul(X, E), Q)

    Zr = Z.copy()
    Zl = Z.copy()

    tt = np.maximum(0,R.astype(int))
    Zl += t[tt]
    Zr += t[tt+1]

    zr = Zr.reshape(nuser*nitem)
    zl = Zl.reshape(nuser*nitem)
    r = R.reshape(nuser*nitem)
    pr = logisticR(zr,r)
    pl = logisticR(zl,r)

    log_prl = np.zeros(r.size)
    idx = (np.isnan(r) == False)
    log_prl[idx] = np.log(pr[idx] - pl[idx])

    logLike = log_prl[idx].sum()
    enorm = np.linalg.norm(E, ord='fro')
    qnorm = np.linalg.norm(Q, ord='fro')
    tnorm = np.linalg.norm(t[1:nintercept+1])

    return -logLike + 0.5 * la * (enorm * enorm + qnorm * qnorm + tnorm * tnorm)

def logitSVD_objTQ(x0, E, X, R, la):

    nuser = X.shape[0]
    nfeature = X.shape[1]
    ndepth = E.shape[1]
    nitem = R.shape[1]
    nintercept = np.nanmax(R).astype(int)

    t = x0[:nintercept]
    t = np.insert(t,0,-np.inf)
    t = np.append(t,np.inf)
    q = x0[nintercept: nintercept+ndepth*nitem]
    Q = q.reshape(ndepth, nitem)
    Z = np.matmul(np.matmul(X, E), Q)
    Zr = Z.copy()
    Zl = Z.copy()

    tt = np.maximum(0,R.astype(int))
    Zl += t[tt]
    Zr += t[tt+1]

    zr = Zr.reshape(nuser*nitem)
    zl = Zl.reshape(nuser*nitem)
    r = R.reshape(nuser*nitem)
    pr = logisticR(zr,r)
    pl = logisticR(zl,r)

    log_prl = np.zeros(r.size)
    idx = (np.isnan(r) == False)
    log_prl[idx] = np.log(pr[idx] - pl[idx])

    logLike = log_prl[idx].sum()

    enorm = np.linalg.norm(E, ord='fro')
    qnorm = np.linalg.norm(Q, ord='fro')
    tnorm = np.linalg.norm(t[1:nintercept+1])

    return -logLike + 0.5 * la * (enorm * enorm + qnorm * qnorm + tnorm * tnorm)


def logitSVD_gradTE(x0, Q, X, R, la):

    nuser = X.shape[0]
    nfeature = X.shape[1]
    ndepth = Q.shape[0]
    nitem = Q.shape[1]
    nintercept = np.nanmax(R).astype(int)

    t = x0[:nintercept]
    t = np.insert(t,0,-np.inf)
    t = np.append(t,np.inf)
    e = x0[nintercept:nintercept + nfeature*ndepth]
    E = e.reshape(ndepth, nfeature).transpose()
    Z = np.matmul(np.matmul(X, E), Q)
    Zr = Z.copy()
    Zl = Z.copy()

    tt = np.maximum(0, R.astype(int))
    Zl += t[tt]
    Zr += t[tt + 1]
    zr = Zr.reshape(nuser*nitem)
    zl = Zl.reshape(nuser*nitem)
    r = R.reshape(nuser*nitem)

    pr = logisticR(zr,r)
    pl = logisticR(zl,r)
    Pr = pr.reshape([nuser, nitem])
    Pl = pl.reshape([nuser, nitem])

    PP = (1 - Pr - Pl)
    PP[np.isnan(R)] = 0.0

    gradT = np.zeros(nintercept)
    for k in range(0,nintercept):
        idx = (R == k)
        gradT[k] -= (Pr[idx]*(1-Pr[idx])/(Pr[idx] - Pl[idx])).sum()
        idx = (R == k+1)
        gradT[k] += (Pl[idx]*(1-Pl[idx])/(Pr[idx] - Pl[idx])).sum()

    mat = -np.matmul(np.matmul(X.transpose(),PP),Q.transpose())
    grad = mat.transpose().reshape(nfeature*ndepth)
    grad = np.append(gradT,grad)
    grad = grad +la*x0

    return grad

def logitSVD_hesseTE(x0, Q, X, R, la):

    nuser = X.shape[0]
    nfeature = X.shape[1]
    ndepth = Q.shape[0]
    nitem = Q.shape[1]
    nintercept = np.nanmax(R).astype(int)

    t = x0[:nintercept]
    t = np.insert(t,0,-np.inf)
    t = np.append(t,np.inf)
    e = x0[nintercept:nintercept + nfeature*ndepth]
    E = e.reshape(ndepth, nfeature).transpose()
    Z = np.matmul(np.matmul(X, E), Q)
    Zr = Z.copy()
    Zl = Z.copy()

    tt = np.maximum(0, R.astype(int))
    Zl += t[tt]
    Zr += t[tt + 1]
    zr = Zr.reshape(nuser*nitem)
    zl = Zl.reshape(nuser*nitem)
    r = R.reshape(nuser*nitem)

    pr = logisticR(zr,r)
    pl = logisticR(zl,r)
    Pr = pr.reshape([nuser, nitem])
    Pl = pl.reshape([nuser, nitem])

    P3 = Pr*(1-Pr) + Pl*(1-Pl)
    P3[np.isnan(R)] = 0.0

    length = nintercept + nfeature*ndepth
    hesse = np.zeros([length,length])

    for k in range(0,nintercept):
        idx = (R == k)
        Pdiff = (Pr[idx] - Pl[idx])
        dgdp = 1+(Pl[idx]*(1-Pl[idx]))/(Pdiff * Pdiff)
        hesse[k,k] += (dgdp* Pr[idx] * (1 - Pr[idx])).sum()
        if k > 0:
            hesse[k,k-1] = -((Pr[idx] * (1 - Pr[idx])*Pl[idx]*(1-Pl[idx]))/(Pdiff * Pdiff)).sum()
        idx = (R == k+1)
        Pdiff = (Pr[idx] - Pl[idx])
        dgdp = 1+(Pr[idx]*(1-Pr[idx]))/(Pdiff * Pdiff)
        hesse[k,k] += (dgdp*Pl[idx]*(1-Pl[idx])).sum()
        if k < nintercept-1:
            hesse[k,k+1] = -((Pr[idx] * (1 - Pr[idx])*Pl[idx]*(1-Pl[idx]))/(Pdiff * Pdiff)).sum()

    Pr2 = Pr*(1-Pr)
    Pl2 = Pl*(1-Pl)
    for k in range(0,nintercept):
        Pr2k = Pr2.copy()
        Pr2k[R!=k] = 0.0
        Mr = np.matmul(np.matmul(X.transpose(), Pr2k), Q.transpose())
        Mr = Mr.transpose().reshape(nfeature*ndepth)
        hesse[k,nintercept : nintercept + nfeature*ndepth] += Mr.T
        hesse[nintercept : nintercept + nfeature*ndepth, k] += Mr

        Pl2k = Pl2.copy()
        Pl2k[R!=(k+1)] = 0.0
        Ml = np.matmul(np.matmul(X.transpose(), Pl2k), Q.transpose())
        Ml = Ml.transpose().reshape(nfeature*ndepth)
        hesse[k,nintercept : nintercept + nfeature*ndepth] += Ml.T
        hesse[nintercept : nintercept + nfeature*ndepth, k] += Ml

    for d1 in range(0, ndepth):
        for d2 in range(0, ndepth):
            offset_row = nfeature * d1
            offset_col = nfeature * d2
            q1 = Q[d1, :]
            q2 = Q[d2, :]
            v = np.matmul(P3, q1 * q2)
            K = (v * X.T).T
            h = np.matmul(X.transpose(), K)
            hesse[nintercept+offset_row:nintercept+offset_row + nfeature, nintercept+offset_col:nintercept+offset_col + nfeature] = h

    hesse = hesse + la * np.identity(length)
    return hesse

def logitSVD_gradTQ(x0, E, X, R, la):

    nuser = X.shape[0]
    nfeature = X.shape[1]
    ndepth = E.shape[1]
    nitem = R.shape[1]
    nintercept = np.nanmax(R).astype(int)

    t = x0[:nintercept]
    t = np.insert(t,0,-np.inf)
    t = np.append(t,np.inf)
    q = x0[nintercept: nintercept+ndepth*nitem]
    Q = q.reshape(ndepth, nitem)
    Z = np.matmul(np.matmul(X, E), Q)
    Zr = Z.copy()
    Zl = Z.copy()

    tt = np.maximum(0, R.astype(int))
    Zl += t[tt]
    Zr += t[tt + 1]
    zr = Zr.reshape(nuser*nitem)
    zl = Zl.reshape(nuser*nitem)
    r = R.reshape(nuser*nitem)

    pr = logisticR(zr,r)
    pl = logisticR(zl,r)
    Pr = pr.reshape([nuser, nitem])
    Pl = pl.reshape([nuser, nitem])

    PP = (1 - Pr - Pl)
    PP[np.isnan(R)] = 0.0

    gradT = np.zeros(nintercept)
    for k in range(0,nintercept):
        idx = (R == k)
        gradT[k] -= (Pr[idx]*(1-Pr[idx])/(Pr[idx] - Pl[idx])).sum()
        idx = (R == k+1)
        gradT[k] += (Pl[idx]*(1-Pl[idx])/(Pr[idx] - Pl[idx])).sum()

    dZdQ = -np.matmul(np.matmul(X,E).transpose(),PP)
    grad = dZdQ.reshape(ndepth*nitem)
    grad = np.append(gradT,grad)
    grad = grad +la*x0

    return grad


def logitSVD_hesseTQ(x0, E, X, R, la):

    nuser = X.shape[0]
    nfeature = X.shape[1]
    ndepth = E.shape[1]
    nitem = R.shape[1]
    nintercept = np.nanmax(R).astype(int)

    t = x0[:nintercept]
    t = np.insert(t,0,-np.inf)
    t = np.append(t,np.inf)
    q = x0[nintercept: nintercept+ndepth*nitem]
    Q = q.reshape(ndepth, nitem)
    Z = np.matmul(np.matmul(X, E), Q)
    Zr = Z.copy()
    Zl = Z.copy()

    tt = np.maximum(0, R.astype(int))
    Zl += t[tt]
    Zr += t[tt + 1]
    zr = Zr.reshape(nuser*nitem)
    zl = Zl.reshape(nuser*nitem)
    r = R.reshape(nuser*nitem)

    pr = logisticR(zr,r)
    pl = logisticR(zl,r)
    Pr = pr.reshape([nuser, nitem])
    Pl = pl.reshape([nuser, nitem])

    P3 = Pr*(1-Pr) + Pl*(1-Pl)
    P3[np.isnan(R)] = 0.0

    length = nintercept + nitem*ndepth
    hesse = np.zeros([length,length])

    for k in range(0,nintercept):
        idx = (R == k)
        Pdiff = (Pr[idx] - Pl[idx])
        dgdp = 1+(Pl[idx]*(1-Pl[idx]))/(Pdiff * Pdiff)
        hesse[k,k] += (dgdp* Pr[idx] * (1 - Pr[idx])).sum()
        if k > 0:
            hesse[k,k-1] = -((Pr[idx] * (1 - Pr[idx])*Pl[idx]*(1-Pl[idx]))/(Pdiff * Pdiff)).sum()
        idx = (R == k+1)
        Pdiff = (Pr[idx] - Pl[idx])
        dgdp = 1+(Pr[idx]*(1-Pr[idx]))/(Pdiff * Pdiff)
        hesse[k,k] += (dgdp*Pl[idx]*(1-Pl[idx])).sum()
        if k < nintercept-1:
            hesse[k,k+1] = -((Pr[idx] * (1 - Pr[idx])*Pl[idx]*(1-Pl[idx]))/(Pdiff * Pdiff)).sum()

    Pr2 = Pr*(1-Pr)
    Pl2 = Pl*(1-Pl)
    for k in range(0,nintercept):
        Pr2k = Pr2.copy()
        Pr2k[R!=k] = 0.0
        Mr = np.matmul(np.matmul(X, E).transpose(), Pr2k)
        Mr = Mr.reshape(ndepth * nitem)
        hesse[k,nintercept : nintercept + nitem*ndepth] += Mr.T
        hesse[nintercept : nintercept + nitem*ndepth , k] += Mr

        Pl2k = Pl2.copy()
        Pl2k[R!=(k+1)] = 0.0
        Ml = np.matmul(np.matmul(X, E).transpose(), Pl2k)
        Ml = Ml.reshape(ndepth * nitem)
        hesse[k,nintercept : nintercept + nitem*ndepth] += Ml.T
        hesse[nintercept : nintercept + nitem*ndepth , k] += Ml

    for d1 in range(0,ndepth):
        for d2 in range(0,ndepth):
            offset_row = nitem*d1;
            offset_col = nitem*d2;
            u1 = np.matmul(X,E[:,d1])
            u2 = np.matmul(X,E[:,d2])
            v = np.matmul(P3.transpose(),u1*u2)
            h = np.diag(v)
            hesse[nintercept+offset_row:nintercept+offset_row+nitem,nintercept+offset_col:nintercept+offset_col+nitem] = h

    hesse = hesse + la * np.identity(length)

    return hesse

def binlogitSVD_objE(x0, Q, X, R, la):

    nuser = X.shape[0]
    nfeature = X.shape[1]
    ndepth = Q.shape[0]
    nitem = Q.shape[1]

    e = x0[:nfeature*ndepth]
    E = e.reshape(ndepth, nfeature).transpose()
    Z = np.matmul(np.matmul(X, E), Q)

    z = Z.reshape(nuser*nitem)
    r = R.reshape(nuser*nitem)

    pidx = (z > 0) & (np.isnan(r) == False)
    nidx = (z <= 0) & (np.isnan(r) == False)
    zr = np.zeros_like(z)
    zr[pidx] = -r[pidx]*np.log(1+np.exp(-z[pidx])) - (1-r[pidx])*z[pidx] -(1-r[pidx])*np.log(1+np.exp(-z[pidx]))
    zr[nidx] = -r[nidx]*np.log(1+np.exp(z[nidx])) + r[nidx]*z[nidx] - (1-r[nidx])*np.log(1+np.exp(z[nidx]))
    logLike = zr[pidx].sum()+zr[nidx].sum()

    enorm = np.linalg.norm(E, ord='fro')
    qnorm = np.linalg.norm(Q, ord='fro')

    return -logLike + 0.5 * la * (enorm * enorm + qnorm * qnorm)


def binlogitSVD_objQ(x0, E, X, R, la):

    nuser = X.shape[0]
    nfeature = X.shape[1]
    ndepth = E.shape[1]
    nitem = R.shape[1]

    q = x0[: ndepth * nitem]
    Q = q.reshape(ndepth, nitem)
    Z = np.matmul(np.matmul(X, E), Q)

    z = Z.reshape(nuser * nitem)
    r = R.reshape(nuser * nitem)

    pidx = (z > 0) & (np.isnan(r) == False)
    nidx = (z <= 0) & (np.isnan(r) == False)
    zr = np.zeros_like(z)
    zr[pidx] = -r[pidx] * np.log(1 + np.exp(-z[pidx])) - (1 - r[pidx]) * z[pidx] - (1 - r[pidx]) * np.log(
        1 + np.exp(-z[pidx]))
    zr[nidx] = -r[nidx] * np.log(1 + np.exp(z[nidx])) + r[nidx] * z[nidx] - (1 - r[nidx]) * np.log(1 + np.exp(z[nidx]))
    logLike = zr[pidx].sum() + zr[nidx].sum()

    enorm = np.linalg.norm(E, ord='fro')
    qnorm = np.linalg.norm(Q, ord='fro')

    return -logLike + 0.5 * la * (enorm * enorm + qnorm * qnorm)


def binlogitSVD_gradE(x0, Q, X, R, la):

    nuser = X.shape[0]
    nfeature = X.shape[1]
    ndepth = Q.shape[0]
    nitem = Q.shape[1]

    e = x0[:nfeature*ndepth]
    E = e.reshape(ndepth, nfeature).transpose()
    Z = np.matmul(np.matmul(X, E), Q)
    P = logistic(Z.reshape(nuser*nitem)).reshape([nuser,nitem])
    PR = P-R
    PR[np.isnan(R)] = 0.0

    mat = np.matmul(np.matmul(X.transpose(),PR),Q.transpose())
    grad = mat.transpose().reshape(nfeature*ndepth)
    grad = grad+la*x0

    return grad


def binlogitSVD_gradQ(x0, E, X, R, la):

    nuser = X.shape[0]
    nfeature = X.shape[1]
    ndepth = E.shape[1]
    nitem = R.shape[1]

    q = x0[: ndepth * nitem]
    Q = q.reshape(ndepth, nitem)
    Z = np.matmul(np.matmul(X, E), Q)
    P = logistic(Z.reshape(nuser*nitem)).reshape([nuser,nitem])
    PR = P-R
    PR[np.isnan(R)] = 0.0

    dZdQ = np.matmul(np.matmul(X,E).transpose(),PR)

    grad = dZdQ.reshape(ndepth*nitem)
    grad = grad+la*x0

    return grad

def binlogitSVD_hesseE(x0, Q, X, R, la):

    nuser = X.shape[0]
    nfeature = X.shape[1]
    ndepth = Q.shape[0]
    nitem = Q.shape[1]

    e = x0[:nfeature*ndepth]
    E = e.reshape(ndepth, nfeature).transpose()
    Z = np.matmul(np.matmul(X, E), Q)
    P = logistic(Z.reshape(nuser*nitem)).reshape([nuser,nitem])
    P2 = P*(1 - P)
    P2[np.isnan(R)] = 0.0

    length = nfeature*ndepth
    hesse = np.zeros([length,length])

    # d2LdEdE
    for d1 in range(0,ndepth):
        for d2 in range(0,ndepth):
            offset_row = nfeature*d1
            offset_col = nfeature*d2
            q1 = Q[d1,:]
            q2 = Q[d2,:]
            v = np.matmul(P2,q1*q2)
            K= (v*X.T).T
            h = np.matmul(X.transpose(),K)
            hesse[offset_row:offset_row+nfeature,offset_col:offset_col+nfeature] = h

    hesse = hesse + la*np.identity(length)

    return hesse

def binlogitSVD_hesseQ(x0, E, X, R, la):

    nuser = X.shape[0]
    nfeature = X.shape[1]
    ndepth = E.shape[1]
    nitem = R.shape[1]

    q = x0[: ndepth*nitem]
    Q = q.reshape(ndepth,nitem)
    Z = np.matmul(np.matmul(X, E), Q)
    P = logistic(Z.reshape(nuser*nitem)).reshape([nuser,nitem])
    P2 = P*(1 - P)
    P2[np.isnan(R)] = 0.0

    hesse = np.zeros([ndepth*nitem,ndepth*nitem])
    for d1 in range(0,ndepth):
        for d2 in range(0,ndepth):
            offset_row = nitem*d1
            offset_col = nitem*d2
            u1 = np.matmul(X,E[:,d1])
            u2 = np.matmul(X,E[:,d2])
            v = np.matmul(P2.transpose(),u1*u2)
            h = np.diag(v)
            hesse[offset_row:offset_row+nitem,offset_col:offset_col+nitem] = h

    hesse = hesse + la * np.identity(nitem*ndepth)

    return hesse


def binlogitSVD_objEQ(x0, X, R, la):

    nuser = R.shape[0]
    nitem = R.shape[1]
    nfeature = X.shape[1]
    ndepth = np.floor((x0.shape[0]+nfeature)/(nfeature+nitem)+0.01).astype(int)

    e = x0[:nfeature*ndepth]
    E = e.reshape(ndepth,nfeature).transpose()
    q = x0[nfeature*ndepth: nfeature*ndepth + ndepth * nitem]
    Q = q.reshape(ndepth, nitem)
    Z = np.matmul(np.matmul(X, E), Q)

    z = Z.reshape(nuser*nitem)
    r = R.reshape(nuser*nitem)

    pidx = (z > 0) & (np.isnan(r) == False)
    nidx = (z <= 0) & (np.isnan(r) == False)
    zr = np.zeros_like(z)
    zr[pidx] = -r[pidx]*np.log(1+np.exp(-z[pidx])) - (1-r[pidx])*z[pidx] -(1-r[pidx])*np.log(1+np.exp(-z[pidx]))
    zr[nidx] = -r[nidx]*np.log(1+np.exp(z[nidx])) + r[nidx]*z[nidx] - (1-r[nidx])*np.log(1+np.exp(z[nidx]))
    logLike = zr[pidx].sum()+zr[nidx].sum()

    enorm = np.linalg.norm(E, ord='fro')
    qnorm = np.linalg.norm(Q, ord='fro')

    return -logLike + 0.5 * la * (enorm * enorm + qnorm * qnorm)


def binlogitSVD_gradEQ(x0, X, R, la):

    nuser = R.shape[0]
    nitem = R.shape[1]
    nfeature = X.shape[1]
    ndepth = np.floor((x0.shape[0]+nfeature)/(nfeature+nitem)+0.01).astype(int)

    e = x0[:nfeature*ndepth]
    E = e.reshape(ndepth, nfeature).transpose()
    q = x0[nfeature*ndepth: nfeature*ndepth + ndepth * nitem]
    Q = q.reshape(ndepth, nitem)

    Z = np.matmul(np.matmul(X, E), Q)
    P = logistic(Z.reshape(nuser*nitem)).reshape([nuser,nitem])
    PR = P-R
    PR[np.isnan(R)] = 0.0

    #dZdE
    dZdE = np.matmul(np.matmul(X.transpose(),PR),Q.transpose())
    grad = dZdE.transpose().reshape(nfeature*ndepth)

    #dZdQ
    dZdQ = np.matmul(np.matmul(X,E).transpose(),PR)
    dZdQ = dZdQ.reshape(ndepth*nitem)

    grad = np.append(grad,dZdQ)
    grad = grad+la*x0

    return grad


def binlogitSVD_hesseEQ(x0, X, R, la):

    nuser = R.shape[0]
    nitem = R.shape[1]
    nfeature = X.shape[1]
    ndepth = np.floor((x0.shape[0]+nfeature)/(nfeature+nitem)+0.01).astype(int)

    e = x0[:nfeature*ndepth]
    E = e.reshape(ndepth, nfeature).transpose()
    q = x0[nfeature*ndepth: nfeature*ndepth + ndepth * nitem]
    Q = q.reshape(ndepth, nitem)

    Z = np.matmul(np.matmul(X, E), Q)
    P = logistic(Z.reshape(nuser*nitem)).reshape([nuser,nitem])
    P2 = P*(1 - P)
    PR = P - R
    PR[np.isnan(R)] = 0.0
    P2[np.isnan(R)] = 0.0

    length = nfeature*ndepth + ndepth*nitem
    hesse = np.zeros([length,length])

    # d2LdEdE
    d2LdEdE = np.zeros([nfeature*ndepth,nfeature*ndepth])
    d2LdQdQ = np.zeros([ndepth*nitem,ndepth*nitem])
    d2LdEdQ = np.zeros([ndepth*nfeature,ndepth*nitem])
    XtPR = np.matmul(X.transpose(), PR)
    for d1 in range(0,ndepth):
        for d2 in range(0,ndepth):
            offset_row = nfeature*d1
            offset_col = nfeature*d2
            q1 = Q[d1,:]
            q2 = Q[d2,:]
            v = np.matmul(P2,q1*q2)
            K= (v*X.T).T
            h = np.matmul(X.transpose(),K)
            d2LdEdE[offset_row:offset_row+nfeature,offset_col:offset_col+nfeature] = h

    # d2LdQdQ
    for d1 in range(0, ndepth):
        for d2 in range(0, ndepth):
            offset_row = nitem*d1
            offset_col = nitem*d2
            u1 = np.matmul(X,E[:,d1])
            u2 = np.matmul(X,E[:,d2])
            v = np.matmul(P2.transpose(),u1*u2)
            h = np.diag(v)
            d2LdQdQ[offset_row:offset_row+nitem,offset_col:offset_col+nitem] = h

    # d2LdEdQ
    for d1 in range(0, ndepth):
        for d2 in range(0, ndepth):
            offset_row = nfeature * d1
            offset_col = nitem * d2
            u2 = np.matmul(X,E[:, d2])
            q1 = Q[d1, :]
            #h = np.matmul(np.matmul(X.transpose(),np.diag(u2)),np.matmul(P2,np.diag(q1)))
            h = np.matmul(u2*X.T,q1*P2)
            if d1 == d2: h = h + XtPR  # same matrix for all d1 = d2
            d2LdEdQ[offset_row:offset_row + nfeature, offset_col:offset_col + nitem] = h

    hesse[0:nfeature*ndepth,0:nfeature*ndepth] = d2LdEdE
    hesse[0:nfeature*ndepth, nfeature*ndepth: nfeature*ndepth + nitem*ndepth]  = d2LdEdQ
    hesse[nfeature*ndepth:nfeature*ndepth + nitem * ndepth, 0: nfeature*ndepth]  = d2LdEdQ.transpose()
    hesse[nfeature*ndepth:nfeature*ndepth + nitem * ndepth, nfeature*ndepth: nfeature*ndepth + nitem*ndepth] = d2LdQdQ

    hesse = hesse + la*np.identity(length)

    return hesse

