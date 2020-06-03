# ------------------------------------------------------------------------------
# Example for the usage of logitSVD
#   - artificially created problem
#
# Author: Christian Wagner, May 2020
# ------------------------------------------------------------------------------

import numpy as np

# define size of the problem
nuser = 10000
nitem = 20
nfeature = 10
ndepth = 5   # model parameter


# create feature matrix
np.random.seed(8170)
userFeature = np.random.rand(nuser,nfeature)
userFeature[:, 0] = 1

# create feature embeddings (parameter)
featureEmbed = np.random.rand(nfeature,ndepth)
featureEmbed[2, :] = 10.0
featureEmbed[7, :] = 10.0
featureEmbed[3, :] = 0
featureEmbed[8, :] = 0
for i in range(0,ndepth-2):
    featureEmbed[5,i] = 0
itemEmbed = np.random.rand(ndepth,nitem)
noise = np.random.rand(nuser,nitem)

Z = np.matmul(np.matmul(userFeature,featureEmbed),itemEmbed) - 2.0*noise
quantile = np.quantile(Z.reshape(nuser,nitem),[0,0.2,0.4,0.6,0.8,1.0])

from logitSVDmain import logitSVD
X = userFeature
la = 1e-2

# create binary target user-item-matrix
R = np.zeros([nuser,nitem])
for u in range(0,nuser):
    for i in range(0, nitem):
        if (Z[u, i] >= quantile[2]): R[u,i] = 0
        else: R[u,i] = 1

missing = np.random.rand(nuser,nitem)
R[missing < 0.2] = np.nan

P, C, Z, E, Q, t, z_score, p_value  = logitSVD(X, R, ndepth, la, E=None, Q=None, method="alter",
                                               tol=1e-6, maxit=20, tolNewton = None, maxitNewton = 100, verbose="all")

# create multinomial target user-item-matrix
R = np.zeros([nuser,nitem])
for u in range(0,nuser):
    for i in range(0, nitem):
        for r in range(0,5):
            if (Z[u,i] >= quantile[r]) & (Z[u,i] <= quantile[r+1]): R[u,i] = r

R[missing < 0.2] = np.nan

P, C, Z, E, Q, t, z_score, p_value  = logitSVD(X, R, ndepth, la, E=None, Q=None, method="alter2",
                                               tol=1e-6, maxit=20, maxitNewton = 100, verbose="all")



