# logitSVD
An approach for building models for product recommendations combining the advantages of SVD and logit models

- Predicts the probability to buy a product or to rate the product e.g. with a certain number of stars
- Strong predictive power with a minimal number of parameter to avoid overfitting
- Avoids cold start problem through the use of user features
- Selection of features is supported by a feature importance measure (Wald test)
- Highly transparent model with interpretable parameters 
- Building separate models for each product is a special case of logitSVD 

## Function call
```python
P, C, Z, E, Q, t, z_score, p_value  = logitSVD(X, R, depth, la, E = None, Q = None, t=None, method ="alternating", tol = 1e-4, maxit = 20, tolNewton = None, maxitNewton = 100, verbose = "warn")
```
Parameter
```
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
```
Output
```
P      : binary: ndarray[nuser,nitem], P[u,i] is the probability that R[u,i] = 1
             multinomial: ndarray[max(R),nuser,nitem], P[k,u,i] is the probability that R[u,i] = k
C      : ndarray[nuser,nitem], most likely class - None for binary
Z      : ndarray[nuser,nitem], Z = X E Q
E      : ndarray[nfeature,depth], solution for the parameter E
Q      : ndarray[depth,nitem], solution for the parameter Q
t      : ndarray[max(R)], solution for the intercepts t - None for binary
z-score: ndarray[nfeature,depth], z-score from Wald test for the parameter E
p-value: ndarray[nfeature,depth], p-values from Wald test for the parameter E
```

## Documentation
For a short description see the [description](https://github.com/ChrisDrWagner/logitSVD/logitSVD.pdf).

## Questions and feedback

Please send questions and all kind of feedback to\ 
Christian Wagner\
Goethe Center for Scientific Computing\
Goethe University Frankfurt\
christiandrwagner@gmail.com



