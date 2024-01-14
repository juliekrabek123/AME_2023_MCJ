import numpy as np
from numpy import linalg as la
from scipy.stats import norm

def starting_values(y,x):
    beta = la.solve(x.T @ x, x.T @ y )
    res = y - x@beta
    sigma_2 = res.T @ res / (x.shape[0] - x.shape[1] -1)
    eta = np.log(sigma_2)/2*x.mean()
    theta = np.append(beta,eta)
    return theta

def q(theta,y,x):
    return - loglik_probit(theta,y,x)

def loglik_probit(theta, y, x):

    beta = theta[0]
    eta = theta[1]

    ll = np.zeros(y.shape[0])

    for i in range(y.shape[0]):

        z = x[i]*beta/(np.exp(x[i]*eta))
        G = norm.cdf(z)

        h = np.sqrt(np.finfo(float).eps)
        G = np.clip(G, h, 1 - h)

        # Make sure g and y is 1-D array
        G = G.reshape(-1, )
        y = y.reshape(-1, )

        ll[i] = y[i]*np.log(G) + (1 - y[i])*np.log(1 - G)

    return ll