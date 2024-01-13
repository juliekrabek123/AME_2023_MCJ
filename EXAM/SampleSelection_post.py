import numpy as np
from numpy import linalg as la
from scipy.stats import norm
      
# def q(beta,y,x):
#     return - loglik_probit(beta,y,x)

# def loglik_probit(beta, y, x):
#     z = x@beta
#     G = norm.cdf(z)

#     # Make sure that no values are below 0 or above 1.
#     h = np.sqrt(np.finfo(float).eps)
#     G = np.clip(G, h, 1 - h)

#     # Make sure g and y is 1-D array
#     G = G.reshape(-1, )
#     y = y.reshape(-1, )

#     ll = y*np.log(G) + (1 - y)*np.log(1 - G)
#     return ll

# def starting_values(y,x):
#     return la.solve(x.T @ x, x.T @ y )

def starting_values_new(y,x):
    beta = la.solve(x.T @ x, x.T @ y )
    res = y - x@beta
    sigma_2 = res.T @ res / (x.shape[0] - x.shape[1] -1)
    eta = np.log(sigma_2)/2*x.mean()
    theta = np.append(beta,eta)
    return theta

def q2(theta,y,x):
    return - loglik_eta(theta,y,x)

def loglik_eta(theta, y, x):

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

        # Make sure that no values are below 0 or above 1.
        
    # ll = y*np.log(G) + (1 - y)*np.log(1 - G)
    return ll