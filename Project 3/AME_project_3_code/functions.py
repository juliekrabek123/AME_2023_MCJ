# a. Partial effects at the average 
import numpy as np
import clogit
import estimation as est
import scipy.stats
import pandas as pd

def PEA(x, thetahat, inter: bool):

    # a. Average car characteristics on market i
    avg_car_i = np.mean(x, axis=1) 
    
    # b. Create average car characteristics for the home market
    avg_car_home = avg_car_i.copy()
    avg_car_home[:, 1] = 1

    # c. Create average car characteristics for the foreign market
    avg_car_for = avg_car_i.copy()
    avg_car_for[:, 1] = 0

    # d. Get the dimensions of the input data
    N, J, K = x.shape

    # e. Create a new array to hold data points with an additional column for the market indicator
    x_new = np.zeros((N, J + 1, K))
    x_new[:, :-1, :] = x.copy()
    
    # f. Add average car characteristics for the foreign market
    x_new[:, -1, :] = avg_car_for
    ccp_for = clogit.choice_prob(thetahat, x_new)

    # g. Add average car characteristics for the home market
    x_new[:, -1, :] = avg_car_home
    ccp_home = clogit.choice_prob(thetahat, x_new)

    # H. Calculate the bias as the mean of the difference between home and foreign market CCPs
    bias = np.mean(ccp_home[:, -1] - ccp_for[:, -1])

    # x. Return the calculated bias
    return bias

# Marginal willingness to pay
def MWP(thetahat):
    return abs(thetahat[1]/thetahat[0])

# Function to calculate elasticities
def elasticities(thetahat, x, inter, x_vars):
    
    # a. dimensions of the input data
    N, J, _ = x.shape
    
    # b. Calculate choice probabilities for the original data
    ccp1 = clogit.choice_prob(thetahat, x)

    # c. Initialize arrays to store own and cross elasticities
    E_own = np.zeros((N, J))
    E_cross = np.zeros((N, J))

    # d. Loop through each product (car)
    for j in range(J):

        # i. Copy the original data
        x2 = x.copy()

        # ii. Increase price just for car j 
        log_price = 0
        rel_change_x = 1e-2
        x2[:, j, log_price] += np.log(1.0 + rel_change_x)

        # iii. Evaluate choice probabilities for the modified data
        ccp2 = clogit.choice_prob(thetahat, x2)

        # iv. Calculate percentage change in choice probabilities 
        rel_change_y = ccp2 / ccp1 - 1.0 

        # v. Calculate elasticities 
        elasticity = rel_change_y / rel_change_x 

        # vi. Store own and cross elasticities
        E_own[:, j] = elasticity[:, j]

        # vii. Calculate average cross elasticities for car j
        k_not_j = [k for k in range(J) if k != j]
        E_cross[:, j] = elasticity[:, k_not_j].mean(axis=1)

    # e. Calculate average own in home 
    E_home = E_own[x[:,:,1]==1]

    # f. Calculate average own in foreign
    E_foreign = E_own[x[:,:,1]==0]

    # vii. Return own and cross elasticities
    return E_own, E_cross, E_home, E_foreign

def result(x, thetahat, cov, print_out: bool, se: bool, inter: bool,N,x_vars):
    
    N, J, _ = x.shape
    
    # a. Calculate Partial Effects at the Average (PEA)
    pea = PEA(x, thetahat, inter)
    
    # b. Calculate the Marginal Willingness to Pay (MWP)
    mwp = MWP(thetahat)
    
    # c. Calculate own and cross price elasticities
    E_own, E_cross, E_home, E_foreign = elasticities(thetahat, x, inter, x_vars)
    elas_h = np.mean(E_home).round(4)
    elas_f = np.mean(E_foreign).round(4)

    # d. Calculate standard errors with delta method
    if se:
        # i. Home bias
        qq0 = lambda theta: PEA(x, theta, inter)
        g0 = est.centered_grad(qq0, thetahat)
        var0 = g0 @ cov @ g0.T
        se_home = var0 / np.sqrt(N)

        # ii. Marginal Willingness to Pay (MWP)
        qq1 = lambda theta: MWP(theta)
        g1 = est.centered_grad(qq1, thetahat)
        var1 = g1 @ cov @ g1.T
        se_mwp = var1 / np.sqrt(N)
        
        # iii. Elasticities - Own
        qq20 = lambda theta: elasticities(theta, x, inter, x_vars)[0]
        g20 = est.centered_grad(qq20, thetahat).mean(axis=0).reshape(1, -1)
        var20 = g20 @ cov @ g20.T
        se_20 = var20 / np.sqrt(N)

        # iv. Elasticities - Own home
        qq22 = lambda theta: elasticities(theta, x, inter, x_vars)[2]
        g22 = est.centered_grad(qq22, thetahat).mean(axis=0).reshape(1, -1)
        var22 = g22 @ cov @ g22.T
        se_22 = var22 / np.sqrt(N)

        # v. Elasticities - Own foreign
        qq23 = lambda theta: elasticities(theta, x, inter, x_vars)[3]
        g23 = est.centered_grad(qq23, thetahat).mean(axis=0).reshape(1, -1)
        var23 = g23 @ cov @ g23.T
        se_23 = var23 / np.sqrt(N)

        # vi. Calculate t-values
        t_values = (pea / se_home, 
                    mwp / se_mwp, 
                    elas_h / se_22, 
                    elas_f / se_23,
                    )

        # vii. Calculate p-values
        p_values = 2 * (scipy.stats.t.sf(np.abs(t_values), df=(x.shape[0] - x.shape[1]))).round(4)

        # viii. Calculate confidence intervals
        # o. Low 
        CI_low = (pea - 1.96 * se_home, 
                  mwp - 1.96 * se_mwp,
                  elas_h - 1.96 * se_22,
                  elas_f - 1.96 * se_23,
                  )
        # oo. High
        CI_high = (pea + 1.96 * se_home, 
                   mwp + 1.96 * se_mwp, 
                   elas_h + 1.96 * se_22,
                   elas_f + 1.96 * se_23,
                   )
        
        # ix. Combine estimates, standard errors, t-values, and p-values
        data = np.concatenate((np.column_stack((pea, mwp, 
                                                elas_h, elas_f)
                                                ),
                               np.column_stack((se_home, 
                                                se_mwp, 
                                                se_22, se_23)
                                                ),
                               np.column_stack(CI_low),
                               np.column_stack(CI_high),
                               np.column_stack(p_values)), axis=0)

    # e. Output the results
    if print_out:
        df = pd.DataFrame(data=data.T, index=['PEA', 'MWP', 
                                              'E_H', 'E_F',
                                              ],
                          columns=['Estimate', 'se', 'CI low', 'CI high','p-value'])
        df = df.round(4)
        return df
    else:
        return data