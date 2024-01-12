# a. Partial effects at the average 
import numpy as np
import clogit
import estimation as est
import scipy.stats
import pandas as pd

def PEA(x, thetahat, inter: bool):

    # i. Average car characteristics on market i
    avg_car_i = np.mean(x, axis=1) 
    
    # ii. Create average car characteristics for the home market
    avg_car_home = avg_car_i.copy()
    avg_car_home[:, 1] = 1

    # iii. Create average car characteristics for the foreign market
    avg_car_for = avg_car_i.copy()
    avg_car_for[:, 1] = 0

    # iv. If interaction term is included, adjust data points for the home value
    if inter:
        avg_car_home[:, -1] = avg_car_home[:, 0]
        avg_car_for[:, -1] = 0

    # v. Get the dimensions of the input data
    N, J, K = x.shape

    # vi. Create a new array to hold data points with an additional column for the market indicator
    x_new = np.zeros((N, J + 1, K))
    x_new[:, :-1, :] = x.copy()
    
    # vii. Add average car characteristics for the foreign market
    x_new[:, -1, :] = avg_car_for
    ccp_for = clogit.choice_prob(thetahat, x_new)

    # viii. Add average car characteristics for the home market
    x_new[:, -1, :] = avg_car_home
    ccp_home = clogit.choice_prob(thetahat, x_new)

    # ix. Calculate the bias as the mean of the difference between home and foreign market CCPs
    bias = np.mean(ccp_home[:, -1] - ccp_for[:, -1])

    # x. Return the calculated bias
    return bias

# a. Marginal willingness to pay
def MWP(thetahat):
    return abs(thetahat[1]/thetahat[0])

# a. Function to calculate elasticities
def elasticities(thetahat, x, inter, x_vars):
    # i. dimensions of the input data
    N, J, _ = x.shape
    
    # ii. Calculate choice probabilities for the original data
    ccp1 = clogit.choice_prob(thetahat, x)
    
    # iii. Set the log_price index based on whether an interaction term is included
    log_price = 0
    if inter:
        log_price = -1

    # iv. Initialize arrays to store own and cross elasticities
    E_own = np.zeros((N, J))
    E_cross = np.zeros((N, J))

    # v. Loop through each product (car)
    for j in range(J):
        # o. Copy the original data
        x2 = x.copy()

        # oo. Increase price just for car j 
        rel_change_x = 1e-3
        x2[:, j, log_price] += np.log(1.0 + rel_change_x)
        
        # If interaction term is included, adjust the data points for the price change
        if inter:
            for h in range(N):
                if 'home' in x_vars:
                    home_i = x_vars.index('home')
                    if x2[h, j, home_i] == 1:
                        x2[h, j, log_price] += np.log(1.0 + rel_change_x)
                else:
                    x2[h, j, log_price] += np.log(1.0 + rel_change_x)

        # ooo. Evaluate choice probabilities for the modified data
        ccp2 = clogit.choice_prob(thetahat, x2)

        # oooo. Calculate percentage change in choice probabilities 
        rel_change_y = ccp2 / ccp1 - 1.0 

        # ooooo. Calculate elasticities 
        elasticity = rel_change_y / rel_change_x 

        # oooooo. Store own and cross elasticities
        E_own[:, j] = elasticity[:, j]

        # ooooooo. Calculate average cross elasticities for car j
        k_not_j = [k for k in range(J) if k != j]
        E_cross[:, j] = elasticity[:, k_not_j].mean(axis=1)

    # vi. Return own and cross elasticities
    return E_own, E_cross



def result(x, thetahat, cov, print_out: bool, se: bool, inter: bool,N,x_vars):
    # i. Calculate Partial Effects at the Average (PEA)
    pea = PEA(x, thetahat, inter)
    
    # ii. Calculate the Marginal Willingness to Pay (MWP)
    mwp = MWP(thetahat)
    
    # iii. Calculate own and cross price elasticities
    el_h, el_f = elasticities(thetahat, x, inter, x_vars)
    own_price_elasticity = np.mean(el_h).round(4)
    cross_price_elasticity = np.mean(el_f).round(4)

    # iv. Calculate standard errors with delta method
    if se:
        # o. Home bias
        qq0 = lambda theta: PEA(x, theta, inter)
        g0 = est.centered_grad(qq0, thetahat)
        var0 = g0 @ cov @ g0.T
        se_home = np.sqrt(np.diag(var0))

        # oo. Marginal Willingness to Pay (MWP)
        qq1 = lambda theta: MWP(theta)
        g1 = est.centered_grad(qq1, thetahat)
        var1 = g1 @ cov @ g1.T
        se_mwp = np.sqrt(np.diag(var1))

        # ooo. Elasticities - Own
        qq20 = lambda theta: elasticities(theta, x, inter, x_vars)[0]
        g20 = est.centered_grad(qq20, thetahat).mean(axis=0).reshape(1, -1)
        var20 = g20 @ cov @ g20.T
        se_20 = np.sqrt(np.diag(var20))

        # oooo. Elasticities - Cross
        qq21 = lambda theta: elasticities(theta, x, inter, x_vars)[1]
        g21 = est.centered_grad(qq21, thetahat).mean(axis=0).reshape(1, -1)
        var21 = g21 @ cov @ g21.T
        se_21 = np.sqrt(np.diag(var21))

        # ooooo. Calculate t-values
        t_values = (pea / se_home, mwp / se_mwp, own_price_elasticity / se_20, cross_price_elasticity / se_21)

        # oooooo. Calculate p-values
        p_values = 2 * (scipy.stats.t.sf(np.abs(t_values), df=(x.shape[0] - x.shape[1]))).round(4)

        # CI_low = (pea - 1.96 * var0/np.sqrt(N), 
        #           mwp - 1.96 * var1/np.sqrt(N), 
        #           own_price_elasticity - 1.96 * var20/np.sqrt(N),
        #           cross_price_elasticity - 1.96 * var21/np.sqrt(N))
        
        # CI_high = (pea + 1.96 * var0/np.sqrt(N), 
        #            mwp + 1.96 * var1/np.sqrt(N), 
        #            own_price_elasticity + 1.96 * var20/np.sqrt(N),
        #            cross_price_elasticity + 1.96 * var21/np.sqrt(N))

        CI_low = (pea - 1.96 * se_home, 
                  mwp - 1.96 * se_mwp,
                  own_price_elasticity - 1.96 * se_20,
                  cross_price_elasticity - 1.96 * se_21)

        CI_high = (pea + 1.96 * se_home, 
                   mwp + 1.96 * se_mwp, 
                   own_price_elasticity + 1.96 * se_20,
                   cross_price_elasticity + 1.96 * se_21)
        
        # ooooooo. Combine estimates, standard errors, t-values, and p-values
        # data = np.concatenate((np.column_stack((pea, mwp, own_price_elasticity, cross_price_elasticity)),
        #                        np.column_stack((se_home, se_mwp, se_20, se_21)),
        #                        np.column_stack(t_values),
        #                        np.column_stack(p_values)), axis=0)

        data = np.concatenate((np.column_stack((pea, mwp, own_price_elasticity, cross_price_elasticity)),
                               np.column_stack((se_home, se_mwp, se_20, se_21)),
                               np.column_stack(CI_low),
                               np.column_stack(CI_high),
                               np.column_stack(p_values)), axis=0)

    # v. Output the results
    if print_out:
        df = pd.DataFrame(data=data.T, index=['PEA', 'MWP', 'OPE', 'CPE'],
                          columns=['Estimate', 'se', 'CI low', 'CI high', 'p-value'])
        df = df.round(4)
        return df
    else:
        return data