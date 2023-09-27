import numpy as np
from numpy import linalg as la
import scipy.stats 
from tabulate import tabulate


def estimate( 
        y: np.ndarray, x: np.ndarray, transform='', T:int=None, robust = False, sigma2_c:int = 0, sigma2_u:int = 0
    ) -> list:
    """Uses the provided estimator (mostly OLS for now, and therefore we do 
    not need to provide the estimator) to perform a regression of y on x, 
    and provides all other necessary statistics such as standard errors, 
    t-values etc.  

    Args:
        >> y (np.ndarray): Dependent variable (Needs to have shape 2D shape)
        >> x (np.ndarray): Independent variable (Needs to have shape 2D shape)
        >> transform (str, optional): Defaults to ''. If the data is 
        transformed in any way, the following transformations are allowed:
            '': No transformations
            'fd': First-difference
            'be': Between transformation
            'fe': Within transformation
            're': Random effects estimation.
        >>t (int, optional): If panel data, t is the number of time periods in
        the panel, and is used for estimating the variance. Defaults to None.

    Returns:
        list: Returns a dictionary with the following variables:
        'b_hat', 'se', 'sigma2', 't_values', 'R2', 'cov'
    """
    
    b_hat = est_ols(y, x)  # Estimated coefficients
    residual = y - x@b_hat  # Calculated residuals
    SSR = residual.T@residual  # Sum of squared residuals
    SST = (y - np.mean(y)).T@(y - np.mean(y))  # Total sum of squares
    R2 = 1 - SSR/SST

    sigma2, cov, se = variance(transform, SSR, x, T, robust, residual, sigma2_c, sigma2_u)
    t_values = b_hat/se

    p_values = []
    for t_val in t_values:
        p_value = 2 * (scipy.stats.t.sf(np.abs(t_val), df=(x.shape[0] - x.shape[1])))
        p_values.append(p_value)
    p_values = np.array(p_values)

    # Map p-values to asterisk symbols
    def map_p_value_to_asterisks(p_value):
        if p_value <= 0.001:
            return '***'
        elif p_value <= 0.01:
            return '**'
        elif p_value <= 0.05:
            return '*'
        elif p_value <= 0.1:
            return '.'
        else:
            return ''
    
    asterisk_symbols = []
    for p_value in p_values:
        asterisk = map_p_value_to_asterisks(p_value[0])  # Assuming p_value is a 1D array
        asterisk_symbols.append([asterisk])
    asterisk_symbols = np.array(asterisk_symbols)

    p_values_float = [float(p) for p in p_values]

    p_values_ast = [f'{p:.4f} ({asterisk})' for p, asterisk in zip(p_values_float, asterisk_symbols)]

    names = ['b_hat', 'se', 'sigma2', 't_values', 'p_values_ast', 'R2', 'cov', 'ast']
    results = [b_hat, se, sigma2, t_values, p_values_ast, R2, cov, asterisk_symbols]

    return dict(zip(names, results))

    
def est_ols( y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Estimates y on x by ordinary least squares, returns coefficents

    Args:
        >> y (np.ndarray): Dependent variable (Needs to have shape 2D shape)
        >> x (np.ndarray): Independent variable (Needs to have shape 2D shape)

    Returns:
        np.array: Estimated beta coefficients.
    """
    return la.inv(x.T@x)@(x.T@y)

def variance( 
        transform: str, 
        SSR: float, 
        x: np.ndarray, 
        T: int,
        robust: bool, 
        residual: np.ndarray,
        sigma2_c: float,
        sigma2_u: float
    ) -> tuple:
    """Calculates the covariance and standard errors from the OLS
    estimation.

    Args:
        >> transform (str): Defaults to ''. If the data is transformed in 
        any way, the following transformations are allowed:
            '': No transformations
            'fd': First-difference
            'be': Between transformation
            'fe': Within transformation
            're': Random effects estimation
        >> SSR (float): Sum of squared residuals
        >> x (np.ndarray): Dependent variables from regression
        >> t (int): The number of time periods in x.

    Raises:
        Exception: If invalid transformation is provided, returns
        an error.

    Returns:
        tuple: Returns the error variance (mean square error), 
        covariance matrix and standard errors.
    """

    # Store n and k, used for DF adjustments.
    K = x.shape[1]
    if transform in ('', 'fd', 'be'):
        N = x.shape[0]
    else:
        N = x.shape[0]/T

    # Calculate sigma2
    if transform in ('', 'fd', 'be'):
        sigma2 = (np.array(SSR/(N - K)))
    elif transform.lower() == 'fe':
        sigma2 = np.array(SSR/(N * (T - 1) - K))
    elif transform.lower() == 're':
        sigma2 = np.array(SSR/(T * N - K))
    else:
        raise Exception('Invalid transform provided.')
    
    cov = sigma2*la.inv(x.T@x)

    if robust is True:
        if not transform.lower() == 're':
            cov_v_out = 0
            for i in range(int(N)):
                idx_i = slice(i*T, (i+1)*T) # index values for individual i 
                xi = x[idx_i]
                res_outer = residual[idx_i]@residual[idx_i].T
                cov_v_out += xi.T@res_outer@xi
            cov_v = la.inv(x.T@x) @ (cov_v_out) @ la.inv(x.T@x)

        if transform.lower() == 're':
            res_split = np.split(residual,N)
            omega = sigma2_u*np.eye(T) + sigma2_c*np.ones((T, T))
            x_split = np.split(x,N)
            A_out = 0
            B_out = 0
            for i in range(int(N)):
                A_out += x_split[i].T@la.inv(omega)@x_split[i]
                B_out += x_split[i].T@la.inv(omega)@res_split[i]@res_split[i].T@la.inv(omega)@x_split[i]
            cov_v = la.inv(A_out) @ B_out @ la.inv(A_out)      

        cov = cov_v.copy()

    se = np.sqrt(cov.diagonal()).reshape(-1, 1)

    return sigma2, cov, se


def print_table(
        labels: tuple,
        results: dict,
        headers=["", "Beta", "Se", "t-values", "p-values"],
        title="Results",
        _lambda:float=None,
        **kwargs
    ) -> None:
    """Prints a nice looking table, must at least have coefficients, 
    standard errors and t-values. The number of coefficients must be the
    same length as the labels.

    Args:
        >> labels (tuple): Touple with first a label for y, and then a list of 
        labels for x.
        >> results (dict): The results from a regression. Needs to be in a 
        dictionary with at least the following keys:
            'b_hat', 'se', 't_values', 'R2', 'sigma2'
        >> headers (list, optional): Column headers. Defaults to 
        ["", "Beta", "Se", "t-values"].
        >> title (str, optional): Table title. Defaults to "Results".
        _lambda (float, optional): Only used with Random effects. 
        Defaults to None.
    """
    
    # Unpack the labels
    label_y, label_x = labels
    
    # Create table, using the label for x to get a variable's coefficient,
    # standard error and t_value.
    table = []
    for i, name in enumerate(label_x):
        row = [
            name, 
            results.get('b_hat')[i], 
            results.get('se')[i], 
            results.get('t_values')[i],
            results.get('p_values_ast')[i]
        ]
        table.append(row)
    
    # Print the table
    print(title)
    print(f"Dependent variable: {label_y}\n")
    print(tabulate(table, headers, **kwargs))
    
    # Print extra statistics of the model.
    print(f"R\u00b2 = {results.get('R2').item():.3f}")
    print(f"\u03C3\u00b2 = {results.get('sigma2').item():.3f}")
    if _lambda: 
        print(f'\u03bb = {_lambda.item():.3f}')


def perm( Q_T: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Takes a transformation matrix and performs the transformation on 
    the given vector or matrix.

    Args:
        Q_T (np.ndarray): The transformation matrix. Needs to have the same
        dimensions as number of years a person is in the sample.
        
        A (np.ndarray): The vector or matrix that is to be transformed. Has
        to be a 2d array.

    Returns:
        np.array: Returns the transformed vector or matrix.
    """
    # We can infer t from the shape of the transformation matrix.
    M,T = Q_T.shape 
    N = int(A.shape[0]/T)
    K = A.shape[1]

    # initialize output 
    Z = np.empty((M*N, K))
    
    for i in range(N): 
        ii_A = slice(i*T, (i+1)*T)
        ii_Z = slice(i*M, (i+1)*M)
        Z[ii_Z, :] = Q_T @ A[ii_A, :]

    return Z