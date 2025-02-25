import code

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import HuberRegressor

from scipy import stats
import statsmodels.api as sm

def robustfit(X, y, wfun='bisquare', tune=4.685, const=True):
    """
    Python implementation of MATLAB's robustfit function.
    
    Parameters:
    -----------
    X : array-like
        Predictor variables (independent variables)
    y : array-like
        Response variable (dependent variable)
    wfun : str, optional
        Weight function for robust fit. Options: 'bisquare' (default), 'huber'
    tune : float, optional
        Tuning constant for the weight function
    const : bool, optional
        Whether to include a constant term (intercept)
        
    Returns:
    --------
    beta : ndarray
        Regression coefficients
    stats : dict
        Dictionary containing regression statistics
    """
    
    # Convert inputs to numpy arrays
    X = np.asarray(X)
    y = np.asarray(y)
    
    # Reshape X if it's 1D
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    # Add constant term if requested
    if const:
        X = sm.add_constant(X)
    
    # Initialize weights
    n = len(y)
    weights = np.ones(n)
    
    # Define weight functions
    def bisquare_weights(r, tune):
        r = np.abs(r)
        w = (1 - (r/tune)**2)**2 * (r < tune)
        return w
    
    def huber_weights(r, tune):
        r = np.abs(r)
        w = np.minimum(1, tune/r)
        return w
    
    # Choose weight function
    if wfun == 'bisquare':
        weight_func = bisquare_weights
    elif wfun == 'huber':
        weight_func = huber_weights
    else:
        raise ValueError("Weight function must be 'bisquare' or 'huber'")
    
    # Iteratively reweighted least squares (IRLS)
    max_iter = 50
    tol = 1e-6
    beta_old = None
    
    for iter in range(max_iter):
        # Weighted least squares
        wls_model = sm.WLS(y, X, weights=weights)
        results = wls_model.fit()
        beta = results.params       # Get coefficients
        
        # Check convergence
        if beta_old is not None and np.all(np.abs(beta - beta_old) < tol):
            break
        
        # Calculate residuals and standardize them
        resid = y - X @ beta
        s = np.median(np.abs(resid - np.median(resid))) / 0.6745
        if s < 1e-6:
            s = 1
        

        W = np.diag(np.sqrt(weights))
        WX = W @ X
        leverage = np.diag(WX @ np.linalg.pinv(WX.T @ WX) @ WX.T)

        # And then in the residuals calculation, use the leverage we just calculated:
        r = resid / (s * np.sqrt(1 - leverage))

        # Update weights
        #r = resid / (s * np.sqrt(1 - results.get_hat_matrix_diag()))
        weights = weight_func(r, tune)
        
        beta_old = beta.copy()
    
    # Calculate final statistics
    #leverage = results.get_hat_matrix_diag()
    W = np.diag(np.sqrt(weights))
    WX = W @ X
    leverage = np.diag(WX @ np.linalg.pinv(WX.T @ WX) @ WX.T)
    s = np.sqrt(np.sum(weights * resid**2) / (np.sum(weights) - X.shape[1]))
    covb = results.cov_params()
    se = np.sqrt(np.diag(covb))
    t = beta / se
    p = 2 * (1 - stats.t.cdf(np.abs(t), n - X.shape[1]))
    
    statistics = {
        'se': se,           # Standard errors
        't_stats': t,       # t-statistics
        'p_values': p,      # p-values
        'resid': resid,     # Residuals
        'weights': weights, # Final weights
        'leverage': leverage,# Leverage values
        's': s,            # Scale estimate
        'covb': covb,      # Covariance matrix
        'iterations': iter + 1  # Number of iterations
    }
    
    return beta, statistics



def recover_dependency(slope_to_recover, spine_activity):
    '''
    Recover the 'coupling' between the two data series.
    '''
    L = 5000
    den = np.concatenate([np.random.poisson(lam=1, size=L) + np.random.random(size=L), np.random.poisson(lam=0, size=L) + np.random.random(size=L)])
    
    sp = np.random.poisson(lam=1, size=L*2) + np.random.random(size=L*2)
    
    sp = spine_activity*sp + slope_to_recover*den

    # Get coefficients w/ quick method
    robust_reg = HuberRegressor()
    robust_reg.fit(den.reshape(-1,1), sp)

    #intercept = robust_reg.intercept_
    quick_slope = robust_reg.coef_[0]

    # Get coefficients w/ iterative method from Claude
    beta_bisquare, stats = robustfit(den, sp, wfun='bisquare', tune=4.685, const=True)
    beta_huber   , stats = robustfit(den, sp, wfun='huber', tune=4.685, const=True)
    return (quick_slope, beta_bisquare, beta_huber)


if __name__ =='__main__':
    # Data observation
    spine_mults = np.array([1, 2, 5, 8, 10, 12, 15, 20])
    true_slopes = np.linspace(0,1)
    recovered_slopes = np.empty((50,3,len(spine_mults)))
    for iter in range(len(spine_mults)):
        recovered_slopes[:,:,iter] = np.load(f'recovered_slopes_multiple-{spine_mults[iter]}.npy')

    # Difficulty of the recovery is a function of slope, since more of the variability of the
    #   data comes from the randomness of the spine with increasing multiples.
    # Three different methods as well.
    for iter in range(len(spine_mults)):
        sklearn_huber_er = np.power((true_slopes - recovered_slopes[:,0,iter]), 2)
        iter_bisquare_er = np.power((true_slopes - recovered_slopes[:,1,iter]), 2)
        iter_huber_er    = np.power((true_slopes - recovered_slopes[:,2,iter]), 2)

        sklearn_huber_er_mu = np.mean(sklearn_huber_er)
        iter_bisuqare_er_mu = np.mean(iter_bisquare_er)
        iter_huber_er_mu    = np.mean(iter_huber_er)

        

        fig, axs = plt.subplots(2,1, figsize=(12,8))
        axs[0].plot(true_slopes, recovered_slopes[:,0,iter], label='sklearn-huber')
        axs[0].plot(true_slopes, recovered_slopes[:,1,iter], label='iter-bisquare')
        axs[0].plot(true_slopes, recovered_slopes[:,2,iter], label='iter-huber')

        axs[0].plot(true_slopes, true_slopes, linestyle=':', color='black')

        axs[0].set_aspect(0.5)
        axs[0].set_ylim([-0.25, 1.25])
        axs[0].set_xlabel('True slope values')
        axs[0].set_ylabel('Recovered slope values')
        axs[0].legend()
        axs[0].set_title('Slopes to recovered slopes')

        axs[1].plot(true_slopes, np.abs(true_slopes - recovered_slopes[:,0,iter]), label='sklearn-huber')
        axs[1].plot(true_slopes, np.abs(true_slopes - recovered_slopes[:,1,iter]), label='iter-bisquare')
        axs[1].plot(true_slopes, np.abs(true_slopes - recovered_slopes[:,2,iter]),    label='iter-huber')
        #axs[1].set_aspect('equal', adjustable='box')\

        axs[1].axhline(y=0, linestyle=':', color='black')

        axs[1].set_ylim([-0.05, 0.30])
        axs[1].set_xlabel('True slope values')
        axs[1].set_ylabel('Abs error per slope')
        axs[1].legend()
        axs[1].set_title('Error')

        fig.suptitle(f'sklearn mu-sq er:{sklearn_huber_er_mu} \n iter-bisquare mu-sq er:{iter_bisuqare_er_mu} \n iter-huber mu-sq er:{iter_huber_er_mu}')
        fig.tight_layout()
        plt.savefig(f'method_comp_spinemult-{spine_mults[iter]}.png')
    code.interact(local=dict(globals(), **locals())) 
    
    # Data generation
    '''
    true_slopes = np.linspace(0,1)
    np.save(f'true_slopes.npy', true_slopes)
    
    recovered_slopes = np.zeros((true_slopes.size, 3))
    spine_multiples = [1, 2, 5, 8, 10, 12, 15, 20]
    
    for sp_val in spine_multiples:
        for iter in range(len(true_slopes)):
            print(f"Spine : {sp_val}, iter : {iter}")
            quick_slope, beta_bisquare, beta_huber = recover_dependency(true_slopes[iter], spine_activity=sp_val)
            recovered_slopes[iter,0] = quick_slope
            recovered_slopes[iter,1] = beta_bisquare[1]
            recovered_slopes[iter,2] = beta_huber[1]
        np.save(f'recovered_slopes_multiple-{sp_val}.npy', recovered_slopes)
    '''
        
    



