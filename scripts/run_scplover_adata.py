import numpy as np
from scipy.special import logsumexp
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import anndata as ad
import click
from datetime import datetime
import pandas as pd
from scipy.special import logsumexp, gammaln
import numpy as np
import scipy
import scipy.interpolate
import statsmodels.formula.api as smf
import warnings
from scipy.stats.mstats import mquantiles
from statsmodels.nonparametric.smoothers_lowess import lowess


offset_mult = [0.0, 5.15912268, 1.61006676, 1.37280321, 1.29739684,
       1.25003311, 1.21287765, 1.18196752, 1.15610803, 1.13485567,
       1.10586721, 1.08814929, 1.07609967, 1.06808739, 1.06300076,
       1.06005268, 1.0586684 , 1.05841737, 1.05897012, 1.06006985,
       1.06151322, 1.06313698, 1.0648085 , 1.06641881, 1.06787753,
       1.06910905, 1.07004961, 1.07064503, 1.070849  , 1.07062167]

offset_mult_spectrum = offset_mult.copy()
offset_mult_spectrum[1] = 630 # SPECTRUM data has unusually many overlap bases at state=1 (consistently 500-1000)

class ConstrainedGaussianHMM:
    def __init__(self, means, transition_matrix, covariance_type='full', 
                 max_iter=100, tol=1e-3, reg_covar=1e-3, n_jobs=1,
                 fix_means=True, fix_transitions=True, 
                 learn_mean_scaling=False, init_mean_scales=None,
                 mean_scale_bounds=None):
        """
        Flexible Gaussian HMM with optional parameter fixing and mean scaling.
        
        Parameters:
        -----------
        means : array-like, shape (n_states, n_features)
            Initial means for each state
        transition_matrix : array-like, shape (n_states, n_states)
            Initial transition probabilities (rows sum to 1)
        covariance_type : str
            Type of covariance parameters ('full', 'diag', 'spherical')
        n_jobs : int
            Number of parallel jobs (-1 for all cores, 1 for sequential)
        fix_means : bool
            If True, means stay fixed. If False, means are learned freely.
        fix_transitions : bool
            If True, transition matrix stays fixed. If False, transitions are learned.
        learn_mean_scaling : bool
            If True, learns scale coefficients per dimension: mean_i = scale[d] * initial_mean_i[d]
            This is mutually exclusive with fix_means=False.
        init_mean_scales : array-like, shape (n_features,), optional
            Initial scale coefficients. If None, defaults to ones.
        mean_scale_bounds : tuple or array-like, optional
            Bounds for mean_scales_. Can be:
            - tuple (lower, upper): same bounds for all dimensions
            - array of shape (n_features, 2): per-dimension bounds
            Example: (0.5, 1.5) constrains all scales to [0.5, 1.5]
                     [[0.5, 1.5], [0.8, 1.2]] different bounds per dimension
        """
        self.initial_means = np.array(means)  # Store initial means
        self.means_ = np.array(means).copy()
        self.n_states = len(means)
        self.n_features = means.shape[1]
        
        self.transition_matrix = np.array(transition_matrix)
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.n_jobs = n_jobs if n_jobs != -1 else mp.cpu_count()
        
        # Parameter control
        self.fix_means = fix_means
        self.fix_transitions = fix_transitions
        self.learn_mean_scaling = learn_mean_scaling
        
        # Validate mutual exclusivity
        if not fix_means and learn_mean_scaling:
            raise ValueError("Cannot have fix_means=False and learn_mean_scaling=True simultaneously. "
                           "Choose one: free means or scaled means.")
        
        # Initialize mean scales
        if init_mean_scales is None:
            self.mean_scales_ = np.ones(self.n_features)
        else:
            self.mean_scales_ = np.array(init_mean_scales)
            if self.mean_scales_.shape[0] != self.n_features:
                raise ValueError(f"init_mean_scales must have length {self.n_features}")
        
        # Set up scale bounds
        self.mean_scale_bounds = self._setup_scale_bounds(mean_scale_bounds)
        
        # Apply bounds to initial scales
        if self.learn_mean_scaling and self.mean_scale_bounds is not None:
            self.mean_scales_ = self._clip_scales(self.mean_scales_)
        

        # Apply initial scaling
        if self.learn_mean_scaling:
            self.means_ = self.initial_means * self.mean_scales_[np.newaxis, :]
        
        # Validate
        if self.transition_matrix.shape != (self.n_states, self.n_states):
            raise ValueError(f"Transition matrix must be {self.n_states}x{self.n_states}")
        if not np.allclose(self.transition_matrix.sum(axis=1), 1.0):
            raise ValueError("Transition matrix rows must sum to 1")
        
        self.log_transmat = np.log(self.transition_matrix + 1e-10)
        self.covariances_ = None
        self.startprob_ = None
        
        # Precompute for faster emission probability computation
        self._chol_factors = None
        self._log_dets = None
        
    def _setup_scale_bounds(self, bounds):
            """
            Setup scale bounds in a consistent format.
            
            Returns array of shape (n_features, 2) where [:, 0] is lower, [:, 1] is upper
            """
            if bounds is None:
                return None
            
            bounds = np.array(bounds)
            
            # Case 1: Single tuple (lower, upper) for all dimensions
            if bounds.shape == (2,):
                lower, upper = bounds
                if lower > upper:
                    raise ValueError("Lower bound must be <= upper bound")
                return np.array([[lower, upper]] * self.n_features)
            
            # Case 2: Per-dimension bounds
            elif bounds.shape == (self.n_features, 2):
                for i in range(self.n_features):
                    if bounds[i, 0] > bounds[i, 1]:
                        raise ValueError(f"Lower bound must be <= upper bound for dimension {i}")
                return bounds
            
            else:
                raise ValueError(f"mean_scale_bounds must be (2,) or ({self.n_features}, 2), got {bounds.shape}")


    def _clip_scales(self, scales):
        """Clip scales to respect bounds"""
        if self.mean_scale_bounds is None:
            return scales
        
        clipped = scales.copy()
        for d in range(self.n_features):
            lower, upper = self.mean_scale_bounds[d]
            clipped[d] = np.clip(scales[d], lower, upper)
        
        return clipped
    
    def _project_scales_to_bounds(self, scales):
        """
        Project scales to bounds using a soft constraint.
        Uses a sigmoid-like transformation to smoothly enforce bounds.
        """
        if self.mean_scale_bounds is None:
            return scales
        
        projected = np.zeros(self.n_features)
        for d in range(self.n_features):
            lower, upper = self.mean_scale_bounds[d]
            
            # If already in bounds, keep it
            if lower <= scales[d] <= upper:
                projected[d] = scales[d]
            else:
                # Project to nearest boundary
                projected[d] = np.clip(scales[d], lower, upper)
        
        return projected
    

    def _precompute_covariance_factors(self):
        """Precompute Cholesky factors and log determinants for speed"""
        self._chol_factors = []
        self._log_dets = []
        
        for state in range(self.n_states):
            if self.covariance_type == 'full':
                cov = self._ensure_positive_definite(self.covariances_[state])
            elif self.covariance_type == 'diag':
                cov = np.diag(self.covariances_[state])
            elif self.covariance_type == 'spherical':
                cov = self.covariances_[state] * np.eye(self.n_features)
            
            try:
                L = np.linalg.cholesky(cov)
                self._chol_factors.append(L)
                self._log_dets.append(2 * np.sum(np.log(np.diag(L))))
            except np.linalg.LinAlgError:
                # Fallback
                U, s, Vt = np.linalg.svd(cov)
                s = np.maximum(s, self.reg_covar)
                self._chol_factors.append((Vt.T / np.sqrt(s)).T)
                self._log_dets.append(np.sum(np.log(s)))
        
    def _initialize_parameters(self, X_list):
        """Initialize covariances and start probabilities"""
        X_concat = np.concatenate(X_list, axis=0)
        
        if self.covariance_type == 'full':
            base_cov = np.cov(X_concat.T) + self.reg_covar * np.eye(self.n_features)
            self.covariances_ = np.array([base_cov.copy() for _ in range(self.n_states)])
        elif self.covariance_type == 'diag':
            base_var = np.var(X_concat, axis=0) + self.reg_covar
            self.covariances_ = np.array([base_var.copy() for _ in range(self.n_states)])
        elif self.covariance_type == 'spherical':
            base_var = np.var(X_concat) + self.reg_covar
            self.covariances_ = np.array([base_var for _ in range(self.n_states)])
        
        self.startprob_ = np.ones(self.n_states) / self.n_states
        self.log_startprob_ = np.log(self.startprob_)
        
        # Precompute factors
        self._precompute_covariance_factors()
    
    def _ensure_positive_definite(self, cov):
        """Ensure covariance matrix is positive definite"""
        cov_reg = cov + self.reg_covar * np.eye(cov.shape[0])
        try:
            np.linalg.cholesky(cov_reg)
            return cov_reg
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(cov_reg)
            eigvals = np.maximum(eigvals, self.reg_covar)
            return eigvecs @ np.diag(eigvals) @ eigvecs.T
    
    def _compute_log_emission_prob_vectorized(self, X):
        """Vectorized computation of log emission probabilities"""
        n_samples = X.shape[0]
        log_prob = np.zeros((n_samples, self.n_states))
        
        for state in range(self.n_states):
            diff = X - self.means_[state]
            L = self._chol_factors[state]
            
            # Solve L * y = diff.T for y
            y = np.linalg.solve(L, diff.T)
            mahal = np.sum(y**2, axis=0)
            
            log_prob[:, state] = -0.5 * (
                self.n_features * np.log(2 * np.pi) + 
                self._log_dets[state] + 
                mahal
            )
        
        return log_prob
    
    def _forward_vectorized(self, log_emission_prob):
        """Optimized forward algorithm"""
        n_samples = log_emission_prob.shape[0]
        log_alpha = np.zeros((n_samples, self.n_states))
        
        # Initialize
        log_alpha[0] = self.log_startprob_ + log_emission_prob[0]
        
        # Vectorized recursion
        for t in range(1, n_samples):
            log_alpha[t] = logsumexp(
                log_alpha[t-1, :, None] + self.log_transmat, 
                axis=0
            ) + log_emission_prob[t]
        
        return log_alpha
    
    def _backward_vectorized(self, log_emission_prob):
        """Optimized backward algorithm"""
        n_samples = log_emission_prob.shape[0]
        log_beta = np.zeros((n_samples, self.n_states))
        
        # Vectorized recursion (backward)
        for t in range(n_samples - 2, -1, -1):
            log_beta[t] = logsumexp(
                self.log_transmat + log_emission_prob[t+1] + log_beta[t+1],
                axis=1
            )
        
        return log_beta
    
    def _compute_posteriors(self, log_alpha, log_beta):
        """Compute state posteriors"""
        log_gamma = log_alpha + log_beta
        log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
        return np.exp(log_gamma)
    
    def _compute_xi_vectorized(self, log_alpha, log_beta, log_emission_prob):
        """Vectorized computation of xi"""
        n_samples = log_emission_prob.shape[0]
        
        # Vectorized computation
        log_xi = (
            log_alpha[:-1, :, None] +
            self.log_transmat[None, :, :] +
            log_emission_prob[1:, None, :] +
            log_beta[1:, None, :]
        )
        
        # Normalize
        log_xi -= logsumexp(log_xi.reshape(n_samples - 1, -1), axis=1)[:, None, None]
        
        return np.exp(log_xi)
    
    def _process_sequence(self, X):
        """Process a single sequence (for parallel execution)"""
        log_emission_prob = self._compute_log_emission_prob_vectorized(X)
        log_alpha = self._forward_vectorized(log_emission_prob)
        log_beta = self._backward_vectorized(log_emission_prob)
        
        gamma = self._compute_posteriors(log_alpha, log_beta)
        xi = self._compute_xi_vectorized(log_alpha, log_beta, log_emission_prob)
        
        log_likelihood = logsumexp(log_alpha[-1])
        
        return gamma, xi, log_likelihood
    
    def _e_step(self, X_list):
        """Parallel E-step"""
        if self.n_jobs == 1 or len(X_list) == 1:
            results = [self._process_sequence(X) for X in X_list]
        else:
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                results = list(executor.map(self._process_sequence, X_list))
        
        all_gamma = [r[0] for r in results]
        all_xi = [r[1] for r in results]
        total_log_likelihood = sum(r[2] for r in results)
        
        return all_gamma, all_xi, total_log_likelihood
    
    def _m_step(self, X_list, all_gamma, all_xi):
        """Optimized M-step with optional parameter updates"""
        # Update start probabilities
        gamma_starts = np.array([gamma[0] for gamma in all_gamma])
        self.startprob_ = gamma_starts.mean(axis=0)
        self.startprob_ /= self.startprob_.sum()
        self.log_startprob_ = np.log(self.startprob_ + 1e-10)
        
        # Update transition matrix (only if not fixed)
        if not self.fix_transitions:
            xi_sum = np.sum([xi.sum(axis=0) for xi in all_xi], axis=0)
            self.transition_matrix = xi_sum / (xi_sum.sum(axis=1, keepdims=True) + 1e-10)
            self.log_transmat = np.log(self.transition_matrix + 1e-10)
        
        # Concatenate data for vectorized updates
        X_concat = np.concatenate(X_list, axis=0)
        gamma_concat = np.concatenate(all_gamma, axis=0)
        
        # Update means or mean scales
        if self.learn_mean_scaling:
            # Learn scaling coefficients with bounds
            for d in range(self.n_features):
                numerator = 0.0
                denominator = 0.0
                
                for state in range(self.n_states):
                    gamma_state = gamma_concat[:, state]
                    gamma_sum = gamma_state.sum() + 1e-10
                    
                    obs_mean_d = np.sum(gamma_state * X_concat[:, d]) / gamma_sum
                    initial_mean_d = self.initial_means[state, d]
                    
                    numerator += gamma_sum * obs_mean_d * initial_mean_d
                    denominator += gamma_sum * (initial_mean_d ** 2)
                
                if abs(denominator) > 1e-10:
                    # Compute unconstrained update
                    new_scale = numerator / denominator
                    new_scale = np.maximum(new_scale, 1e-6)  # Keep positive
                    
                    # Apply bounds
                    if self.mean_scale_bounds is not None:
                        lower, upper = self.mean_scale_bounds[d]
                        new_scale = np.clip(new_scale, lower, upper)
                    
                    self.mean_scales_[d] = new_scale
                else:
                    # If no information, stay at current value (respecting bounds)
                    if self.mean_scale_bounds is not None:
                        lower, upper = self.mean_scale_bounds[d]
                        self.mean_scales_[d] = np.clip(self.mean_scales_[d], lower, upper)
            
            # Apply scaling to get updated means
            self.means_ = self.initial_means * self.mean_scales_[np.newaxis, :]
            
        elif not self.fix_means:
            # Learn means freely
            for state in range(self.n_states):
                gamma_state = gamma_concat[:, state]
                gamma_sum = gamma_state.sum() + 1e-10
                self.means_[state] = np.sum(gamma_state[:, None] * X_concat, axis=0) / gamma_sum
        
        # Always update covariances
        for state in range(self.n_states):
            gamma_state = gamma_concat[:, state]
            gamma_sum = gamma_state.sum() + 1e-10
            
            diff = X_concat - self.means_[state]
            
            if self.covariance_type == 'full':
                weighted_diff = gamma_state[:, None] * diff
                self.covariances_[state] = (weighted_diff.T @ diff) / gamma_sum
                self.covariances_[state] += self.reg_covar * np.eye(self.n_features)
                self.covariances_[state] = self._ensure_positive_definite(self.covariances_[state])
                
            elif self.covariance_type == 'diag':
                self.covariances_[state] = np.sum(gamma_state[:, None] * diff**2, axis=0) / gamma_sum
                self.covariances_[state] = np.maximum(self.covariances_[state], self.reg_covar)
                
            elif self.covariance_type == 'spherical':
                self.covariances_[state] = np.sum(gamma_state[:, None] * diff**2) / (gamma_sum * self.n_features)
                self.covariances_[state] = max(self.covariances_[state], self.reg_covar)
        
        # Recompute factors after update
        self._precompute_covariance_factors()
    
    def fit(self, X_list, verbose=False):
        """Fit the HMM"""
        if not isinstance(X_list, list):
            X_list = [X_list]
        
        self._initialize_parameters(X_list)
        
        log_likelihood_old = -np.inf
        self.converged_ = False
        
        if verbose:
            print(f"Training HMM with:")
            if self.learn_mean_scaling:
                print(f"  - Means: SCALED (learning {self.n_features} scale coefficients)")
            else:
                print(f"  - Means: {'FIXED' if self.fix_means else 'LEARNABLE'}")
            print(f"  - Transitions: {'FIXED' if self.fix_transitions else 'LEARNABLE'}")
            print(f"  - Covariances: LEARNABLE")
        
        for iteration in range(self.max_iter):
            try:
                all_gamma, all_xi, log_likelihood = self._e_step(X_list)
                self._m_step(X_list, all_gamma, all_xi)
                
                if verbose and iteration % 10 == 0:
                    print(f"Iteration {iteration}: Log-likelihood = {log_likelihood:.4f}")
                    if self.learn_mean_scaling:
                        print(f"  Current scales: {self.mean_scales_}")
                
                if np.isfinite(log_likelihood) and abs(log_likelihood - log_likelihood_old) < self.tol:
                    if verbose:
                        print(f"Converged at iteration {iteration}")
                    self.converged_ = True
                    break
                
                log_likelihood_old = log_likelihood
                
            except Exception as e:
                if verbose:
                    print(f"Error at iteration {iteration}: {e}")
                    import traceback
                    traceback.print_exc()
                break
        
        if not self.converged_ and verbose:
            print(f"Did not converge after {self.max_iter} iterations")
        
        return self
    
    def _viterbi_single(self, X):
        """Viterbi algorithm for single sequence"""
        n_samples = X.shape[0]
        log_emission_prob = self._compute_log_emission_prob_vectorized(X)
        
        log_delta = np.zeros((n_samples, self.n_states))
        psi = np.zeros((n_samples, self.n_states), dtype=int)
        
        # Initialize
        log_delta[0] = self.log_startprob_ + log_emission_prob[0]
        
        # Vectorized recursion
        for t in range(1, n_samples):
            temp = log_delta[t-1, :, None] + self.log_transmat
            psi[t] = np.argmax(temp, axis=0)
            log_delta[t] = temp[psi[t], np.arange(self.n_states)] + log_emission_prob[t]
        
        # Backtrack
        state_sequence = np.zeros(n_samples, dtype=int)
        state_sequence[-1] = np.argmax(log_delta[-1])
        
        for t in range(n_samples - 2, -1, -1):
            state_sequence[t] = psi[t+1, state_sequence[t+1]]
        
        return state_sequence
    
    def predict(self, X):
        """Predict state sequences"""
        if isinstance(X, list):
            if self.n_jobs == 1 or len(X) == 1:
                return [self._viterbi_single(x) for x in X]
            else:
                with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                    return list(executor.map(self._viterbi_single, X))
        else:
            return self._viterbi_single(np.array(X))
    
    def score(self, X):
        """Compute log-likelihood"""
        if isinstance(X, list):
            def score_single(x):
                log_emission_prob = self._compute_log_emission_prob_vectorized(x)
                log_alpha = self._forward_vectorized(log_emission_prob)
                return logsumexp(log_alpha[-1])
            
            if self.n_jobs == 1 or len(X) == 1:
                return np.array([score_single(x) for x in X])
            else:
                with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                    return np.array(list(executor.map(score_single, X)))
        else:
            X = np.array(X)
            log_emission_prob = self._compute_log_emission_prob_vectorized(X)
            log_alpha = self._forward_vectorized(log_emission_prob)
            return logsumexp(log_alpha[-1])
    
    def predict_proba(self, X):
        """Compute state probabilities"""
        def proba_single(x):
            log_emission_prob = self._compute_log_emission_prob_vectorized(x)
            log_alpha = self._forward_vectorized(log_emission_prob)
            log_beta = self._backward_vectorized(log_emission_prob)
            return self._compute_posteriors(log_alpha, log_beta)
        
        if isinstance(X, list):
            if self.n_jobs == 1 or len(X) == 1:
                return [proba_single(x) for x in X]
            else:
                with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                    return list(executor.map(proba_single, X))
        else:
            return proba_single(np.array(X))


def valid(df, field_name='reads'):
    """adds valid column (calls with atleast one reads and non negative gc)

    :params df: pandas dataframe
    """

    df.loc[:, "valid"] = True

    df.loc[(df[field_name] <= 0) | (df['gc'] < 0), "valid"] = False

    return df

def ideal(df, field_name='reads'):
    """adds ideal column

    :params df: pandas dataframe
    """
    df.loc[:, "ideal"] = True

    valid_reads = df[df["valid"]][field_name]
    valid_gc = df[df["valid"]]["gc"]

    routlier = 0.01
    doutlier = 0.001

    range_l, range_h = mquantiles(valid_reads, prob=[0, 1 - routlier],
                                  alphap=1, betap=1)
    domain_l, domain_h = mquantiles(valid_gc, prob=[doutlier, 1 - doutlier],
                                    alphap=1, betap=1)

    df.loc[(df["valid"] == False) |
           (df[field_name] <= range_l) |
           (df[field_name] > range_h) |
           (df["gc"] < domain_l) |
           (df["gc"] > domain_h),
           "ideal"] = False

    return df
    
def modal_quantile_regression(df_regression, lowess_frac=0.2, degree=2, knots=[0.38], field='reads'):
    '''
    Fits a B-spline polynomial curve through the "modal" quantile of the data:
    * Runs quantile regression to fit a B-spline curve for each percentile 10-90
    * Estimates the modal quantile as the quantile where difference in AUC is minimized
    * Uses the curve fit to this modal quantile for normalization

    Parameters:
        df_regression: pandas.DataFrame with at least columns [chr, start, end, reads, gc]
        lowess_frac: float, fraction of data used to estimate each y-value in Lowess smoothing of AUC curve
        degree: int, degree of polynomial to fit to each section of the B-spline curve
        knots: list of floats, GC values where B-spline polynomial is allowed to change

    Returns:
        pandas.DataFrame with additional columns
            modal_curve: modal curve's predicted # reads for GC value in this row
            modal_quantile: quantile selected as the mode (should be the same for all bins)
            modal_corrected: corrected read count (i.e., reads / modal_curve)
    '''
    np.random.seed(0)

    q_range = range(10, 91, 1)
    quantiles = np.array(q_range) / 100
    quantile_names = [str(x) for x in q_range]

    # need at least 3 values to compute the quantiles
    if len(df_regression) < 10 or sum(df_regression[field]) < 100:
        df_regression['modal_quantile'] = None
        df_regression['modal_curve'] = None
        df_regression['modal_corrected'] = None
        return df_regression, None
    
    if knots[0] < df_regression['gc'].min():
        knots[0] = df_regression['gc'].quantile(0.2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        poly_quantile_model = smf.quantreg(f'{field} ~ bs(gc, degree={degree}, knots={knots}, include_intercept = True)',
                                        data=df_regression)
        poly_quantile_fit = [poly_quantile_model.fit(q=q) for q in quantiles]


    poly_quantile_predict = [poly_quantile_fit[i].predict(df_regression) for i in range(len(quantiles))]

    poly_quantile_params = pd.DataFrame()

    for i in range(len(quantiles)):
        df_regression[quantile_names[i]] = poly_quantile_predict[i]
        poly_quantile_params[quantile_names[i]] = poly_quantile_fit[i].params

    # integration and mode selection

    gc_min = df_regression['gc'].quantile(q=0.10)
    gc_max = df_regression['gc'].quantile(q=0.90)

    true_min = df_regression['gc'].min()
    true_max = df_regression['gc'].max()

    poly_quantile_integration = np.zeros(len(quantiles) + 1)

    # form (k+1)-regular knot vector
    repeats = degree + 1
    my_t = np.r_[[true_min] * repeats, knots, [true_max] * repeats]
    for i in range(len(quantiles)):
        # compose params into piecewise polynomial
        params = poly_quantile_params[quantile_names[i]].to_numpy()
        pp = scipy.interpolate.PPoly.from_spline((my_t, params[1:] + params[0], degree))

        # compute integral
        poly_quantile_integration[i + 1] = pp.integrate(gc_min, gc_max)

        # find the modal quantile
    distances = poly_quantile_integration[1:] - poly_quantile_integration[:-1]

    df_dist = pd.DataFrame({'quantiles': quantiles, 'quantile_names': quantile_names, 'distances': distances})
    dist_max = df_dist['distances'].quantile(q=0.95)
    df_dist_filter = df_dist[df_dist['distances'] < dist_max].copy()
    df_dist_filter['lowess'] = lowess(df_dist_filter['distances'], df_dist_filter['quantiles'], frac=lowess_frac,
                                      return_sorted=False)

    modal_quantile = df_dist_filter.set_index('quantile_names')['lowess'].idxmin()

    # add values to table

    df_regression['modal_quantile'] = modal_quantile
    df_regression['modal_curve'] = df_regression[modal_quantile]
    df_regression['modal_corrected'] = df_regression[field] / df_regression[modal_quantile]

    modal_params = poly_quantile_params[modal_quantile].to_numpy()
    modal_polynomial = scipy.interpolate.PPoly.from_spline((my_t, modal_params[1:] + modal_params[0], degree))
    return df_regression, modal_polynomial

def fit_cell_restrict_states(regdf, max_k=12, means='fixed',
                    bin_length=5e5, return_all=False, verbose=False, covariance_type='full',
                    fit_transitions=False, min_mean_scale=0, max_mean_scale=np.inf, is_spectrum=False):

    if covariance_type not in ['full', 'diag', 'spherical']:
        raise ValueError(f"Expected 'full', 'diag', or 'spherical' for argument 'covariance_type', got: {covariance_type}")

    if means == 'fixed':
        fix_means = True
        learn_mean_scaling = False
    elif means == 'scale':
        fix_means=True
        learn_mean_scaling = True
    elif means == 'free':
        fix_means= False
        learn_mean_scaling = False
    else:
        raise ValueError(f"Expected 'fixed', 'scale', or 'free' for argument 'means', got: {means}")
        
    cell_id = regdf['cell_id'].iloc[0]
    k_range = np.arange(max_k + 1)
    total_reads = regdf['reads'].sum()
    total_fragments = regdf['n_fragments'].sum()
    ell = (regdf['mean_fragment_length'] * regdf['n_fragments']).sum() / regdf['n_fragments'].sum()
     

    regdf['chr'] = regdf.index.str.split(':', expand=True).get_level_values(0)
    regdf['start'] = regdf.index.str.split(':', expand=True).get_level_values(1).str.split('-', expand=True).get_level_values(0).astype(int)
    Xs = [df.sort_values(by='start')[['reads', 'overlap_bases']].to_numpy() for _, df in regdf.groupby('chr')]
    index = pd.concat([df.sort_values(by='start')[[]] for _, df in regdf.groupby('chr')]).index


    median_state = int(regdf['state'].median().round())
    multipliers = [m for m in [1] + [i for i in range(2, max(3, min(6, int(max_k // median_state))))]]
    multipliers += [1/m for m in range(2, min(6, median_state + 1))]
    assert len(multipliers) == len(np.unique(multipliers))
    if is_spectrum:
        scale_reads = False
        my_offset_mult = offset_mult_spectrum.copy()
        state1_bins = regdf[regdf['state'] == 1]
        if len(state1_bins) == 0:
            my_offset_mult[1] = (total_fragments**2)/1.2e10
            print(f'cell {regdf["cell_id"].iloc[0]} has no state 1 bases, using calculated offset {my_offset_mult[1]}')
        else:
            my_offset_mult[1] = state1_bins['overlap_bases'].mean()

            print(f'cell {regdf["cell_id"].iloc[0]} has mean overlap bases {my_offset_mult[1]} in state 1')
        # don't consider increment parameter sets -- high confidence in SPECTRUM CN calls
    else:
        scale_reads = True
        my_offset_mult = offset_mult

    cell_results = []
    for value in multipliers:
        initial_ploidy = regdf['state'].mean() * value
        allowed_states = np.unique(np.clip(np.round(regdf['state'].unique() * value), a_min=0, a_max=max_k))
        allowed_states = np.array(sorted(allowed_states)).astype(int)
            
        rpc = total_reads / (initial_ploidy * len(regdf))
        fpc = total_fragments / (initial_ploidy * len(regdf))

        #print(datetime.now(), instruction, value, initial_ploidy, fpc, rpc, allowed_states)
        
        means_b = (k_range-1) * (fpc * k_range) * (fpc * k_range - 1) * (ell ** 2 / (2 * ell - 1)) * (ell / (k_range * bin_length - ell))
        means_b = np.maximum(means_b, 1)
        means_b *= my_offset_mult[:max_k + 1]
        means_r = np.maximum(1, rpc * k_range)

        offdiag_val = 1e-6
        diag_val = 1 - (len(allowed_states) - 1) * offdiag_val

        T = np.diag(np.ones(len(allowed_states)) * (diag_val - offdiag_val)) + offdiag_val
                
        initial_means = np.vstack([means_r[allowed_states], means_b[allowed_states]]).T
        
        if scale_reads:
            mean_scale_bounds=(min_mean_scale, max_mean_scale)
        else:
            mean_scale_bounds = np.array([[1,1],
                                          [min_mean_scale, max_mean_scale]])

        hmm = ConstrainedGaussianHMM(
            means=initial_means,
            transition_matrix=T,
            covariance_type=covariance_type,
            fix_means=fix_means,
            learn_mean_scaling=learn_mean_scaling,
            fix_transitions=not fit_transitions,
            mean_scale_bounds=mean_scale_bounds
        )

        hmm = hmm.fit(Xs)
        states_idx = np.concatenate(hmm.predict(Xs))
        states = allowed_states[states_idx]
        probs = np.concatenate([hmm.predict_proba(X) for X in Xs])
        map_states_idx = np.argmax(probs, axis=1)
        map_states = allowed_states[map_states_idx]
        map_score = np.sum(probs[np.arange(probs.shape[0]), map_states_idx])
        viterbi_score = np.sum(probs[np.arange(probs.shape[0]), states_idx])

        score = np.sum(hmm.score(Xs))

        cell_results.append({
            'cell_id':cell_id,
            'n_bins':len(regdf),
            'initial_ploidy':initial_ploidy,
            'multiplier':value,
            'rpc':rpc,
            'fpc':fpc,
            'ploidy_result':np.mean(states),
            'score':score,
            'model':hmm,
            'states':states,
            'map_states':map_states,
            'map_ploidy':np.mean(map_states),
            'map_score':map_score,
            'viterbi_score':viterbi_score,
            'probs':probs,
            'index':index,
            'bases_var':regdf['overlap_bases'].var(),
            'reads_var':regdf['reads'].var(),
            'bases_mean':regdf['overlap_bases'].mean(),
            'reads_mean':regdf['reads'].mean(),
            'bases_median':regdf['overlap_bases'].median(),
            'reads_median':regdf['reads'].median(),
            'bases_min':regdf['overlap_bases'].min(),
            'reads_min':regdf['reads'].min(),
            'bases_max':regdf['overlap_bases'].max(),
            'reads_max':regdf['reads'].max(),
            'min_mean_scale':min_mean_scale,
            'max_mean_scale':max_mean_scale,
            'reads_mean_scale':hmm.mean_scales_[0],
            'bases_mean_scale':hmm.mean_scales_[1],
        })
        if verbose:
            print(datetime.now(), f"Fitted {cell_id} with ploidy guess {initial_ploidy} and score {score:.2f}")
    
    if return_all:
        return cell_results
    else:
        best_idx = None
        best_score = np.inf * -1
        for i, r in enumerate(cell_results):
            if r['score'] > best_score:
                best_idx = i
                best_score = r['score']
        return cell_results[best_idx]

def fit_cell_restrict_states_wrapper(params):
    return fit_cell_restrict_states(**params)


def get_cell_df(adata, cell_id):
    cell_adata = adata[cell_id].copy()
    regdf = pd.DataFrame({
        'reads':cell_adata[cell_id].X.toarray().flatten(),
        **{layer: cell_adata[cell_id, :].layers[layer].toarray().flatten() 
           for layer in ['overlaps', 'overlap_bases', 'n_fragments', 'mean_fragment_length', 'state']}},
                        index=cell_adata.var_names)
    regdf['cell_id'] = cell_id
    return regdf
    
def identify_outliers(regdf, outlier_threshold=None):
    if outlier_threshold is not None:
        # remove points outside of <outlier_threshold> x IQR (only the most extreme outliers)
        for outlier_field in ['reads', 'overlap_bases']:
            iqr = regdf[outlier_field].quantile(0.75) - regdf[outlier_field].quantile(0.25)
            median = regdf[outlier_field].quantile(0.5)
            ulim = median + outlier_threshold * iqr
            llim = median - outlier_threshold * iqr
            #print(outlier_field, llim, ulim)
            regdf[f'is_outlier_{outlier_field}'] = False
            regdf.loc[(regdf[outlier_field] > ulim) | (regdf[outlier_field] < llim), 
                f'is_outlier_{outlier_field}'] = True
        regdf['is_outlier'] = np.any(regdf[[f'is_outlier_{f}' for f in ['reads', 'overlap_bases']]], axis=1)
    else:
        regdf['is_outlier'] = False

    return regdf

def identify_outliers_state(regdf, outlier_threshold=None):
    if outlier_threshold is not None:
        assert outlier_threshold > 0
        for outlier_field in ['reads', 'overlap_bases']:
            iqr = regdf.groupby('state')[outlier_field].quantile(0.75) - regdf.groupby('state')[outlier_field].quantile(0.25)
            median = regdf.groupby('state')[outlier_field].quantile(0.5)
            ulim = median + outlier_threshold * iqr
            llim = median - outlier_threshold * iqr
            #print(outlier_field, ulim, llim)
            regdf[f'is_outlier_{outlier_field}'] = False
            regdf.loc[regdf[outlier_field] >= regdf['state'].map(ulim), f'is_outlier_{outlier_field}'] = True
            regdf.loc[regdf[outlier_field] <= regdf['state'].map(llim), f'is_outlier_{outlier_field}'] = True
        regdf['is_outlier'] = np.any(regdf[[f'is_outlier_{f}' for f in ['reads', 'overlap_bases']]], axis=1)
    else:
        regdf['is_outlier'] = False

    return regdf

def remove_rare_states(regdf, min_bins_per_state=0):
    sizes = regdf.groupby('state').size()
    rare_states = sizes[sizes < min_bins_per_state].index
    return regdf[~regdf['state'].isin(rare_states)].copy()


def correct_reads(regdf, lowess_frac=0.2):
    regdf = valid(regdf)
    regdf = ideal(regdf)
    regdf2 = regdf[regdf['valid'] & regdf['ideal'] & ~regdf['in_blacklist']].copy()
    regdf2.sort_values(by='gc', inplace=True)
    regdf2, curve2 = modal_quantile_regression(regdf2, lowess_frac=lowess_frac, field='reads')
    if curve2 is None:
        return regdf2
    regdf2['modal_curve_reads'] = regdf2.gc.apply(curve2)
    regdf2['modal_corrected_reads'] = regdf2['reads'] / regdf2['modal_curve_reads']
    return regdf2


def correct_bases(regdf, lowess_frac=0.2):
    # correct relationship between overlap bases and GC
    regdf = valid(regdf, field_name='overlap_bases')
    regdf = ideal(regdf, field_name='overlap_bases')
    regdf2 = regdf[regdf['valid'] & regdf['ideal'] & ~regdf['in_blacklist']].copy()
    regdf2.sort_values(by='gc', inplace=True)
    
    df_regression, curve = modal_quantile_regression(regdf2, lowess_frac=lowess_frac, field='overlap_bases')
    if curve is None:
        return regdf2

    regdf2['modal_curve_bases'] = regdf2.gc.apply(curve)
    regdf2['modal_corrected_bases'] = regdf2['overlap_bases'] / regdf2['modal_curve_bases']
    return regdf2

@click.command()
@click.option('--adata', type=str, required=True)
@click.option('--output_row', type=str, required=True)
@click.option('--output_adata', type=str, required=True)
@click.option('--output_table', type=str, required=True)
@click.option('--cells', type=str, required=False, help="Comma-separated list of cells to include in analysis")
@click.option('--cells_file', type=str, required=False, help="File indicating list of cells to analyze (one cell per line)")
@click.option('--min_bins_per_state', type=int, required=False, default=0)
@click.option('--cores', type=int, required=False, default=1)
@click.option('--max_k', type=int, required=False, default=12)
@click.option('--iqr_threshold', type=int, required=False, default=None)
@click.option('--covariance_type', type=str, required=False, default='full')
@click.option('--means', type=str, required=False, default='fixed')
@click.option('--correct_gc', is_flag=True, default=False)
@click.option('--lowess_frac', type=float, required=False, default=0.2)
@click.option('--clip_corrected_values', is_flag=True, default=False)
@click.option('--fit_transitions', is_flag=True, default=False)
@click.option('--min_mean_scale', type=float, required=False, default=0)
@click.option('--max_mean_scale', type=float, required=False, default=np.inf)
@click.option('--bases_dist_quantile', type=float, required=False, default=0.8)
@click.option('--is_spectrum', is_flag=True, default=False)
def run_scplover_adata(adata, cores, max_k, iqr_threshold,
output_row, output_table, output_adata, cells, cells_file, min_bins_per_state, covariance_type, means, correct_gc, lowess_frac
, clip_corrected_values, fit_transitions, bases_dist_quantile, min_mean_scale, max_mean_scale, is_spectrum):
    assert max_k <= len(offset_mult), f'max_k above {len(offset_mult)} not supported'
    if covariance_type not in ['full', 'diag', 'spherical']:
        raise ValueError(f"Expected 'full', 'diag', or 'spherical' for argument 'covariance_type', got: {covariance_type}")
    if means not in ['fixed', 'scale', 'free']:
        raise ValueError(f"Expected 'fixed', 'scale', or 'free' for argument 'means', got: {means}")

    adata = ad.read_h5ad(adata)
        
    extra_blacklist = ['8:43000001-43500000']
    adata.var['in_blacklist'] = adata.var['in_blacklist'] | adata.var.index.isin(extra_blacklist)

    assert cells is None or cells_file is None, "Found both 'cells' and 'cell_file' arguments."
    if cells:
        cells = cells.strip().split(',')
    elif cells_file:
        cells = [a.strip() for a in open(cells_file).readlines()]
    else:
        cells = adata.obs.index
    #print(datetime.now(), 'loaded anndata')
    missing_cells = set(cells) - set(adata.obs.index)
    if len(missing_cells) > 0:
        raise ValueError(f"Missing {len(missing_cells)} cells from anndata that were indicated in {'cell list' if cells_file is not None else 'cells'} argument.")
    adata = adata[cells].copy()

    # check for and fix earlier bug which stored total fragment length in all fragment length fields
    buggy_cells = np.where(np.logical_and(
        np.any(adata.layers['mean_fragment_length'] > 0, axis=1), 
        np.all(adata.layers['max_fragment_length'] == adata.layers['mean_fragment_length'], axis=1)))[0]
    if len(buggy_cells) > 0:
        print(datetime.now(), f'Found {len(buggy_cells)} cells with total fragment length in mean fragment length field')
        for i in buggy_cells:
            adata.layers['mean_fragment_length'][i] = adata.layers['max_fragment_length'][i] / np.maximum(1, adata.layers['n_fragments'][i])
    
    buggy_bins = np.where(np.logical_and(
        np.any(adata.layers['mean_fragment_length'] > 0, axis=0), 
        np.all(adata.layers['max_fragment_length'] == adata.layers['mean_fragment_length'], axis=0)))[0]
    if len(buggy_bins) > 0:
        print(datetime.now(), f'Found {len(buggy_bins)} bins with total fragment length in mean fragment length field')
        for j in buggy_bins:
            adata.layers['mean_fragment_length'][:, j] = adata.layers['max_fragment_length'][:, j] / np.maximum(1, adata.layers['n_fragments'][:, j])

    adata.obs['mean_fragment_length'] = (adata.layers['mean_fragment_length'] * adata.layers['n_fragments']).sum(axis=1) / adata.layers['n_fragments'].sum(axis=1)
    assert np.max(adata.obs['mean_fragment_length']) < 1000, f'Found excessive mean fragment length: {adata.obs["mean_fragment_length"].max()}'

    bin_lengths = adata.var['end'] - adata.var['start']
    assert np.all(bin_lengths == bin_lengths.iloc[0]), 'Not all bin lengths are the same - currently unsupported'
    bin_length = bin_lengths.iloc[0] 

    all_params = []
    skipped_outlier_filtering = set()
    for my_cell in adata.obs.index:
        regdf = get_cell_df(adata, my_cell)
        regdf = regdf.merge(adata.var[['gc', 'map', 'in_blacklist']], left_on='bin', right_index=True, validate='1:1')
        regdf = regdf[~regdf['in_blacklist']]
        regdf = regdf[regdf['n_fragments'] > 0]
        regdf = identify_outliers_state(regdf, outlier_threshold=iqr_threshold)
        if (~regdf['is_outlier']).sum() > 3000:
            # skip filtering if very few bins would remain
            regdf = regdf[~regdf['is_outlier']]    
        else:
            skipped_outlier_filtering.add(my_cell)
            print(f'WARNING: Skipping outlier filtering for cell {my_cell} due to low number of bins')    

        if correct_gc:
            # compute parameters needed to scale corrected values to original space
            k_range = np.arange(max_k + 1)
            ploidy = regdf['state'].mean()
            fpc = regdf['n_fragments'].sum() / (ploidy * len(regdf))
            ell = (regdf['mean_fragment_length'] * regdf['n_fragments']).sum() / regdf['n_fragments'].sum()
            rpc = regdf['reads'].sum() / (ploidy * len(regdf))

            if clip_corrected_values:
                min_reads, max_reads = regdf['reads'].min(), regdf['reads'].max()
                min_bases, max_bases = regdf['overlap_bases'].min(), regdf['overlap_bases'].max()

            # perform modal quantile regression
            regdf = correct_reads(regdf, lowess_frac=lowess_frac)
            regdf = correct_bases(regdf, lowess_frac=lowess_frac)
            regdf = remove_rare_states(regdf, min_bins_per_state)

            ## identify the multiplicative factors that align modes in GC-corrected space to original space
            ## different value for reads vs. overlap bases
            ## shortcut here uses existing copy-number states for directness, could be done by other approaches
            # assume existing CN profile identifies reasonable modes in the data

            # restore reads per copy by regressing against state
            regdf['expected_reads'] = regdf['state'] * rpc
            model = smf.ols('expected_reads ~ modal_corrected_reads + 0', data=regdf)
            result = model.fit()
            regdf['reads'] = regdf['modal_corrected_reads'] * result.params['modal_corrected_reads']

            means_b = (k_range-1) * (fpc * k_range) * (fpc * k_range - 1) * offset_mult[:max_k + 1] * (ell ** 2 / (2 * ell - 1)) * (ell / (k_range * bin_length - ell))
            regdf['expected_bases'] = regdf['state'].map(lambda x:means_b[min(max_k, x)])
            
            # ignore extreme bases for purposes of scaling
            regdf['bases_L1'] = np.abs(regdf['overlap_bases'] - regdf['expected_bases'])
            bases_dist_threshold = np.quantile(regdf['bases_L1'], bases_dist_quantile)

            model = smf.ols('expected_bases ~ modal_corrected_bases + 0', data=regdf[regdf['bases_L1'] < bases_dist_threshold])
            result = model.fit()
            regdf['overlap_bases'] = regdf['modal_corrected_bases'] * result.params['modal_corrected_bases']
            if clip_corrected_values:
                regdf['overlap_bases'] = np.clip(regdf['overlap_bases'], min_bases, max_bases)
                regdf['reads'] = np.clip(regdf['reads'], min_reads, max_reads)

        else:
            regdf = remove_rare_states(regdf, min_bins_per_state)

        all_params.append(
            {'regdf':regdf, 
            'max_k':max_k, 
            'means':means, 
            'bin_length':bin_length, 
            'covariance_type':covariance_type,
            'return_all':True,
            'fit_transitions':fit_transitions,
            'verbose':False,
            'min_mean_scale':min_mean_scale,
            'max_mean_scale':max_mean_scale,
            'is_spectrum':is_spectrum,
            }
        )

    #print(datetime.now(), f'generated {len(all_params)} params')

    if cores > 1 and len(all_params) > 1:
        with mp.Pool(cores) as p:
            results = p.map(fit_cell_restrict_states_wrapper, all_params)
    else:
        results = [fit_cell_restrict_states_wrapper(p) for p in all_params]
    # results is a list of lists, each inner list contains all results for a single cell (each of which is a dict)

    # Assemble scalar values into dataframe and write to file
    df = pd.concat([pd.DataFrame(r) for r in results])

    dont_write_columns = ['model', 'states', 'map_states', 'mean_scales', 'index', 'probs']

    df['means'] = means
    df['max_k'] = max_k
    drop_columns = df.columns.intersection(dont_write_columns) 
    df.drop(columns=drop_columns).to_csv(output_table, index=False)

    # write table with 1 row per cell (most likely ploidy)
    best_rows = []
    best_results = {}
    for cell_id, celldf in df.groupby('cell_id'):
        idx = celldf['score'].idxmax()
        best_rows.append(celldf.loc[idx])

        best_initial_ploidy = celldf.loc[idx, 'initial_ploidy']
        cell_index = [i for i in range(len(results)) if results[i][0]['cell_id'] == cell_id]
        assert len(cell_index) == 1, 'found more than 1 set of results for cell {cell_id}'
        cell_index = cell_index[0]
        cell_results = results[cell_index]
        best_result = [a for a in cell_results if a['initial_ploidy'] == best_initial_ploidy and np.isclose(a['score'], celldf['score'].max())]
        assert len(best_result) == 1, f'found more than 1 result for cell {cell_id} with initial ploidy {best_initial_ploidy} and score {celldf["score"].max()}'
        best_result = best_result[0]
        if cell_id in skipped_outlier_filtering:
            best_result['skipped_outlier_filtering'] = True
        else:
            best_result['skipped_outlier_filtering'] = False

        best_results[cell_id] = best_result
    best_df = pd.DataFrame(best_rows)

    # Add inferred states (for most likely result per cell) to adata
    adata.layers['ghmm_state'] = np.ones(adata.shape) * np.nan
    for c, result in best_results.items():
        my_row = adata.obs.index.get_loc(c)
        my_cols = np.array([adata.var.index.get_loc(b) for b in result['index']])
        adata.layers['ghmm_state'][my_row, my_cols] = result['states']
    adata.write_h5ad(output_adata)

    # write full table - maybe can use margins to identify confident predictions
    drop_columns = best_df.columns.intersection(dont_write_columns) 
    best_df.drop(columns=drop_columns).to_csv(output_row, index=False)

if __name__ == '__main__':
    run_scplover_adata()