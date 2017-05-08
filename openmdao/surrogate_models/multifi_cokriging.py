"""
This module integrates the Multi-Fidelity Co-Kriging method described in
[LeGratiet2013].
(Author: Remi Vauclin <vauclin.remi@gmail.com>)

This code was implemented using the package scikit-learn as basis.
(Author: Vincent Dubourg <vincent.dubourg@gmail.com>)

OpenMDAO adaptation. Regression and correlation functions were directly copied
from scikit-learn package here to avoid scikit-learn dependency.
(Author: Remi Lafage <remi.lafage@onera.fr>)

ISAE/DMSM - ONERA/DCPS
"""

import numpy as np
from numpy import atleast_2d as array2d

from scipy import linalg
from scipy.optimize import minimize
from scipy.spatial.distance import squareform

from openmdao.surrogate_models.surrogate_model import MultiFiSurrogateModel

import logging
_logger = logging.getLogger()

MACHINE_EPSILON = np.finfo(np.double).eps # machine precision
NUGGET= 10.* MACHINE_EPSILON # nugget for robustness

INITIAL_RANGE_DEFAULT = 0.3 # initial range for optimizer
TOLERANCE_DEFAULT     = 1e-6    # stopping criterion for MLE optimization

THETA0_DEFAULT = 0.5
THETAL_DEFAULT = 1e-5
THETAU_DEFAULT = 50

if hasattr(linalg, 'solve_triangular'):
    # only in scipy since 0.9
    solve_triangular = linalg.solve_triangular
else:
    # slower, but works
    def solve_triangular(x, y, lower=True):
        return linalg.solve(x, y)


def constant_regression(x):
    """
    Zero order polynomial (constant, p = 1) regression model.
    x --> f(x) = 1
    """
    x = np.asarray(x, dtype=np.float)
    n_eval = x.shape[0]
    f = np.ones([n_eval, 1])
    return f

def linear_regression(x):
    """
    First order polynomial (linear, p = n+1) regression model.
    x --> f(x) = [ 1, x_1, ..., x_n ].T
    """
    x = np.asarray(x, dtype=np.float)
    n_eval = x.shape[0]
    f = np.hstack([np.ones([n_eval, 1]), x])
    return f

def squared_exponential_correlation(theta, d):
    """
    Squared exponential correlation model (Radial Basis Function).
    (Infinitely differentiable stochastic process, very smooth)::

                                            n
        theta, dx --> r(theta, dx) = exp(  sum  - theta_i * (dx_i)^2 )
                                          i = 1

    Args
    ----

    theta: array_like
        An array with shape 1 (isotropic) or n (anisotropic) giving the
        autocorrelation parameter(s).

    dx: array_like
        An array with shape (n_eval, n_features) giving the componentwise
        distances between locations x and x' at which the correlation model
        should be evaluated.

    Returns
    -------

    r: array_like
        An array with shape (n_eval, ) containing the values of the
        autocorrelation model.
    """

    theta = np.asarray(theta, dtype=np.float)
    d = np.asarray(d, dtype=np.float)

    if d.ndim > 1:
        n_features = d.shape[1]
    else:
        n_features = 1

    if theta.size == 1:
        return np.exp(-theta[0] * np.sum(d ** 2, axis=1))
    elif theta.size != n_features:
        raise ValueError("Length of theta must be 1 or %s" % n_features)
    else:
        return np.exp(-np.sum(theta.reshape(1, n_features) * d ** 2, axis=1))


def l1_cross_distances(X, Y=None):
    """
Computes the nonzero componentwise L1 cross-distances between the vectors
in X and Y.

Args
----

X: array_like
    An array with shape (n_samples_X, n_features)

Y: array_like
    An array with shape (n_samples_Y, n_features)

Returns
-------

D: array with shape (n_samples * (n_samples - 1) / 2, n_features)
    The array of componentwise L1 cross-distances.

"""
    if Y is None:
        X = array2d(X)
        n_samples, n_features = X.shape
        n_nonzero_cross_dist = n_samples * (n_samples - 1) // 2
        D = np.zeros((n_nonzero_cross_dist, n_features))
        ll_1 = 0
        for k in range(n_samples - 1):
            ll_0 = ll_1
            ll_1 = ll_0 + n_samples - k - 1
            D[ll_0:ll_1] = np.abs(X[k] - X[(k + 1):])

        return D

    else:
        X = array2d(X)
        Y = array2d(Y)
        n_samples_X, n_features_X = X.shape
        n_samples_Y, n_features_Y = Y.shape
        if n_features_X != n_features_Y:
            raise ValueError("X and Y must have the same dimensions.")
        n_features = n_features_X

        n_nonzero_cross_dist = n_samples_X * n_samples_Y
        D = np.zeros((n_nonzero_cross_dist, n_features))
        ll_1 = 0
        for k in range(n_samples_X):
            ll_0 = ll_1
            ll_1 = ll_0 + n_samples_Y# - k - 1
            D[ll_0:ll_1] = np.abs(X[k] - Y)

        return D


class MultiFiCoKriging(object):

    """
This class integrates the Multi-Fidelity Co-Kriging method described in
[LeGratiet2013]_.

Args
----

regr: string or callable, optional
    A regression function returning an array of outputs of the linear
    regression functional basis for Universal Kriging purpose.
    regr is assumed to be the same for all levels of code.
    Default assumes a simple constant regression trend.
    Available built-in regression models are:
    'constant', 'linear'

rho_regr: string or callable, optional
    A regression function returning an array of outputs of the linear
    regression functional basis. Defines the regression function for the
    autoregressive parameter rho.
    rho_regr is assumed to be the same for all levels of code.
    Default assumes a simple constant regression trend.
    Available built-in regression models are:
    'constant', 'linear'

theta: double, array_like or list, optional
    Value of correlation parameters if they are known; no optimization is run.
    Default is None, so that optimization is run.
    if double: value is replicated for all features and all levels.
    if array_like: an array with shape (n_features, ) for
    isotropic calculation. It is replicated for all levels.
    if list: a list of nlevel arrays specifying value for each level

theta0: double, array_like or list, optional
    Starting point for the maximum likelihood estimation of the
    best set of parameters.
    Default is None and meaning use of the default 0.5*np.ones(n_features)
    if double: value is replicated for all features and all levels.
    if array_like: an array with shape (n_features, ) for
    isotropic calculation. It is replicated for all levels.
    if list: a list of nlevel arrays specifying value for each level

thetaL: double, array_like or list, optional
    Lower bound on the autocorrelation parameters for maximum
    likelihood estimation.
    Default is None meaning use of the default 1e-5*np.ones(n_features).
    if double: value is replicated for all features and all levels.
    if array_like: An array with shape matching theta0's. It is replicated
    for all levels of code.
    if list: a list of nlevel arrays specifying value for each level

thetaU: double, array_like or list, optional
    Upper bound on the autocorrelation parameters for maximum
    likelihood estimation.
    Default is None meaning use of default value 50*np.ones(n_features).
    if double: value is replicated for all features and all levels.
    if array_like: An array with shape matching theta0's. It is replicated
    for all levels of code.
    if list: a list of nlevel arrays specifying value for each level


Attributes
----------

`theta`: list
    Specified theta for each level OR the best set of autocorrelation parameters
    (the sought maximizer of the reduced likelihood function).

`rlf_value`: list
    The optimal negative concentrated reduced likelihood function value
    for each level.


Examples
--------

>>> from openmdao.surrogate_models.multifi_cokriging import MultiFiCoKriging
>>> import numpy as np
>>> # Xe: DOE for expensive code (nested in Xc)
>>> # Xc: DOE for cheap code
>>> # ye: expensive response
>>> # yc: cheap response
>>> Xe = np.array([[0],[0.4],[1]])
>>> Xc = np.vstack((np.array([[0.1],[0.2],[0.3],[0.5],[0.6],[0.7],[0.8],[0.9]]),Xe))
>>> ye = ((Xe*6-2)**2)*np.sin((Xe*6-2)*2)
>>> yc = 0.5*((Xc*6-2)**2)*np.sin((Xc*6-2)*2)+(Xc-0.5)*10. - 5
>>> model = MultiFiCoKriging(theta0=1, thetaL=1e-5, thetaU=50.)
>>> model.fit([Xc, Xe], [yc, ye])
>>> # Prediction on x=0.05
>>> np.abs(float(model.predict([0.05])[0])- ((0.05*6-2)**2)*np.sin((0.05*6-2)*2)) < 0.05
True


Notes
-----

Implementation is based on the Package Scikit-Learn
(Author: Vincent Dubourg <vincent.dubourg@gmail.com>) which translates
the DACE Matlab toolbox, see [NLNS2002]_.


References
----------

.. [NLNS2002] H. B. Nielsen, S. N. Lophaven, and J. Sondergaard.
   `DACE - A MATLAB Kriging Toolbox.` (2002)
   http://www2.imm.dtu.dk/~hbn/dace/dace.pdf

.. [WBSWM1992] W. J. Welch, R. J. Buck, J. Sacks, H. P. Wynn, T. J. Mitchell,
   and M. D. Morris (1992). "Screening, predicting, and computer experiments."
   `Technometrics,` 34(1) 15--25.
   http://www.jstor.org/pss/1269548

.. [LeGratiet2013] L. Le Gratiet (2013). "Multi-fidelity Gaussian process
   regression for computer experiments."
   PhD thesis, Universite Paris-Diderot-Paris VII.

.. [TBKH2011] Toal, D. J., Bressloff, N. W., Keane, A. J., & Holden, C. M. E. (2011).
   "The development of a hybridized particle swarm for kriging hyperparameter
   tuning." `Engineering optimization`, 43(6), 675-699.
   """

    _regression_types = {
        'constant': constant_regression,
        'linear': linear_regression}

    def __init__(self, regr='constant', rho_regr='constant',
                 theta=None, theta0=None, thetaL=None, thetaU=None):

        self.corr     = squared_exponential_correlation
        self.regr     = regr
        self.rho_regr = rho_regr
        self.theta  = theta
        self.theta0 = theta0
        self.thetaL = thetaL
        self.thetaU = thetaU

        self._nfev = 0

    def _build_R(self, lvl, theta):
        """
        Builds the correlation matrix with given theta for the specified level.
        """

        D = self.D[lvl]
        n_samples = self.n_samples[lvl]

        R = np.eye(n_samples) * (1. + NUGGET)

        corr = squareform(self.corr(theta, D))
        R = R + corr

        return R


    def fit(self, X, y,
            initial_range=INITIAL_RANGE_DEFAULT, tol=TOLERANCE_DEFAULT):
        """
The Multi-Fidelity co-kriging model fitting method.

Args
----

X: list of double array_like elements
    A list of arrays with the input at which observations were made, from lowest
    fidelity to highest fidelity. Designs must be nested
    with X[i] = np.vstack([..., X[i+1])

y: list of double array_like elements
    A list of arrays with the observations of the scalar output to be predicted,
    from lowest fidelity to highest fidelity.

initial_range: float
    Initial range for the optimizer.

tol: float
    Optimizer terminates when the tolerance tol is reached.

"""
        # Run input checks
        # Transforms floats and arrays in lists to have a multifidelity structure
        self._check_list_structure(X, y)
        # Checks if all parameters are structured as required
        self._check_params()

        X = self.X
        y = self.y
        nlevel = self.nlevel
        n_samples = self.n_samples

        # initialize lists
        self.beta = nlevel*[0]
        self.beta_rho = nlevel*[None]
        self.beta_regr = nlevel*[None]
        self.C = nlevel*[0]
        self.D = nlevel*[0]
        self.F = nlevel*[0]
        self.p = nlevel*[0]
        self.q = nlevel*[0]
        self.G = nlevel*[0]
        self.sigma2 = nlevel*[0]
        self._R_adj = nlevel*[None]

        y_best = y[nlevel-1]
        for i in range(nlevel-1)[::-1]:
            y_best = np.concatenate((y[i][:-n_samples[i+1]],y_best))
        self.y_best = y_best

        self.y_mean = np.zeros(1)
        self.y_std  = np.ones(1)
        self.X_mean = np.zeros(1)
        self.X_std  = np.ones(1)

        for lvl in range(nlevel):

            # Calculate matrix of distances D between samples
            self.D[lvl] = l1_cross_distances(X[lvl])
            if (np.min(np.sum(self.D[lvl], axis=1)) == 0.):
                raise Exception("Multiple input features cannot have the same"
                                " value.")

            # Regression matrix and parameters
            self.F[lvl] = self.regr(X[lvl])
            self.p[lvl] = self.F[lvl].shape[1]

            # Concatenate the autoregressive part for levels > 0
            if lvl > 0:
                F_rho = self.rho_regr(X[lvl])
                self.q[lvl] = F_rho.shape[1]
                self.F[lvl] = np.hstack((F_rho*np.dot((self.y[lvl-1])[-n_samples[lvl]:],
                                              np.ones((1,self.q[lvl]))), self.F[lvl]))
            else:
                self.q[lvl] = 0

            n_samples_F_i = self.F[lvl].shape[0]

            if n_samples_F_i != n_samples[lvl]:
                raise Exception("Number of rows in F and X do not match. Most "
                                "likely something is going wrong with the "
                                "regression model.")

            if int(self.p[lvl] + self.q[lvl]) >= n_samples_F_i:
                raise Exception(("Ordinary least squares problem is undetermined "
                                 "n_samples=%d must be greater than the regression"
                                 " model size p+q=%d.")
                                 % (n_samples[i], self.p[lvl]+self.q[lvl]))

        # Set attributes
        self.X = X
        self.y = y

        self.rlf_value = np.zeros(nlevel)

        for lvl in range(nlevel):
            # Determine Gaussian Process model parameters
            if self.theta[lvl] is None:
                # Maximum Likelihood Estimation of the parameters
                sol = self._max_rlf(lvl=lvl, initial_range=initial_range, tol=tol)
                self.theta[lvl] = sol['theta']
                self.rlf_value[lvl] = sol['rlf_value']

                if np.isinf(self.rlf_value[lvl]):
                    raise Exception("Bad parameter region. "
                                    "Try increasing upper bound")
            else:
                self.rlf_value[lvl] = self.rlf(lvl=lvl)
                if np.isinf(self.rlf_value[lvl]):
                    raise Exception("Bad point. Try increasing theta0.")

        return



    def rlf(self, lvl, theta=None):
        """
This function determines the BLUP parameters and evaluates the negative reduced
likelihood function for the given autocorrelation parameters theta.

Maximizing this function wrt the autocorrelation parameters theta is
equivalent to maximizing the likelihood of the assumed joint Gaussian
distribution of the observations y evaluated onto the design of
experiments X.

Args
----

self: Multi-Fidelity Co-Kriging object

lvl: Integer
    Level of fidelity

theta: array_like, optional
    An array containing the autocorrelation parameters at which the
    Gaussian Process model parameters should be determined.
    Default uses the built-in autocorrelation parameters
    (ie ``theta = self.theta``).

Returns
-------

rlf_value: double
    The value of the negative concentrated reduced likelihood function
    associated to the given autocorrelation parameters theta.
"""

        if theta is None:
            # Use built-in autocorrelation parameters
            theta = self.theta[lvl]

        # Initialize output
        rlf_value = 1e20

        # Retrieve data
        n_samples = self.n_samples[lvl]
        y = self.y[lvl]
        F = self.F[lvl]
        p = self.p[lvl]
        q = self.q[lvl]

        R = self._build_R(lvl, theta)

        try:
            C = linalg.cholesky(R, lower=True)
        except linalg.LinAlgError:
            _logger.warning(('Cholesky decomposition of R at level %i failed' % lvl) +
                             ' with theta='+str(theta))
            return rlf_value

        # Get generalized least squares solution
        Ft = solve_triangular(C, F, lower=True)
        Yt = solve_triangular(C, y, lower=True)
        try:
            Q, G = linalg.qr(Ft, econ=True)
        except:
            # DeprecationWarning: qr econ argument will be removed after scipy
            # 0.7. The economy transform will then be available through the
            # mode='economic' argument.
            Q, G = linalg.qr(Ft, mode='economic')
            pass

        # Universal Kriging
        beta = solve_triangular(G, np.dot(Q.T, Yt))

        err = Yt - np.dot(Ft,beta)
        err2 = np.dot(err.T, err)[0,0]
        self._err = err
        sigma2 = err2 /(n_samples - p - q)
        detR = ((np.diag(C))**(2./n_samples)).prod()

        rlf_value = (n_samples - p - q)*np.log10(sigma2) \
                    + n_samples*np.log10(detR)

        self.beta_rho[lvl] = beta[:q]
        self.beta_regr[lvl] = beta[q:]
        self.beta[lvl] = beta
        self.sigma2[lvl] = sigma2
        self.C[lvl] = C
        self.G[lvl] = G

        return rlf_value


    def _max_rlf(self, lvl, initial_range, tol):
        """
This function estimates the autocorrelation parameter theta
as the maximizer of the reduced likelihood function of the given level (lvl).
(Minimization of the negative reduced likelihood function is used for convenience.)

Args
----

self: Most parameters are stored in the Gaussian Process model object.

lvl: integer
    Level of fidelity

initial_range: float
    Initial range of the optimizer

tol: float
    Optimizer terminates when the tolerance tol is reached.

Returns
-------

optimal_theta: array_like
optimal_rlf_value: double
    The optimal negative reduced likelihood function value.

res: dict
    res['theta']: optimal theta
    res['rlf_value']: optimal value for likelihood
"""
        # Initialize input
        thetaL = self.thetaL[lvl]
        thetaU = self.thetaU[lvl]

        def rlf_transform(x):
            return self.rlf(theta=10.**x, lvl=lvl)

        # Use specified starting point as first guess
        theta0 = self.theta0[lvl]
        x0 = np.log10(theta0[0])

        constraints = []
        for i in range(theta0.size):
            constraints.append({'type': 'ineq', 'fun': lambda log10t,i=i:
                                log10t[i] - np.log10(thetaL[0][i])})
            constraints.append({'type': 'ineq', 'fun': lambda log10t,i=i:
                                np.log10(thetaU[0][i]) - log10t[i]})

        constraints = tuple(constraints)
        sol = minimize(rlf_transform, x0, method='COBYLA',
                       constraints=constraints,
                       options={'rhobeg': initial_range,
                                'tol': tol, 'disp': 0})

        log10_optimal_x = sol['x']
        optimal_rlf_value = sol['fun']
        self._nfev += sol['nfev']

        optimal_theta = 10. ** log10_optimal_x

        res = {}
        res['theta'] = optimal_theta
        res['rlf_value'] = optimal_rlf_value

        return res


    def predict(self, X, eval_MSE=True):
        """
This function performs the predictions of the kriging model on X.

Args
----

X: array_like
    An array with shape (n_eval, n_features) giving the point(s) at
    which the prediction(s) should be made.

eval_MSE: boolean, optional
    A boolean specifying whether the Mean Squared Error should be
    evaluated or not. Default assumes evalMSE is True.

Returns
-------

y: array_like
    An array with shape (n_eval, ) with the Best Linear Unbiased
    Prediction at X. If all_levels is set to True, an array
    with shape (n_eval, nlevel) giving the BLUP for all levels.

MSE: array_like, optional (if eval_MSE is True)
    An array with shape (n_eval, ) with the Mean Squared Error at X.
    If all_levels is set to True, an array with shape (n_eval, nlevel)
    giving the MSE for all levels.
"""

        X = array2d(X)
        nlevel = self.nlevel
        n_eval, n_features_X = X.shape

        # Calculate kriging mean and variance at level 0
        mu = np.zeros((n_eval, nlevel))

        f = self.regr(X)
        f0 = self.regr(X)
        dx = l1_cross_distances(X, Y=self.X[0])

        # Get regression function and correlation
        F = self.F[0]
        C = self.C[0]

        beta = self.beta[0]
        Ft = solve_triangular(C, F, lower=True)
        yt = solve_triangular(C, self.y[0], lower=True)
        r_ = self.corr(self.theta[0], dx).reshape(n_eval, self.n_samples[0])
        gamma = solve_triangular(C.T, yt - np.dot(Ft,beta), lower=False)

        # Scaled predictor
        mu[:,0]= (np.dot(f, beta) + np.dot(r_,gamma)).ravel()

        if eval_MSE:
            self.sigma2_rho = nlevel*[None]
            MSE = np.zeros((n_eval,nlevel))
            r_t = solve_triangular(C, r_.T, lower=True)
            G = self.G[0]

            u_ = solve_triangular(G.T, f.T - np.dot(Ft.T, r_t), lower=True)
            MSE[:,0] = self.sigma2[0] * (1 \
                        - (r_t**2).sum(axis=0) + (u_**2).sum(axis=0))

        # Calculate recursively kriging mean and variance at level i
        for i in range(1,nlevel):
            C = self.C[i]
            F = self.F[i]
            g = self.rho_regr(X)
            dx = l1_cross_distances(X, Y=self.X[i])
            r_ = self.corr(self.theta[i], dx).reshape(n_eval, self.n_samples[i])
            f = np.vstack((g.T*mu[:,i-1], f0.T))

            Ft = solve_triangular(C, F, lower=True)
            yt = solve_triangular(C, self.y[i], lower=True)
            r_t = solve_triangular(C,r_.T, lower=True)
            G = self.G[i]
            beta = self.beta[i]

            # scaled predictor
            mu[:,i] = (np.dot(f.T, beta) \
                       + np.dot(r_t.T, yt - np.dot(Ft,beta))).ravel()

            if eval_MSE:
                Q_ = (np.dot((yt-np.dot(Ft,beta)).T, yt-np.dot(Ft,beta)))[0,0]
                u_ = solve_triangular(G.T, f - np.dot(Ft.T, r_t), lower=True)
                sigma2_rho = np.dot(g, \
                    self.sigma2[i]*linalg.inv(np.dot(G.T,G))[:self.q[i],:self.q[i]] \
                        + np.dot(beta[:self.q[i]], beta[:self.q[i]].T))
                sigma2_rho = (sigma2_rho * g).sum(axis=1)

                MSE[:,i] = sigma2_rho * MSE[:,i-1] \
                        + Q_/(2*(self.n_samples[i]-self.p[i]-self.q[i])) \
                        * (1 - (r_t**2).sum(axis=0)) \
                        + self.sigma2[i] * (u_**2).sum(axis=0)

        # scaled predictor
        for i in range(nlevel):# Predictor
            mu[:,i] = self.y_mean + self.y_std * mu[:,i]
            if eval_MSE:
                MSE[:,i] = self.y_std**2 * MSE[:,i]

        if eval_MSE:
            return mu[:,-1].reshape((n_eval,1)), MSE[:,-1].reshape((n_eval,1))
        else:
            return mu[:,-1].reshape((n_eval,1))


    def _check_list_structure(self, X, y):

        if type(X) is not list:
            nlevel = 1
            X = [X]
        else:
            nlevel = len(X)


        if type(y) is not list:
            y = [y]

        if len(X) != len(y):
            raise ValueError("X and y must have the same length.")

        n_samples = np.zeros(nlevel, dtype = int)
        n_features = np.zeros(nlevel, dtype = int)
        n_samples_y = np.zeros(nlevel, dtype = int)
        for i in range(nlevel):
            n_samples[i], n_features[i] = X[i].shape
            if i>1 and n_features[i] != n_features[i-1]:
                raise ValueError("All X must have the same number of columns.")
            y[i] = np.asarray(y[i]).ravel()[:, np.newaxis]
            n_samples_y[i] = y[i].shape[0]
            if n_samples[i] != n_samples_y[i]:
                raise ValueError("X and y must have the same number of rows.")

        self.n_features = n_features[0]

        if type(self.theta) is not list:
            self.theta = nlevel*[self.theta]
        elif len(self.theta) != nlevel:
            raise ValueError("theta must be a list of %d element(s)." % nlevel)

        if type(self.theta0) is not list:
            self.theta0 = nlevel*[self.theta0]
        elif len(self.theta0) != nlevel:
            raise ValueError("theta0 must be a list of %d elements." % nlevel)

        if type(self.thetaL) is not list:
            self.thetaL = nlevel*[self.thetaL]
        elif len(self.thetaL) != nlevel:
            raise ValueError("thetaL must be a list of %d elements." % nlevel)

        if type(self.thetaU) is not list:
            self.thetaU = nlevel*[self.thetaU]
        elif len(self.thetaU) != nlevel:
            raise ValueError("thetaU must be a list of %d elements." % nlevel)

        self.nlevel = nlevel
        self.X = X[:]
        self.y = y[:]
        self.n_samples = n_samples

        return


    def _check_params(self):

        # Check regression model
        if not callable(self.regr):
            if self.regr in self._regression_types:
                self.regr = self._regression_types[self.regr]
            else:
                raise ValueError("regr should be one of %s or callable, "
                                 "%s was given."
                                 % (self._regression_types.keys(), self.regr))

        # Check rho regression model
        if not callable(self.rho_regr):
            if self.rho_regr in self._regression_types:
                self.rho_regr = self._regression_types[self.rho_regr]
            else:
                raise ValueError("rho_regr should be one of %s or callable, "
                                 "%s was given."
                                 % (self._regression_types.keys(), self.rho_regr))

        for i in range(self.nlevel):
            # Check correlation parameters
            if self.theta[i] is not None:
                self.theta[i] = array2d(self.theta[i])
                if np.any(self.theta[i] <= 0):
                    raise ValueError("theta0 must be strictly positive.")

            if self.theta0[i] is not None:
                self.theta0[i] = array2d(self.theta0[i])
                if np.any(self.theta0[i] <= 0):
                    raise ValueError("theta0 must be strictly positive.")
            else:
                self.theta0[i] = array2d(self.n_features*[THETA0_DEFAULT])

            lth = self.theta0[i].size

            if self.thetaL[i] is not None:
                self.thetaL[i] = array2d(self.thetaL[i])
                if self.thetaL[i].size != lth:
                    raise ValueError("theta0 and thetaL must have the "
                                     "same length.")
            else:
                self.thetaL[i] = array2d(self.n_features*[THETAL_DEFAULT])

            if self.thetaU[i] is not None:
                self.thetaU[i] = array2d(self.thetaU[i])
                if self.thetaU[i].size != lth:
                    raise ValueError("theta0 and thetaU must have the "
                                     "same length.")
            else:
                self.thetaU[i] = array2d(self.n_features*[THETAU_DEFAULT])

            if np.any(self.thetaL[i] <= 0) or np.any(self.thetaU[i] < self.thetaL[i]):
                raise ValueError("The bounds must satisfy O < thetaL <= "
                                 "thetaU.")

        return


class MultiFiCoKrigingSurrogate(MultiFiSurrogateModel):
    """
    OpenMDAO adapter of multi-fidelity recursive cokriging method described
    in [LeGratiet2013]. See MultiFiCoKriging class.
    """

    def __init__(self, regr='constant', rho_regr='constant',
                 theta=None, theta0=None, thetaL=None, thetaU=None,
                 tolerance=TOLERANCE_DEFAULT, initial_range=INITIAL_RANGE_DEFAULT):
        super(MultiFiCoKrigingSurrogate, self).__init__()

        self.tolerance=tolerance
        self.initial_range=initial_range
        self.model = MultiFiCoKriging(regr=regr,rho_regr=rho_regr, theta=theta,
                                      theta0=theta0, thetaL=thetaL, thetaU=thetaU)

    def predict(self, new_x):
        """Calculates a predicted value of the response based on the current
        trained model for the supplied list of inputs.
        """
        Y_pred, MSE = self.model.predict([new_x])
        return Y_pred, np.sqrt(np.abs(MSE))

    def train_multifi(self,X,Y):
        """Train the surrogate model with the given set of inputs and outputs.
        """
        X, Y = self._fit_adapter(X, Y)
        self.model.fit(X, Y,tol=self.tolerance, initial_range=self.initial_range)

    def _fit_adapter(self, X, Y):
        # Manage special case with one fidelity
        # where can be called as [[xval1],[xval2]] instead of [[[xval1],[xval2]]]
        # we detect if shape(X[0]) is like (m,) instead of (m, n)
        if len(np.shape(np.array(X[0]))) == 1:
            X = [X]
            Y = [Y]

        X = [np.array(x) for x in reversed(X)]
        Y = [np.array(y) for y in reversed(Y)]
        return (X,Y)


class FloatMultiFiCoKrigingSurrogate(MultiFiCoKrigingSurrogate):
    """Predictions are returned as floats, which are the mean of the
    NormalDistribution predicted by the base class model."""

    def predict(self, new_x):
        dist = super(FloatMultiFiCoKrigingSurrogate, self).predict(new_x)
        return dist.mu


if __name__ == "__main__":
    import doctest
    doctest.testmod()
