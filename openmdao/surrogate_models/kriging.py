"""Surrogate model based on Kriging."""
import numpy as np
import scipy.linalg as linalg
from scipy.optimize import minimize

from openmdao.surrogate_models.surrogate_model import SurrogateModel

MACHINE_EPSILON = np.finfo(np.double).eps


class KrigingSurrogate(SurrogateModel):
    """
    Surrogate Modeling method based on the simple Kriging interpolation.

    Predictions are returned as a tuple of mean and RMSE. Based on Gaussian Processes
    for Machine Learning (GPML) by Rasmussen and Williams. (see also: scikit-learn).

    Attributes
    ----------
    alpha : ndarray
        Reduced likelihood parameter: alpha
    L : ndarray
        Reduced likelihood parameter: L
    n_dims : int
        Number of independents in the surrogate
    n_samples : int
        Number of training points.
    sigma2 : ndarray
        Reduced likelihood parameter: sigma squared
    thetas : ndarray
        Kriging hyperparameters.
    X : ndarray
        Training input values, normalized.
    X_mean : ndarray
        Mean of training input values, normalized.
    X_std : ndarray
        Standard deviation of training input values, normalized.
    Y : ndarray
        Training model response values, normalized.
    Y_mean : ndarray
        Mean of training model response values, normalized.
    Y_std : ndarray
        Standard deviation of training model response values, normalized.
    """

    def __init__(self, **kwargs):
        """
        Initialize all attributes.

        Parameters
        ----------
        **kwargs : dict
            options dictionary.
        """
        super(KrigingSurrogate, self).__init__(**kwargs)

        self.n_dims = 0                 # number of independent
        self.n_samples = 0              # number of training points
        self.thetas = np.zeros(0)

        self.alpha = np.zeros(0)
        self.L = np.zeros(0)
        self.sigma2 = np.zeros(0)

        # Normalized Training Values
        self.X = np.zeros(0)
        self.Y = np.zeros(0)
        self.X_mean = np.zeros(0)
        self.X_std = np.zeros(0)
        self.Y_mean = np.zeros(0)
        self.Y_std = np.zeros(0)

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        self.options.declare('eval_rmse', types=bool, default=False,
                             desc="Flag indicating whether the Root Mean Squared Error (RMSE) "
                                  "should be computed. Set to False by default.")

        # nugget smoothing parameter from [Sasena, 2002]
        self.options.declare('nugget', default=10. * MACHINE_EPSILON,
                             desc="Nugget smoothing parameter for smoothing noisy data. Represents "
                                  "the variance of the input values. If nugget is an ndarray, it "
                                  "must be of the same length as the number of training points. "
                                  "Default: 10. * Machine Epsilon")

        self.options.declare('lapack_driver', types=str, default='gesvd',
                             desc="Which lapack driver should be used for scipy's linalg.svd."
                                  "Options are 'gesdd' which is faster but not as robust,"
                                  "or 'gesvd' which is slower but more reliable."
                                  "'gesvd' is the default.")

    def train(self, x, y):
        """
        Train the surrogate model with the given set of inputs and outputs.

        Parameters
        ----------
        x : array-like
            Training input locations
        y : array-like
            Model responses at given inputs.
        """
        super(KrigingSurrogate, self).train(x, y)

        x, y = np.atleast_2d(x, y)

        self.n_samples, self.n_dims = x.shape

        if self.n_samples <= 1:
            self._raise('KrigingSurrogate requires at least 2 training points.',
                        exc_type=ValueError)

        # Normalize the data
        X_mean = np.mean(x, axis=0)
        X_std = np.std(x, axis=0)
        Y_mean = np.mean(y, axis=0)
        Y_std = np.std(y, axis=0)

        X_std[X_std == 0.] = 1.
        Y_std[Y_std == 0.] = 1.

        X = (x - X_mean) / X_std
        Y = (y - Y_mean) / Y_std

        self.X = X
        self.Y = Y
        self.X_mean, self.X_std = X_mean, X_std
        self.Y_mean, self.Y_std = Y_mean, Y_std

        def _calcll(thetas):
            """Calculate loglike (callback function)."""
            loglike = self._calculate_reduced_likelihood_params(np.exp(thetas))[0]
            return -loglike

        bounds = [(np.log(1e-5), np.log(1e5)) for _ in range(self.n_dims)]

        optResult = minimize(_calcll, 1e-1 * np.ones(self.n_dims), method='slsqp',
                             options={'eps': 1e-3},
                             bounds=bounds)

        if not optResult.success:
            msg = 'Kriging Hyper-parameter optimization failed: {0}'.format(optResult.message)
            self._raise(msg, exc_type=ValueError)

        self.thetas = np.exp(optResult.x)
        _, params = self._calculate_reduced_likelihood_params()
        self.alpha = params['alpha']
        self.U = params['U']
        self.S_inv = params['S_inv']
        self.Vh = params['Vh']
        self.sigma2 = params['sigma2']

    def _calculate_reduced_likelihood_params(self, thetas=None):
        """
        Calculate quantity with same maximum location as the log-likelihood for a given theta.

        Parameters
        ----------
        thetas : ndarray, optional
            Given input correlation coefficients. If none given, uses self.thetas
            from training.


        Returns
        -------
        ndarray
            Calculated reduced_likelihood
        dict
            Dictionary containing the parameters.
        """
        if thetas is None:
            thetas = self.thetas

        X, Y = self.X, self.Y
        params = {}

        # Correlation Matrix
        distances = np.zeros((self.n_samples, self.n_dims, self.n_samples))
        for i in range(self.n_samples):
            distances[i, :, i + 1:] = np.abs(X[i, ...] - X[i + 1:, ...]).T
            distances[i + 1:, :, i] = distances[i, :, i + 1:].T

        R = np.exp(-thetas.dot(np.square(distances)))
        R[np.diag_indices_from(R)] = 1. + self.options['nugget']

        [U, S, Vh] = linalg.svd(R, lapack_driver=self.options['lapack_driver'])

        # Penrose-Moore Pseudo-Inverse:
        # Given A = USV^* and Ax=b, the least-squares solution is
        # x = V S^-1 U^* b.
        # Tikhonov regularization is used to make the solution significantly
        # more robust.
        h = 1e-8 * S[0]
        inv_factors = S / (S ** 2. + h ** 2.)

        alpha = Vh.T.dot(np.einsum('j,kj,kl->jl', inv_factors, U, Y))
        logdet = -np.sum(np.log(inv_factors))
        sigma2 = np.dot(Y.T, alpha).sum(axis=0) / self.n_samples
        reduced_likelihood = -(np.log(np.sum(sigma2)) +
                               logdet / self.n_samples)

        params['alpha'] = alpha
        params['sigma2'] = sigma2 * np.square(self.Y_std)
        params['S_inv'] = inv_factors
        params['U'] = U
        params['Vh'] = Vh

        return reduced_likelihood, params

    def predict(self, x):
        """
        Calculate predicted value of the response based on the current trained model.

        Parameters
        ----------
        x : array-like
            Point at which the surrogate is evaluated.

        Returns
        -------
        ndarray
            Kriging prediction.
        ndarray, optional (if eval_rmse is True)
            Root mean square of the prediction error.
        """
        super(KrigingSurrogate, self).predict(x)

        thetas = self.thetas
        if isinstance(x, list):
            x = np.array(x)
        x = np.atleast_2d(x)
        n_eval = x.shape[0]

        # Normalize input
        x_n = (x - self.X_mean) / self.X_std

        r = np.zeros((n_eval, self.n_samples), dtype=x.dtype)
        for r_i, x_i in zip(r, x_n):
            r_i[:] = np.exp(-thetas.dot(np.square((x_i - self.X).T)))

        # Scaled Predictor
        y_t = np.dot(r, self.alpha)

        # Predictor
        y = self.Y_mean + self.Y_std * y_t

        if self.options['eval_rmse']:
            mse = (1. - np.dot(np.dot(r, self.Vh.T),
                               np.einsum('j,kj,lk->jl', self.S_inv, self.U, r))) * self.sigma2

            # Forcing negative RMSE to zero if negative due to machine precision
            mse[mse < 0.] = 0.
            return y, np.sqrt(mse)

        return y

    def linearize(self, x):
        """
        Calculate the jacobian of the Kriging surface at the requested point.

        Parameters
        ----------
        x : array-like
            Point at which the surrogate Jacobian is evaluated.

        Returns
        -------
        ndarray
            Jacobian of surrogate output wrt inputs.
        """
        thetas = self.thetas

        # Normalize Input
        x_n = (x - self.X_mean) / self.X_std

        r = np.exp(-thetas.dot(np.square((x_n - self.X).T)))

        # Z = einsum('i,ij->ij', X, Y) is equivalent to, but much faster and
        # memory efficient than, diag(X).dot(Y) for vector X and 2D array Y.
        # i.e., Z[i,j] = X[i]*Y[i,j]
        gradr = r * -2 * np.einsum('i,ij->ij', thetas, (x_n - self.X).T)
        jac = np.einsum('i,j,ij->ij', self.Y_std, 1. /
                        self.X_std, gradr.dot(self.alpha).T)
        return jac
