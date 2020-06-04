
"""
Surrogate Model based on second order response surface equations.
"""

from numpy import zeros, einsum
from numpy.dual import lstsq
from openmdao.surrogate_models.surrogate_model import SurrogateModel


class ResponseSurface(SurrogateModel):
    """
    Surrogate Model based on second order response surface equations.

    Attributes
    ----------
    betas : ndarray
        Vector of response surface equation coefficients.
    m : int
        Number of training points.
    n : int
        Number of independent variables.
    """

    def __init__(self):
        """
        Initialize all attributes.
        """
        super(ResponseSurface, self).__init__()

        self.m = 0  # number of training points
        self.n = 0  # number of independents
        # vector of response surface equation coefficients
        self.betas = zeros(0)

    def train(self, x, y):
        """
        Calculate response surface equation coefficients using least squares regression.

        Parameters
        ----------
        x : array-like
            Training input locations
        y : array-like
            Model responses at given inputs.
        """
        super(ResponseSurface, self).train(x, y)

        m = self.m = x.shape[0]
        n = self.n = x.shape[1]

        X = zeros((m, ((n + 1) * (n + 2)) // 2))

        # Modify X to include constant, squared terms and cross terms

        # Constant Terms
        X[:, 0] = 1.0

        # Linear Terms
        X[:, 1:n + 1] = x

        # Quadratic Terms
        X_offset = X[:, n + 1:]
        for i in range(n):
            # Z = einsum('i,ij->ij', X, Y) is equivalent to, but much faster and
            # memory efficient than, diag(X).dot(Y) for vector X and 2D array Y.
            # I.e. Z[i,j] = X[i]*Y[i,j]
            X_offset[:, :n - i] = einsum('i,ij->ij', x[:, i], x[:, i:])
            X_offset = X_offset[:, n - i:]

        # Determine response surface equation coefficients (betas) using least
        # squares
        self.betas, rs, r, s = lstsq(X, y)

    def predict(self, x):
        """
        Calculate predicted value of response based on the current response surface model.

        Parameters
        ----------
        x : array-like
            Point at which the surrogate is evaluated.

        Returns
        -------
        float
            Predicted response.
        """
        super(ResponseSurface, self).predict(x)

        n = x.size

        X = zeros(((self.n + 1) * (self.n + 2)) // 2)

        # Modify X to include constant, squared terms and cross terms

        # Constant Terms
        X[0] = 1.0

        # Linear Terms
        X[1:n + 1] = x

        # Quadratic Terms
        X_offset = X[n + 1:]
        for i in range(n):
            X_offset[:n - i] = x[i] * x[i:]
            X_offset = X_offset[n - i:]

        # Predict new_y using X and betas
        return X.dot(self.betas)

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
        n = self.n
        betas = self.betas

        x = x.flat

        jac = betas[1:n + 1, :].copy()
        beta_offset = betas[n + 1:, :]
        for i in range(n):
            jac[i, :] += x[i:].dot(beta_offset[:n - i, :])
            jac[i:, :] += x[i] * beta_offset[:n - i, :]
            beta_offset = beta_offset[n - i:, :]

        return jac.T
