import numpy as np

from math import ceil
from scipy.spatial import cKDTree

class NNBase(object):
    """
    Base class for common functionality between nearest neighbor interpolants.
    """

    def __init__(self, training_points, training_values, num_leaves=2):
        """
        Initialize the nearest neighbor interpolant by scaling input to the
        unit hypercube.

        Args
        ----
        training_points : ndarray
            ndarray of shape (num_points x independent dims) containing
            training inpit locations.

        training_values : ndarray
            ndarray of shape (num_points x dependent dims) containing
            training output values.

        """
        # training_points and training_values are the known points and their
        # respective values which will be interpolated against.
        # Grab the mins and ranges of each dimension
        self._tpm = np.amin(training_points, axis=0)
        self._tpr = (np.amax(training_points, axis=0) - self._tpm)
        self._tvm = np.amin(training_values, axis=0)
        self._tvr = (np.amax(training_values, axis=0) - self._tvm)

        # This prevents against collinear data (range = 0)
        self._tpr[self._tpr == 0] = 1
        self._tvr[self._tvr == 0] = 1

        # Normalize all points
        self._tp = (training_points - self._tpm) / self._tpr
        self._tv = (training_values - self._tvm) / self._tvr

        # Record number of dimensions and points
        self._indep_dims = training_points.shape[1]
        self._dep_dims = training_values.shape[1]
        self._ntpts = training_points.shape[0]

        # Make training data into a Tree
        leavesz = ceil(self._ntpts / float(num_leaves))
        self._KData = cKDTree(self._tp, leafsize=leavesz)

        # Cache for gradients
        self._pt_cache = None
