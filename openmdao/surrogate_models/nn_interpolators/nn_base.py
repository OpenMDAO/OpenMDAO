"""Define the NNBase class."""

import numpy as np

from math import ceil
from scipy.spatial import cKDTree


class NNBase(object):
    """
    Base class for common functionality between nearest neighbor interpolants.

    Attributes
    ----------
    _tpm : ndarray
        ndarray of shape (1 x independent dims) containing the minimum in each dimension of
         the training input locations.
    _tpr : ndarray
        ndarray of shape (1x independent dims) containing the range of each dimension of
        the training input locations.
    _tvm : ndarray
        ndarray of shape (1 x independent dims) containing the minimum in each dimension of
        the training output values.
    _tvr : ndarray
        ndarray of shape (1x independent dims) containing the range of each dimension of
        the training output values.
    _tp : ndarray
        ndarray of shape (num_points x independent dims) containing normalized training
        input locations.
    _tv : ndarray
        ndarray of shape (num_points x independent dims) containing normalized training
        output values.
    _indep_dims : int
        Number of independent dims
    _dep_dims : int
        Number of dependent dims
    _ntpts : int
        Number of training points
    _KData : scipy.spatial.cKDTree
        KDTree used for finding the nearest neighbors.
    _pt_cache : tuple(ndarray, ndarray, ndarray)
        Internal cache of the last found neighbors.
    _parent_name : str or None
        Absolute pathname of metamodel component that owns this surrogate.
    """

    def __init__(self, training_points, training_values, num_leaves=2, parent_name=''):
        """
        Initialize nearest neighbor interpolant by scaling input to the unit hypercube.

        Parameters
        ----------
        training_points : ndarray
            ndarray of shape (num_points x independent dims) containing training input locations.
        training_values : ndarray
            ndarray of shape (num_points x dependent dims) containing training output values.
        num_leaves : int
            How many leaves the tree should have.
        parent_name : str
            Absolute pathname of metamodel component that owns this surrogate.
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

        self._parent_name = parent_name

    def _raise(self, msg, exc_type=RuntimeError):
        """
        Raise the given exception type, with parent's name prepended to the message.

        Parameters
        ----------
        msg : str
            The error message.
        exc_type : class
            The type of the exception to be raised.
        """
        if self._parent_name is None:
            full_msg = msg
        else:
            full_msg = '{}: {}'.format(self._parent_name, msg)
        raise exc_type(full_msg)
