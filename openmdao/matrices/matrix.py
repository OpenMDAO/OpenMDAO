"""Define the base Matrix class."""
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix


# scipy sparse types allowed to be subjacs
sparse_types = (csr_matrix, csc_matrix, coo_matrix)


class Matrix(object):
    """
    Base matrix class.

    This class is used for global Jacobians.

    Parameters
    ----------
    submats : dict
        Dictionary of sub-matrix data keyed by (row_name, col_name).

    Attributes
    ----------
    _matrix : object
        implementation-specific representation of the actual matrix.
    _submats : dict
        dictionary of sub-matrix data keyed by (out_name, in_name).
    """

    def __init__(self, submats):
        """
        Initialize all attributes.
        """
        self._matrix = None
        self._submats = submats

    def _pre_update(self):
        """
        Do anything that needs to be done at the beginning of AssembledJacobian._update.
        """
        pass

    def _post_update(self):
        """
        Do anything that needs to be done at the end of AssembledJacobian._update.
        """
        pass

    def _update(self, randgen=None):
        """
        Update the matrix with the new sub-Jacobians.
        """
        self._pre_update()
        for subjac in self._submats.values():
            subjac._update(randgen)
        self._post_update()

    def set_complex_step_mode(self, active):
        """
        Turn on or off complex stepping mode.

        When turned on, the value in each subjac is cast as complex, and when turned
        off, they are returned to real values.

        Parameters
        ----------
        active : bool
            Complex mode flag; set to True prior to commencing complex step.
        """
        pass
