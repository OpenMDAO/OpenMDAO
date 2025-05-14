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
    _metadata : dict
        implementation-specific data for the sub-matrices.
    """

    def __init__(self, submats):
        """
        Initialize all attributes.
        """
        self._matrix = None
        self._submats = submats
        self._metadata = {}

    def _build(self, num_rows, num_cols, dtype=float):
        """
        Allocate the matrix.

        Parameters
        ----------
        num_rows : int
            number of rows in the matrix.
        num_cols : int
            number of cols in the matrix.
        dtype : dtype
            The dtype of the matrix.
        """
        pass

    def _update_submat(self, key, jac):
        """
        Update the values of a sub-jacobian.

        Parameters
        ----------
        key : (str, str)
            the global output and input variable names.
        jac : ndarray or scipy.sparse or tuple
            the sub-jacobian, the same format with which it was declared.
        """
        raise NotImplementedError("Matrix._update_submat")

    def _prod(self, vec, mode, mask=None):
        """
        Perform a matrix vector product.

        Parameters
        ----------
        vec : ndarray[:]
            incoming vector to multiply.
        mode : str
            'fwd' or 'rev'.
        mask : ndarray of type bool, or None
            Array used to mask out part of the input vector.

        Returns
        -------
        ndarray[:]
            vector resulting from the product.
        """
        raise NotImplementedError("Matrix._prod")

    def transpose(self):
        """
        Transpose the matrix.

        Returns
        -------
        sparse_matrix
            Transposed matrix.
        """
        raise NotImplementedError("Matrix.transpose")

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
