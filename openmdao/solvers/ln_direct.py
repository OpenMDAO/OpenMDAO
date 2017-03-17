"""LinearSolver that uses linalg.solve or LU factor/solve."""

from __future__ import division, print_function

from six.moves import range

import numpy as np
import scipy.linalg
import scipy.sparse.linalg

from openmdao.solvers.solver import LinearSolver
from openmdao.matrices.coo_matrix import COOMatrix
from openmdao.matrices.csr_matrix import CSRMatrix
from openmdao.matrices.dense_matrix import DenseMatrix


class DirectSolver(LinearSolver):
    """
    LinearSolver that uses linalg.solve or LU factor/solve.

    Attributes
    ----------
    _print_name : str ('Direct')
        print name.
    """

    SOLVER = 'LN: Direct'

    def __init__(self, **kwargs):
        """
        Declare the solver option.

        Parameters
        ----------
        **kwargs : {}
            dictionary of options set by the instantiating class/script.
        """
        super(DirectSolver, self).__init__(**kwargs)

        self._print_name = 'Direct'

    def _linearize(self):
        """
        Perform factorization.
        """
        system = self._system

        if system._owns_global_jac:
            mtx = system._jacobian._int_mtx
            # Perform dense or sparse lu factorization
            if isinstance(mtx, DenseMatrix):
                np.set_printoptions(precision=3)
                self._lup = scipy.linalg.lu_factor(mtx._matrix)
            elif isinstance(mtx, (COOMatrix, CSRMatrix)):
                np.set_printoptions(precision=3)
                self._lu = scipy.sparse.linalg.splu(mtx._matrix)
            else:
                raise RuntimeError('Direct solver not implemented for mtx type %s in system %s'
                                   % (type(mtx), system.pathname))

        else:
            # First make a backup of the vectors
            b_data = system._vectors['residual']['linear'].get_data()
            x_data = system._vectors['output']['linear'].get_data()

            # Assemble the Jacobian by running the identity matrix through apply_linear
            nmtx = system._vectors['output']['linear'].get_data().size
            eye = np.eye(nmtx)
            mtx = np.empty((nmtx, nmtx))
            for i in range(nmtx):
                mtx[:, i] = self._mat_vec(eye[:, i])

            # Restore the backed-up vectors
            system._vectors['residual']['linear'].set_data(b_data)
            system._vectors['output']['linear'].set_data(x_data)

            self._lup = scipy.linalg.lu_factor(mtx)

    def _mat_vec(self, in_vec):
        """
        Compute matrix-vector product.

        Parameters
        ----------
        in_vec : ndarray
            the incoming array (combines all varsets).

        Returns
        -------
        ndarray
            the outgoing array after the product (combines all varsets).
        """
        vec_name = 'linear'
        system = self._system

        # assign x and b vectors based on fwd mode
        x_vec = system._vectors['output'][vec_name]
        b_vec = system._vectors['residual'][vec_name]

        # set value of x vector to provided value
        x_vec.set_data(in_vec)

        # apply linear
        ind1, ind2 = system._var_allprocs_idx_range['output']
        var_inds = [ind1, ind2, ind1, ind2]
        system._apply_linear([vec_name], 'fwd', var_inds)

        # return result
        return b_vec.get_data()

    def solve(self, vec_names, mode):
        """
        Run the solver.

        Parameters
        ----------
        vec_names : [str, ...]
            list of names of the right-hand-side vectors.
        mode : str
            'fwd' or 'rev'.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        float
            absolute error.
        float
            relative error.
        """
        self._vec_names = vec_names
        self._mode = mode

        system = self._system

        for vec_name in self._vec_names:
            self._vec_name = vec_name

            # assign x and b vectors based on mode
            if self._mode == 'fwd':
                x_vec = system._vectors['output'][vec_name]
                b_vec = system._vectors['residual'][vec_name]
                trans_lu = 0
                trans_splu = 'N'
            elif self._mode == 'rev':
                x_vec = system._vectors['residual'][vec_name]
                b_vec = system._vectors['output'][vec_name]
                trans_lu = 1
                trans_splu = 'T'

            b_data = b_vec.get_data()
            if system._owns_global_jac and isinstance(system._jacobian._int_mtx,
                                                      (COOMatrix, CSRMatrix)):
                x_data = self._lu.solve(b_data, trans_splu)
            else:
                x_data = scipy.linalg.lu_solve(self._lup, b_data, trans=trans_lu)
            x_vec.set_data(x_data)

        return False, 0., 0.
