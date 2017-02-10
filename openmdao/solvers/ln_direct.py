"""LinearSolver that uses linalg.solve or LU factor/solve."""

from __future__ import division, print_function

import numpy

from scipy.linalg import lu_factor, lu_solve

from openmdao.solvers.solver import LinearSolver


class DirectSolver(LinearSolver):
    """LinearSolver that uses linalg.solve or LU factor/solve.

    Attributes
    ----------
    _print_name : str ('Direct')
        print name.
    """

    SOLVER = 'LN: Direct'

    def __init__(self, **kwargs):
        """Declare the solver option.

        Parameters
        ----------
        kwargs : {}
            dictionary of options set by the instantiating class/script.
        """
        super(DirectSolver, self).__init__(**kwargs)

        self._print_name = 'Direct'

    def _declare_options(self):
        """Declare options before kwargs are processed in the init method."""
        self.options.declare('method', value='solve', values=['LU', 'solve'],
                             desc="Solution method, either 'solve' for " +
                             "linalg.solve, or 'LU' for linalg.lu_factor " +
                             "and linalg.lu_solve.")

    def _mat_vec(self, in_vec):
        """Compute matrix-vector product.

        Parameters
        ----------
        in_vec : ndarray
            the incoming array (combines all varsets).

        Returns
        -------
        ndarray
            the outgoing array after the product (combines all varsets).
        """
        # assign x and b vectors based on mode
        vec_name = self._vec_name
        system = self._system

        if self._mode == 'fwd':
            x_vec = system._vectors['output'][vec_name]
            b_vec = system._vectors['residual'][vec_name]
        elif self._mode == 'rev':
            x_vec = system._vectors['residual'][vec_name]
            b_vec = system._vectors['output'][vec_name]

        # set value of x vector to provided value
        x_vec.set_data(in_vec)

        # apply linear
        ind1, ind2 = system._var_allprocs_range['output']
        var_inds = [ind1, ind2, ind1, ind2]
        system._apply_linear([vec_name], self._mode, var_inds)

        # return result
        return b_vec.get_data()

    def solve(self, vec_names, mode):
        """See LinearSolver."""
        self._vec_names = vec_names
        self._mode = mode

        system = self._system

        for vec_name in self._vec_names:
            self._vec_name = vec_name

            # assign x and b vectors based on mode
            if self._mode == 'fwd':
                x_vec = system._vectors['output'][vec_name]
                b_data = system._vectors['residual'][vec_name].get_data()
            elif self._mode == 'rev':
                x_vec = system._vectors['residual'][vec_name]
                b_data = system._vectors['output'][vec_name].get_data()

            # assemble jacobian
            n_edge = b_data.size
            ident = numpy.eye(n_edge)
            jacobian = numpy.empty((n_edge, n_edge))
            for i in range(n_edge):
                jacobian[:, i] = self._mat_vec(ident[:, i])

            # solve
            if self.options['method'] == 'LU':
                lup = lu_factor(jacobian)
                result = lu_solve(lup, b_data)
            else:
                result = numpy.linalg.solve(jacobian, b_data)

            x_vec.set_data(result)
