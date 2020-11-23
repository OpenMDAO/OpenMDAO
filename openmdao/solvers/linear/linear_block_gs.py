"""Define the LinearBlockGS class."""

import numpy as np

from openmdao.solvers.solver import BlockLinearSolver


class LinearBlockGS(BlockLinearSolver):
    """
    Linear block Gauss-Seidel solver.

    Attributes
    ----------
    _delta_d_n_1 : dict of ndarray
        Cached change in the d_output vectors for the previous iteration. Only used if the
        aitken acceleration option is turned on. The dictionary is keyed by linear vector name.
    _theta_n_1 : dict of float
        Cached relaxation factor from previous iteration. Only used if the aitken acceleration
        option is turned on. The dictionary is keyed by linear vector name.
    """

    SOLVER = 'LN: LNBGS'

    def __init__(self, **kwargs):
        """
        Initialize all attributes.

        Parameters
        ----------
        **kwargs : dict
            options dictionary.
        """
        super().__init__(**kwargs)

        self._theta_n_1 = {}
        self._delta_d_n_1 = {}

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        super()._declare_options()

        self.options.declare('use_aitken', types=bool, default=False,
                             desc='set to True to use Aitken relaxation')
        self.options.declare('aitken_min_factor', default=0.1,
                             desc='lower limit for Aitken relaxation factor')
        self.options.declare('aitken_max_factor', default=1.5,
                             desc='upper limit for Aitken relaxation factor')
        self.options.declare('aitken_initial_factor', default=1.0,
                             desc='initial value for Aitken relaxation factor')

    def _iter_initialize(self):
        """
        Perform any necessary pre-processing operations.

        Returns
        -------
        float
            initial error.
        float
            error at the first iteration.
        """
        if self.options['use_aitken']:
            system = self._system()
            if self._mode == 'fwd':
                d_vec = system._vectors['output']
            else:
                d_vec = system._vectors['residual']
            vec_names = self._vec_names
            for vec_name in vec_names:
                self._delta_d_n_1[vec_name] = d_vec[vec_name].asarray(copy=True)
                self._theta_n_1[vec_name] = 1.0

        return super()._iter_initialize()

    def _single_iteration(self):
        """
        Perform the operations in the iteration loop.
        """
        system = self._system()
        mode = self._mode
        vec_names = self._vec_names
        use_aitken = self.options['use_aitken']

        if use_aitken:
            aitken_min_factor = self.options['aitken_min_factor']
            aitken_max_factor = self.options['aitken_max_factor']

            # some variables that are used for Aitken's relaxation
            delta_d_n_1 = self._delta_d_n_1
            theta_n_1 = self._theta_n_1

            # store a copy of the outputs, used to compute the change in outputs later
            d_n = {}
            delta_d_n = {}
            if self._mode == 'fwd':
                d_out_vec = system._vectors['output']
            else:
                d_out_vec = system._vectors['residual']

            for vec_name in vec_names:
                d_n[vec_name] = d_out_vec[vec_name].asarray(copy=True)
                delta_d_n[vec_name] = d_out_vec[vec_name].asarray(copy=True)

        if mode == 'fwd':
            for subsys, _ in system._subsystems_allprocs.values():
                if self._rel_systems is not None and subsys.pathname not in self._rel_systems:
                    continue
                for vec_name in vec_names:
                    # must always do the transfer on all procs even if subsys not local
                    system._transfer(vec_name, mode, subsys.name)

                if not subsys._is_local:
                    continue

                scope_out, scope_in = system._get_scope(subsys)
                subsys._apply_linear(None, vec_names, self._rel_systems, mode, scope_out, scope_in)
                for vec_name in vec_names:
                    if vec_name in subsys._rel_vec_names:
                        b_vec = system._vectors['residual'][vec_name]
                        b_vec *= -1.0
                        b_vec += self._rhs_vecs[vec_name]
                subsys._solve_linear(vec_names, mode, self._rel_systems)

        else:  # rev
            subsystems = list(system._subsystems_allprocs)
            subsystems.reverse()
            for sname in subsystems:
                subsys, _ = system._subsystems_allprocs[sname]

                if self._rel_systems is not None and subsys.pathname not in self._rel_systems:
                    continue

                if subsys._is_local:
                    for vec_name in vec_names:
                        if vec_name in subsys._rel_vec_names:
                            b_vec = system._vectors['output'][vec_name]
                            b_vec.set_val(0.0)
                            system._transfer(vec_name, mode, sname)
                            b_vec *= -1.0
                            b_vec += self._rhs_vecs[vec_name]

                    subsys._solve_linear(vec_names, mode, self._rel_systems)
                    scope_out, scope_in = system._get_scope(subsys)
                    subsys._apply_linear(None, vec_names, self._rel_systems, mode,
                                         scope_out, scope_in)
                else:   # subsys not local
                    for vec_name in vec_names:
                        system._transfer(vec_name, mode, sname)

        if use_aitken:
            for vec_name in vec_names:
                if self._mode == 'fwd':
                    d_resid_vec = system._vectors['residual'][vec_name]
                    d_out_vec = system._vectors['output'][vec_name]
                else:
                    d_resid_vec = system._vectors['output'][vec_name]
                    d_out_vec = system._vectors['residual'][vec_name]

                theta_n = self.options['aitken_initial_factor']

                # compute the change in the outputs after the NLBGS iteration
                delta_d_n[vec_name] -= d_out_vec.asarray()
                delta_d_n[vec_name] *= -1

                if self._iter_count >= 2:
                    # Compute relaxation factor. This method is used by Kenway et al. in
                    # "Scalable Parallel Approach for High-Fidelity Steady-State Aero-
                    # elastic Analysis and Adjoint Derivative Computations" (ln 22 of Algo 1)

                    temp = delta_d_n[vec_name].copy()
                    temp -= delta_d_n_1[vec_name]

                    # If MPI, piggyback on the residual vector to perform a distributed norm.
                    if system.comm.size > 1:
                        backup_r = d_resid_vec.asarray(copy=True)
                        d_resid_vec.set_val(temp)
                        temp_norm = d_resid_vec.get_norm()
                    else:
                        temp_norm = np.linalg.norm(temp)

                    if temp_norm == 0.:
                        temp_norm = 1e-12  # prevent division by 0 below

                    # If MPI, piggyback on the output and residual vectors to perform a distributed
                    # dot product.
                    if system.comm.size > 1:
                        backup_o = d_out_vec.asarray(copy=True)
                        d_out_vec.set_val(delta_d_n[vec_name])
                        tddo = d_resid_vec.dot(d_out_vec)
                        d_resid_vec.set_val(backup_r)
                        d_out_vec.set_val(backup_o)
                    else:
                        tddo = temp.dot(delta_d_n[vec_name])

                    theta_n = theta_n_1[vec_name] * (1 - tddo / temp_norm ** 2)

                else:
                    # keep the initial the relaxation factor
                    pass

                # limit relaxation factor to the specified range
                theta_n = max(aitken_min_factor, min(aitken_max_factor, theta_n))

                # save relaxation factor for the next iteration
                self._theta_n_1[vec_name] = theta_n

                d_out_vec.set_val(d_n[vec_name])

                # compute relaxed outputs
                d_out_vec += theta_n * delta_d_n[vec_name]

                # save update to use in next iteration
                delta_d_n_1[vec_name][:] = delta_d_n[vec_name]
