"""Define the LinearSchur class."""

import numpy as np

from openmdao.core.constants import _UNDEFINED
from openmdao.solvers.solver import LinearSolver
import scipy


class LinearSchur(LinearSolver):
    """
    Linear Schur complement solver.

    Parameters
    ----------
    mode_linear : str
        Specifies whether the solver is operating in forward ('fwd') or reverse ('rev') mode.
    sys_names : Sequence
        The two coupled systems being solved with this solver. If unspecified, LinearSchur
        will attempt to determine them automatically.
    **kwargs : dict
        Options dictionary.

    Attributes
    ----------
    _theta_n_1 : float
        Cached relaxation factor from previous iteration. Only used if the aitken acceleration
        option is turned on.
    _delta_d_n_1 : ndarray
        Cached change in the d_output vectors for the previous iteration. Only used if the
        aitken acceleration option is turned on.
    _mode_linear : str
        Specifies whether the solver is operating in forward ('fwd') or reverse ('rev') mode.
    _sys_names : str
        The two coupled systems being solved with this solver. If unspecified, LinearSchur
        will attempt to determine them automatically.
    """

    SOLVER = 'LN: LNSCHUR'

    def __init__(self, mode_linear='rev', sys_names=None, **kwargs):
        """
        Initialize all attributes.
        """
        super().__init__(**kwargs)

        self._theta_n_1 = None
        self._delta_d_n_1 = None
        self._mode_linear = mode_linear
        self._sys_names = sys_names

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        super()._declare_options()
        # this solver does not iterate
        self.options.undeclare('maxiter')
        self.options.undeclare('err_on_non_converge')

        self.options.undeclare('atol')
        self.options.undeclare('rtol')

    def _setup_solvers(self, system, depth):
        """
        Assign system instance, set depth, and optionally perform setup.

        Parameters
        ----------
        system : System
            pointer to the owning system.
        depth : int
            depth of the current system (already incremented).
        """
        super()._setup_solvers(system, depth)

        if self._sys_names is None:
            self._sys_names = [s for s in system._subsystems_allprocs.keys() if s != '_auto_ivc']

        if len(self._sys_names) != 2:
            raise ValueError(f'System {self.pathname} has a LinearSchur solver and is required to '
                             'contain two subsystems, but it has '
                             f'{len(self._sys_names)}.\n{self._sys_names}')

    def solve(self, mode, rel_systems=None):
        """
        Perform the operations in the iteration loop.

        Parameters
        ----------
        mode : str
            Specifies whether the solver is operating in forward ('fwd') or reverse ('rev') mode.
        rel_systems : set of str
            Set of names of relevant systems based on the current linear solve.  Deprecated.
        """
        self._mode = mode
        self._rel_systems = rel_systems
        system = self._system()
        mode = self._mode_linear

        if self._mode == 'fwd':
            self._rhs_vec = system._dresiduals.asarray(True)
        else:
            self._rhs_vec = system._doutputs.asarray(True)

        # take the subsystems
        subsys1, _ = system._subsystems_allprocs[self._sys_names[0]]
        subsys2, _ = system._subsystems_allprocs[self._sys_names[1]]

        # TODO this may not be the most general case. think about just solving for a subset
        subsys2_outputs = subsys2._doutputs
        subsys2_residuals = subsys2._dresiduals

        # list of variables we solve for here. this should include all variables in
        # subsys2 ideally because we dont do anything else for this subsystem here.
        vars_to_solve = [*subsys2_outputs.keys()]
        resd_to_solve = [*subsys2_residuals.keys()]

        # total size of the jacobian
        n_vars = 0
        for var in vars_to_solve:
            n_vars += subsys2_outputs[var].size

        # initialize the schur complement jacobian for these variables
        # TODO better way to get the dtype?
        schur_jac = np.zeros((n_vars, n_vars),
                             dtype=system._vectors['residual']['linear'].asarray(copy=False).dtype)
        schur_rhs = np.zeros((n_vars),
                             dtype=system._vectors['residual']['linear'].asarray(copy=False).dtype)

        # backup the vectors we are working with
        rvec = system._vectors['residual']['linear']
        ovec = system._vectors['output']['linear']
        ivec = system._vectors['input']['linear']

        # Cache a copy the linear residual of subsys2
        subsys2_linear_resid_array = subsys2._vectors['residual']['linear'].asarray(copy=True)

        if mode == 'fwd':
            parent_offset = system._dresiduals._root_offset

            # take the d_resdiuals for both of the subsys
            b_vec = subsys1._dresiduals
            off = b_vec._root_offset - parent_offset
            b_vec2 = subsys2._dresiduals
            off2 = b_vec2._root_offset - parent_offset

            # cache the rhs vector since we ll need this later
            subsys1_rhs = self._rhs_vec[off: off + len(b_vec)].copy()
            subsys2_rhs = self._rhs_vec[off2: off2 + len(b_vec2)].copy()

            # Beg solve for subsys 2
            # schur_jacobian

            # Schur_Jac = D - C A^-1 B ##

            # put back the vectors
            rvec.set_val(0.0)
            ovec.set_val(0.0)
            ivec.set_val(0.0)

            for ii, var in enumerate(vars_to_solve):
                # set the linear seed of the variable we want to solve for in subsys 2
                ovec[f'{subsys2.name}.{var}'] = 1.0

                # transfer this seed to the first subsystem
                system._transfer('linear', mode, subsys1.name)

                # run the jac-vec computation in the first subsystem,
                # this ll give us the B[:,{ii}] vector
                scope_out, scope_in = system._get_matvec_scope(subsys1)
                scope_out = self._vars_union(self._scope_out, scope_out)
                scope_in = self._vars_union(self._scope_in, scope_in)
                subsys1._apply_linear(None, mode, scope_out, scope_in)

                # amd then, by performing solve_linear we get A^-1 B[:,{ii}]
                subsys1._solve_linear(mode)

                # do another mat-mult with the solution of this
                # linear system, we want to get the final
                # jacobian using the schur method here, so we
                # will need to do a bit more math

                # first negate the vector from the linear solve
                subsys1._vectors['output']['linear'] *= -1.0

                # finally, set the seed of the variable to 1 as well
                # to get the diagonal contribution
                # system._vectors['output']['linear'][f'{subsys2.name}.{var}']
                # this should already be at one since we perturbed it above!

                # transfer the outputs to inputs
                system._transfer('linear', mode)

                # run the apply linear. we do it on the complete system here
                # the result is the final jacobian for this using the schur
                # complement method D[:,{ii}] - C A^-1 B[:,{ii}]
                scope_out, scope_in = system._get_matvec_scope()
                scope_out = self._vars_union(self._scope_out, scope_out)
                scope_in = self._vars_union(self._scope_in, scope_in)
                system._apply_linear(None, mode, scope_out, scope_in)

                # put this value into the jacobian.
                schur_jac[:, ii] = subsys2_linear_resid_array

                # set back the seed to zero for the next vector
                ovec[f'{subsys2.name}.{var}'] = 0.0
                # put back the vectors
                rvec.set_val(0.0)
                ovec.set_val(0.0)
                ivec.set_val(0.0)

            # schur_jacobian
            subsys1._vectors['residual']['linear'].set_val(subsys1_rhs)

            # now we work with the RHS
            scope_out, scope_in = system._get_matvec_scope(subsys1)
            scope_out = self._vars_union(self._scope_out, scope_out)
            scope_in = self._vars_union(self._scope_in, scope_in)
            subsys1._solve_linear(mode)
            system._transfer('linear', 'fwd', subsys2.name)

            scope_out, scope_in = system._get_matvec_scope(subsys2)
            scope_out = self._vars_union(self._scope_out, scope_out)
            scope_in = self._vars_union(self._scope_in, scope_in)
            subsys2._apply_linear(None, mode, scope_out, scope_in)

            schur_rhs = subsys2_linear_resid_array

            # Beg solve for subsys 2

            schur_rhs = subsys2_rhs - schur_rhs
            iD_schur = np.eye(n_vars,
                              dtype=system._vectors['residual']['linear'].asarray(copy=False).dtype)
            iD_schur *= 1e-16
            schur_jac = schur_jac + iD_schur
            d_subsys2 = scipy.linalg.solve(schur_jac, schur_rhs)

            for ii, var in enumerate(vars_to_solve):
                system._doutputs[f'{subsys2.name}.{var}'] = d_subsys2[ii]

            # End solve for subsys 2

            # Beg solve for subsys 1

            system._transfer('linear', mode, subsys1.name)

            scope_out, scope_in = system._get_matvec_scope(subsys1)
            scope_out = self._vars_union(self._scope_out, scope_out)
            scope_in = self._vars_union(self._scope_in, scope_in)

            subsys1._apply_linear(None, mode, scope_out, scope_in)
            b_vec *= -1.0
            b_vec += subsys1_rhs

            subsys1._solve_linear(mode)

            # End solve for subsys 1

        else:  # rev
            parent_offset = system._doutputs._root_offset

            # update the output of subsys2
            # system._transfer('linear', mode, subsys2.name)
            b_vec = subsys1._doutputs
            # b_vec_cache1 = subsys1._doutputs.asarray(copy=True)
            off = b_vec._root_offset - parent_offset

            b_vec2 = subsys2._doutputs
            # b_vec_cache2 = subsys2._doutputs.asarray(copy=True)
            off2 = b_vec2._root_offset - parent_offset

            subsys1_rhs = self._rhs_vec[off: off + len(b_vec)].copy()
            subsys2_rhs = self._rhs_vec[off2: off2 + len(b_vec2)].copy()

            # schur_jacobian

            # Schur_Jac = D - C A^-1 B ##

            rvec.set_val(np.zeros(len(rvec)))
            ovec.set_val(np.zeros(len(ovec)))
            ivec.set_val(np.zeros(len(ivec)))
            for ii, var in enumerate(resd_to_solve):
                # set the linear seed of the variable we want to solve for in subsys 2

                rvec[f'{subsys2.name}.{var}'] = 1.0

                # we get the C[{ii},:] vector by apply_linear on the system
                scope_out, scope_in = system._get_matvec_scope()
                scope_out = self._vars_union(self._scope_out, scope_out)
                scope_in = self._vars_union(self._scope_in, scope_in)
                system._apply_linear(None, mode, scope_out, scope_in)

                # do a solve_linear to find C[{ii},:] A^-1
                scope_out, scope_in = system._get_matvec_scope(subsys1)
                scope_out = self._vars_union(self._scope_out, scope_out)
                scope_in = self._vars_union(self._scope_in, scope_in)
                subsys1._solve_linear(mode, scope_out, scope_in)

                # negate the resdiual first
                subsys1._vectors['residual']['linear'] *= -1.0

                # do a apply_linear on the subsys1 to find the D[{ii},:] - C[{ii},:] A^-1 B
                scope_out, scope_in = system._get_matvec_scope(subsys1)
                subsys1._apply_linear(None, mode, scope_out, scope_in)

                system._transfer('linear', mode, subsys2.name)

                # put this value into the jacobian.
                schur_jac[:, ii] = subsys2._vectors['output']['linear'].asarray()

                # set back the seed to zero for the next vector
                rvec[f'{subsys2.name}.{var}'] = 0.0
                # set back the seed to zero for the next vector
                rvec.set_val(np.zeros(len(rvec)))
                ovec.set_val(np.zeros(len(ovec)))
                ivec.set_val(np.zeros(len(ivec)))

            # schur_jacobian
            subsys1._vectors['output']['linear'].set_val(subsys1_rhs)

            scope_out, scope_in = system._get_matvec_scope(subsys1)
            scope_out = self._vars_union(self._scope_out, scope_out)
            scope_in = self._vars_union(self._scope_in, scope_in)
            subsys1._solve_linear(mode, scope_out, scope_in)

            subsys1._apply_linear(None, mode, scope_out, scope_in)

            b_vec2.set_val(0.0)
            system._transfer('linear', mode, subsys2.name)

            # the same solve requires in the rhs too, so we save them
            schur_rhs = subsys2._vectors['output']['linear'].asarray(copy=True)

            # Beg solve for subsys 2

            b_vec2.set_val(0.0)

            system._transfer('linear', mode, subsys2.name)

            schur_rhs = subsys2_rhs - schur_rhs
            iD_schur = np.eye(n_vars,
                              dtype=system._vectors['residual']['linear'].asarray(copy=True).dtype)
            iD_schur *= 1e-16
            schur_jac = schur_jac + iD_schur
            d_subsys2 = scipy.linalg.solve(schur_jac, schur_rhs)

            # print(d_subsys2, schur_jac, schur_rhs)
            scope_out, scope_in = system._get_matvec_scope(subsys2)
            scope_out = self._vars_union(self._scope_out, scope_out)
            scope_in = self._vars_union(self._scope_in, scope_in)

            # loop over the variables just to be safe with the ordering
            for ii, var in enumerate(resd_to_solve):
                system._dresiduals[f'{subsys2.name}.{var}'] = d_subsys2[ii]

            subsys2._apply_linear(None, mode, scope_out, scope_in)
            # End solve for subsys 2

            # Beg solve for subsys 1
            b_vec.set_val(0.0)

            system._transfer('linear', mode, subsys1.name)

            b_vec *= -1.0
            # b_vec += subsys1_rhs

            b_vec += subsys1_rhs

            scope_out, scope_in = system._get_matvec_scope(subsys1)
            scope_out = self._vars_union(self._scope_out, scope_out)
            scope_in = self._vars_union(self._scope_in, scope_in)
            subsys1._solve_linear(mode, scope_out, scope_in)

            # if subsys1._iter_call_apply_linear():
            if subsys1._iter_call_apply_linear():
                subsys1._apply_linear(None, mode, scope_out, scope_in)
            else:
                b_vec.set_val(0.0)

            # End solve for subsys 1

        # reset after solve is done
        self._scope_in = self._scope_out = _UNDEFINED
