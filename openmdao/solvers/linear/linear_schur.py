"""Define the LinearBlockGS class."""

import sys
import numpy as np

from openmdao.core.constants import _UNDEFINED
from openmdao.solvers.solver import BlockLinearSolver
from openmdao.utils.general_utils import ContainsAll
import scipy


class LinearSchur(BlockLinearSolver):
    """
    Linear block Gauss-Seidel solver.

    Parameters
    ----------
    **kwargs : dict
        Options dictionary.

    Attributes
    ----------
    _delta_d_n_1 : ndarray
        Cached change in the d_output vectors for the previous iteration. Only used if the
        aitken acceleration option is turned on.
    _theta_n_1 : float
        Cached relaxation factor from previous iteration. Only used if the aitken acceleration
        option is turned on.
    """

    SOLVER = "LN: Schur"

    def __init__(self, mode_linear="rev", groupNames=["group1", "group2"], **kwargs):
        """
        Initialize all attributes.
        """
        super().__init__(**kwargs)

        self._theta_n_1 = None
        self._delta_d_n_1 = None
        self._mode_linear = mode_linear
        self._groupNames = groupNames

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        super()._declare_options()

        self.options.declare("use_aitken", types=bool, default=False, desc="set to True to use Aitken relaxation")
        self.options.declare("aitken_min_factor", default=0.1, desc="lower limit for Aitken relaxation factor")
        self.options.declare("aitken_max_factor", default=1.5, desc="upper limit for Aitken relaxation factor")
        self.options.declare("aitken_initial_factor", default=1.0, desc="initial value for Aitken relaxation factor")

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
        if self.options["use_aitken"]:
            if self._mode_linear == "fwd":
                self._delta_d_n_1 = self._system()._doutputs.asarray(copy=True)
            else:
                self._delta_d_n_1 = self._system()._dresiduals.asarray(copy=True)
            self._theta_n_1 = 1.0

        return super()._iter_initialize()

    def _single_iteration(self):
        """
        Perform the operations in the iteration loop.
        """
        system = self._system()
        mode = self._mode_linear
        use_aitken = self.options["use_aitken"]

        if use_aitken:
            aitken_min_factor = self.options["aitken_min_factor"]
            aitken_max_factor = self.options["aitken_max_factor"]

            # some variables that are used for Aitken's relaxation
            delta_d_n_1 = self._delta_d_n_1
            theta_n_1 = self._theta_n_1

            # store a copy of the outputs, used to compute the change in outputs later
            if self._mode_linear == "fwd":
                d_out_vec = system._doutputs
            else:
                d_out_vec = system._dresiduals

            d_n = d_out_vec.asarray(copy=True)
            delta_d_n = d_out_vec.asarray(copy=True)

        subsys1, _ = system._subsystems_allprocs[self._groupNames[0]]
        subsys2, _ = system._subsystems_allprocs[self._groupNames[1]]
        # print(system._doutputs.keys())

        # TODO this may not be the most general case. think about just solving for a subset
        subsys2_outputs = subsys2._outputs
        subsys2_residuals = subsys2._residuals
        # list of variables we solve for here. this should include all variables in
        # subsys2 ideally because we dont do anything else for this subsystem here
        vars_to_solve = [*subsys2_outputs.keys()]

        resd_to_solve = [*subsys2_residuals.keys()]
        # print(vars_to_solve)

        # total size of the jacobian
        n_vars = 0
        for var in vars_to_solve:
            n_vars += subsys2_outputs[var].size

        # initialize the schur complement jacobian for these variables
        # TODO better way to get the dtype?
        schur_jac = np.zeros((n_vars, n_vars), dtype=system._vectors["residual"]["linear"].asarray(copy=True).dtype)

        subsystem_list = [subsys1, subsys2]
        print("Linear FWD", system._dresiduals, system._doutputs, system._dinputs)
        if mode == "fwd":
            print("Linear FWD", system._dresiduals._root_offset)
            parent_offset = system._dresiduals._root_offset

            # for subsys in subsystem_list:
            if self._rel_systems is not None and subsys1.pathname not in self._rel_systems:
                return
            # must always do the transfer on all procs even if subsys not local

            # subsys1._vectors["input"]["linear"] = subsys2._vectors["output"]["linear"]
            system._transfer("linear", mode, subsys1.name)

            if not subsys1._is_local:
                return

            b_vec = subsys1._dresiduals
            b_vec_cache_1 = subsys1._dresiduals
            print(b_vec, "bvec")
            scope_out, scope_in = system._get_matvec_scope(subsys1)
            # we use _vars_union to combine relevant variables from the current solve
            # with those of the subsystem solve, because for recursive block linear solves
            # we'll be skipping a direct call to _apply_linear and instead counting on
            # _apply_linear to be called once at the bottom of the recursive block linear
            # solve on the component, using the full set of relevant variables from the
            # top group in the block linear solve and all intervening groups (assuming all
            # of those groups are doing block linear solves).
            scope_out = self._vars_union(self._scope_out, scope_out)
            scope_in = self._vars_union(self._scope_in, scope_in)
            off = b_vec._root_offset - parent_offset
            if subsys1._iter_call_apply_linear():
                subsys1._apply_linear(None, self._rel_systems, mode, scope_out, scope_in)

                b_vec *= -1.0
                b_vec += self._rhs_vec[off : off + len(b_vec)]
            else:
                b_vec.set_val(self._rhs_vec[off : off + len(b_vec)])

            print("subsys1 solve", subsys1._vectors["residual"]["linear"].asarray())
            print("subsys1 solve", subsys1._vectors["output"]["linear"].asarray())
            print("subsys1 solve", subsys1._vectors["input"]["linear"].asarray())

            subsys1._solve_linear(mode, self._rel_systems, scope_out, scope_in)

            # print("subsys1 solve", subsys1._vectors["residual"]["linear"].asarray())
            # print("subsys1 solve", subsys1._vectors["output"]["linear"].asarray())
            # print("subsys1 solve", subsys1._vectors["input"]["linear"].asarray())
            print("subsys1 F", subsys1._vectors["residual"]["linear"].asarray())
            print("subsys1", subsys1._vectors["output"]["linear"].asarray())
            print("subsys1", subsys1._vectors["input"]["linear"].asarray())
            ##### Subsys2 #####

            # for subsys2 in subsystem_list:
            if self._rel_systems is not None and subsys2.pathname not in self._rel_systems:
                return
            # must always do the transfer on all procs even if subsys not local
            system._transfer("linear", mode, subsys2.name)

            # print("subsys1", subsys1._vectors["residual"]["linear"].asarray())
            # print("subsys1", subsys1._vectors["output"]["linear"].asarray())
            # print("subsys1", subsys1._vectors["input"]["linear"].asarray())

            # print("subsys2", subsys2._vectors["residual"]["linear"].asarray())
            # print("subsys2", subsys2._vectors["output"]["linear"].asarray())
            # print("subsys2", subsys2._vectors["input"]["linear"].asarray())

            if not subsys2._is_local:
                return

            b_vec2 = subsys2._dresiduals

            scope_out, scope_in = system._get_matvec_scope(subsys2)
            # we use _vars_union to combine relevant variables from the current solve
            # with those of the subsystem solve, because for recursive block linear solves
            # we'll be skipping a direct call to _apply_linear and instead counting on
            # _apply_linear to be called once at the bottom of the recursive block linear
            # solve on the component, using the full set of relevant variables from the
            # top group in the block linear solve and all intervening groups (assuming all
            # of those groups are doing block linear solves).
            scope_out = self._vars_union(self._scope_out, scope_out)
            scope_in = self._vars_union(self._scope_in, scope_in)
            off = b_vec2._root_offset - parent_offset

            ########################
            #### schur_jacobian ####
            ########################
            rvec = system._vectors["residual"]["linear"]
            ovec = system._vectors["output"]["linear"]
            ivec = system._vectors["input"]["linear"]

            r_data = rvec.asarray(copy=True)
            o_data = ovec.asarray(copy=True)
            i_data = ivec.asarray(copy=True)
            # print(rvec, ovec, ivec)
            # set the ovec to zeros
            ovec.set_val(np.zeros(len(ovec)))
            # print(subsys2._outputs.keys())
            for ii, var in enumerate(vars_to_solve):
                # set the linear seed of the variable we want to solve for in subsys 2
                ovec[f"{subsys2.name}.{var}"] = 1.0

                # transfer this seed to the first subsystem
                system._transfer("linear", mode, subsys1.name)

                # run the jac-vec computation in the first subsystem
                scope_out, scope_in = system._get_matvec_scope(subsys1)
                print("subsys1", subsys1._vectors["residual"]["linear"].asarray())
                print("subsys1", subsys1._vectors["output"]["linear"].asarray())
                print("subsys1", subsys1._vectors["input"]["linear"].asarray())
                subsys1._apply_linear(None, self._rel_systems, mode, scope_out, scope_in)

                if system.comm.rank == 0:
                    print(f"\nComputing Jacobian columns for {var}")
                    print(f"B[:,{ii}]                    =", subsys1._vectors["output"]["linear"].asarray())
                    print(
                        f"|B[:,{ii}]|                  =",
                        np.linalg.norm(subsys1._vectors["output"]["linear"].asarray()),
                        flush=True,
                    )

                # using the result from this jac-vec product, solve the RHS for this subsystem
                print("subsys1", subsys1._vectors["residual"]["linear"].asarray())
                print("subsys1", subsys1._vectors["output"]["linear"].asarray())
                print("subsys1", subsys1._vectors["input"]["linear"].asarray())
                subsys1._solve_linear(mode, self._rel_systems, ContainsAll())
                if system.comm.rank == 0:
                    print(f"A^-1 B[:,{ii}]               =", subsys1._vectors["output"]["linear"].asarray())
                    print(
                        f"|A^-1 B[:,{ii}]|             =",
                        np.linalg.norm(subsys1._vectors["output"]["linear"].asarray()),
                        flush=True,
                    )

                # do another mat-mult with the solution of this linear system, we want to get the final
                # jacobian using the schur method here, so we will need to do a bit more math

                # first negate the vector from the linear solve
                subsys1._vectors["output"]["linear"] *= -1.0
                if system.comm.rank == 0:
                    print("seed for C | D            =", system._vectors["output"]["linear"].asarray(), flush=True)

                # finally, set the seed of the variable to 1 as well to get the diagonal contribution
                # system._vectors["output"]["linear"][f"{subsys2.name}.{var}"]
                # this should already be at one since we perturbed it above!

                # transfer the outputs to inputs
                system._transfer("linear", mode)

                # run the apply linear. we do it on the complete system here
                scope_out, scope_in = system._get_matvec_scope()
                system._apply_linear(None, self._rel_systems, mode, scope_out, scope_in)

                # the result is the final jacobian for this using the schur complement method
                if system.comm.rank == 0:
                    print(
                        f"D[:,{ii}] - C A^-1 B[:,{ii}]    =",
                        subsys2._vectors["residual"]["linear"].asarray(),
                        flush=True,
                    )

                # quit()

                # put this value into the jacobian.
                schur_jac[:, ii] = subsys2._vectors["residual"]["linear"].asarray()

                # set back the seed to zero for the next vector
                ovec[f"{subsys2.name}.{var}"] = 0.0

            # rvec.set_val(r_data)
            # ovec.set_val(o_data)
            # ivec.set_val(i_data)
            off1 = b_vec._root_offset - parent_offset
            b_vec.set_val(self._rhs_vec[off1 : off1 + len(b_vec)])
            b_vec2.set_val(self._rhs_vec[off : off + len(b_vec2)])

            # b_vec.set_val(self._rhs_vec[off : off + len(b_vec)])
            # subsys1._dresiduals = b_vec_cache_1
            # subsys2._dresiduals = b_vec2
            print("dres", b_vec2, self._rhs_vec, b_vec)
            ########################
            #### schur_jacobian ####
            ########################
            # if subsys2._iter_call_apply_linear():
            # scope_out, scope_in = system._get_matvec_scope(subsys1)
            # print("subsys1", subsys1._vectors["residual"]["linear"].asarray())
            # print("subsys1", subsys1._vectors["output"]["linear"].asarray())
            # print("subsys1", subsys1._vectors["input"]["linear"].asarray())

            # print("subsys2", subsys2._vectors["residual"]["linear"].asarray())
            # print("subsys2", subsys2._vectors["output"]["linear"].asarray())
            # print("subsys2", subsys2._vectors["input"]["linear"].asarray())
            subsys1._solve_linear(mode, "linear", ContainsAll())
            # print("subsys1", subsys1._vectors["residual"]["linear"].asarray())
            # print("subsys1", subsys1._vectors["output"]["linear"].asarray())
            # print("subsys1", subsys1._vectors["input"]["linear"].asarray())

            # print("subsys2", subsys2._vectors["residual"]["linear"].asarray())
            # print("subsys2", subsys2._vectors["output"]["linear"].asarray())
            # print("subsys2", subsys2._vectors["input"]["linear"].asarray())
            # # first negate the vector from the linear solve
            subsys1._vectors["output"]["linear"] *= -1.0
            # # transfer the outputs to inputs
            # system._transfer("linear", "fwd")

            scope_out, scope_in = system._get_matvec_scope()
            scope_out = self._vars_union(self._scope_out, scope_out)
            scope_in = self._vars_union(self._scope_in, scope_in)
            system._apply_linear(None, self._rel_systems, mode, scope_out, scope_in)
            # print("subsys2", subsys2._vectors["residual"]["linear"].asarray())
            # print("subsys2", subsys2._vectors["output"]["linear"].asarray())
            # print("subsys2", subsys2._vectors["input"]["linear"].asarray())
            # b_vec2 = scipy.linalg.solve(schur_jac, subsys2._vectors["residual"]["linear"].asarray())
            # print("dres", b_vec2, self._rhs_vec)
            b_vec2 += self._rhs_vec[off : off + len(b_vec2)]
            # print("dres", b_vec2, self._rhs_vec)
            # print("subsys2", subsys2._vectors["residual"]["linear"].asarray())
            # print("subsys2", subsys2._vectors["output"]["linear"].asarray())
            # print("subsys2", subsys2._vectors["input"]["linear"].asarray())
            # # subsys2._apply_linear(None, self._rel_systems, mode, scope_out, scope_in)
            # b_vec2 *= -1.0
            # b_vec2 = subsys2._dresiduals
            # print("b2", subsys2._vectors["residual"]["linear"].asarray())
            # else:
            # b_vec2.set_val(self._rhs_vec[off : off + len(b_vec2)])
            # quit()
            # subsys2._solve_linear(mode, self._rel_systems, scope_out, scope_in)
            d_subsys2 = scipy.linalg.solve(schur_jac, subsys2._vectors["residual"]["linear"].asarray())
            # loop over the variables just to be safe with the ordering
            for ii, var in enumerate(vars_to_solve):
                system._doutputs[f"{subsys2.name}.{var}"] += d_subsys2[ii]

        else:  # rev
            if sys.version_info >= (3, 8):
                subsystems = reversed(system._subsystems_allprocs.values())
            else:
                subsystems = list(system._subsystems_allprocs.values())
                subsystems.reverse()
            parent_offset = system._doutputs._root_offset

            subsys1, _ = subsystems["coupling_2"]
            subsys2, _ = subsystems["balance"]

            for subsys, _ in subsystems:
                if self._rel_systems is not None and subsys.pathname not in self._rel_systems:
                    continue

                if subsys._is_local:
                    b_vec = subsys._doutputs
                    b_vec.set_val(0.0)

                    system._transfer("linear", mode, subsys.name)

                    b_vec *= -1.0
                    off = b_vec._root_offset - parent_offset
                    b_vec += self._rhs_vec[off : off + len(b_vec)]

                    scope_out, scope_in = system._get_matvec_scope(subsys)
                    scope_out = self._vars_union(self._scope_out, scope_out)
                    scope_in = self._vars_union(self._scope_in, scope_in)

                    subsys._solve_linear(mode, self._rel_systems, scope_out, scope_in)

                    if subsys._iter_call_apply_linear():
                        subsys._apply_linear(None, self._rel_systems, mode, scope_out, scope_in)
                    else:
                        b_vec.set_val(0.0)
                else:  # subsys not local
                    system._transfer("linear", mode, subsys.name)

        if use_aitken:
            if self._mode_linear == "fwd":
                d_resid_vec = system._dresiduals
                d_out_vec = system._doutputs
            else:
                d_resid_vec = system._doutputs
                d_out_vec = system._dresiduals

            theta_n = self.options["aitken_initial_factor"]

            # compute the change in the outputs after the NLBGS iteration
            delta_d_n -= d_out_vec.asarray()
            delta_d_n *= -1

            if self._iter_count >= 2:
                # Compute relaxation factor. This method is used by Kenway et al. in
                # "Scalable Parallel Approach for High-Fidelity Steady-State Aero-
                # elastic Analysis and Adjoint Derivative Computations" (ln 22 of Algo 1)

                temp = delta_d_n.copy()
                temp -= delta_d_n_1

                # If MPI, piggyback on the residual vector to perform a distributed norm.
                if system.comm.size > 1:
                    backup_r = d_resid_vec.asarray(copy=True)
                    d_resid_vec.set_val(temp)
                    temp_norm = d_resid_vec.get_norm()
                else:
                    temp_norm = np.linalg.norm(temp)

                if temp_norm == 0.0:
                    temp_norm = 1e-12  # prevent division by 0 below

                # If MPI, piggyback on the output and residual vectors to perform a distributed
                # dot product.
                if system.comm.size > 1:
                    backup_o = d_out_vec.asarray(copy=True)
                    d_out_vec.set_val(delta_d_n)
                    tddo = d_resid_vec.dot(d_out_vec)
                    d_resid_vec.set_val(backup_r)
                    d_out_vec.set_val(backup_o)
                else:
                    tddo = temp.dot(delta_d_n)

                theta_n = theta_n_1 * (1 - tddo / temp_norm**2)

            else:
                # keep the initial the relaxation factor
                pass

            # limit relaxation factor to the specified range
            theta_n = max(aitken_min_factor, min(aitken_max_factor, theta_n))

            # save relaxation factor for the next iteration
            self._theta_n_1 = theta_n

            d_out_vec.set_val(d_n)

            # compute relaxed outputs
            d_out_vec += theta_n * delta_d_n

            # save update to use in next iteration
            delta_d_n_1[:] = delta_d_n
