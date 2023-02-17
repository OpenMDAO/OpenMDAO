"""Define the NewtonSolver class."""


import numpy as np

from openmdao.solvers.linesearch.backtracking import BoundsEnforceLS
from openmdao.solvers.solver import NonlinearSolver
from openmdao.recorders.recording_iteration_stack import Recording
from openmdao.utils.mpi import MPI
from openmdao.utils.general_utils import ContainsAll
import scipy


class SchurSolver(NonlinearSolver):
    """
    Newton solver.

    The default linear solver is the linear_solver in the containing system.

    Attributes
    ----------
    linear_solver : LinearSolver
        Linear solver to use to find the Newton search direction. The default
        is the parent system's linear solver.
    linesearch : NonlinearSolver
        Line search algorithm. Default is None for no line search.
    """

    SOLVER = "NL: Schur"

    def __init__(self, **kwargs):
        """
        Initialize all attributes.

        Parameters
        ----------
        **kwargs : dict
            options dictionary.
        """
        super().__init__(**kwargs)

        # Slot for linear solver
        self.linear_solver = None

        # Slot for linesearch
        self.linesearch = BoundsEnforceLS()

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        super()._declare_options()

        self.options.declare("solve_subsystems", types=bool, desc="Set to True to turn on sub-solvers (Hybrid Newton).")
        self.options.declare("max_sub_solves", types=int, default=10, desc="Maximum number of subsystem solves.")
        self.options.declare(
            "cs_reconverge",
            types=bool,
            default=True,
            desc="When True, when this driver solves under a complex step, nudge "
            "the Solution vector by a small amount so that it reconverges.",
        )
        self.options.declare(
            "reraise_child_analysiserror",
            types=bool,
            default=False,
            desc="When the option is true, a solver will reraise any "
            "AnalysisError that arises during subsolve; when false, it will "
            "continue solving.",
        )

        self.options.declare("mode", default=None, types=str)

        self.supports["gradients"] = True
        self.supports["implicit_components"] = True
        mode = self.options["mode"]
        print(mode)
        if mode is None:
            self._modeNS = "rev"
        else:
            if mode not in ["fwd", "rev"]:
                msg = f"{self.msginfo}: Unsupported mode: '{mode}'. Use either 'fwd' or 'rev'."
                raise ValueError(msg)
            self._modeNS = mode
        print(self._modeNS)

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
        rank = MPI.COMM_WORLD.rank if MPI is not None else 0

        self._disallow_discrete_outputs()

        if not isinstance(self.options._dict["solve_subsystems"]["val"], bool):
            msg = "{}: solve_subsystems must be set by the user."
            raise ValueError(msg.format(self.msginfo))

        if self.linear_solver is not None:
            self.linear_solver._setup_solvers(system, self._depth + 1)
        else:
            self.linear_solver = system.linear_solver

        if self.linesearch is not None:
            self.linesearch._setup_solvers(system, self._depth + 1)

    def _assembled_jac_solver_iter(self):
        """
        Return a generator of linear solvers using assembled jacs.
        """
        if self.linear_solver is not None:
            for s in self.linear_solver._assembled_jac_solver_iter():
                yield s

    def _set_solver_print(self, level=2, type_="all"):
        """
        Control printing for solvers and subsolvers in the model.

        Parameters
        ----------
        level : int
            iprint level. Set to 2 to print residuals each iteration; set to 1
            to print just the iteration totals; set to 0 to disable all printing
            except for failures, and set to -1 to disable all printing including failures.
        type_ : str
            Type of solver to set: 'LN' for linear, 'NL' for nonlinear, or 'all' for all.
        """
        super()._set_solver_print(level=level, type_=type_)

        if self.linear_solver is not None and type_ != "NL":
            self.linear_solver._set_solver_print(level=level, type_=type_)

        if self.linesearch is not None:
            self.linesearch._set_solver_print(level=level, type_=type_)

    def _run_apply(self):
        """
        Run the apply_nonlinear method on the system.
        """
        self._recording_iter.push(("_run_apply", 0))

        system = self._system()

        # Disable local fd
        approx_status = system._owns_approx_jac
        system._owns_approx_jac = False

        try:
            system._apply_nonlinear()
        finally:
            self._recording_iter.pop()

        # Enable local fd
        system._owns_approx_jac = approx_status

    def _linearize_children(self):
        """
        Return a flag that is True when we need to call linearize on our subsystems' solvers.

        Returns
        -------
        boolean
            Flag for indicating child linerization
        """
        return self.options["solve_subsystems"] and self._iter_count <= self.options["max_sub_solves"]

    def _linearize(self):
        """
        Perform any required linearization operations such as matrix factorization.
        """
        if self.linear_solver is not None:
            self.linear_solver._linearize()

        if self.linesearch is not None:
            self.linesearch._linearize()

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
        system = self._system()

        if self.options["debug_print"]:
            self._err_cache["inputs"] = system._inputs._copy_views()
            self._err_cache["outputs"] = system._outputs._copy_views()

        # When under a complex step from higher in the hierarchy, sometimes the step is too small
        # to trigger reconvergence, so nudge the outputs slightly so that we always get at least
        # one iteration of Newton.
        if system.under_complex_step and self.options["cs_reconverge"]:
            system._outputs += np.linalg.norm(system._outputs.asarray()) * 1e-10

        # Execute guess_nonlinear if specified.
        system._guess_nonlinear()

        with Recording("Newton_subsolve", 0, self):
            if self.options["solve_subsystems"] and (self._iter_count <= self.options["max_sub_solves"]):

                self._solver_info.append_solver()

                # should call the subsystems solve before computing the first residual
                self._gs_iter()

                self._solver_info.pop()

        self._run_apply()
        norm = self._iter_get_norm()

        norm0 = norm if norm != 0.0 else 1.0
        return norm0, norm

    def _single_iteration(self):
        """
        Perform the operations in the iteration loop.
        """
        system = self._system()
        self._solver_info.append_subsolver()
        do_subsolve = self.options["solve_subsystems"] and (self._iter_count < self.options["max_sub_solves"])
        do_sub_ln = self.linear_solver._linearize_children()

        # Disable local fd
        approx_status = system._owns_approx_jac
        system._owns_approx_jac = False

        system._vectors["residual"]["linear"].set_vec(system._residuals)
        # system._vectors["output"]["linear"].set_vec(system._outputs)
        # system._vectors["input"]["linear"].set_vec(system._inputs)
        system._vectors["residual"]["linear"] *= -1.0
        my_asm_jac = self.linear_solver._assembled_jac

        system._linearize(my_asm_jac, sub_do_ln=do_sub_ln)
        if my_asm_jac is not None and system.linear_solver._assembled_jac is not my_asm_jac:
            my_asm_jac._update(system)

        # print("\nFull Jacobian:")
        # print(my_asm_jac._int_mtx._matrix.todense())
        # print(my_asm_jac["group1.comp1.x1", "group1.comp1.x1"], my_asm_jac["group1.comp1.x1", "group1.comp1.x2"])
        # print(my_asm_jac["group2.comp2.x2", "group2.comp2.x1"], my_asm_jac["group2.comp2.x2", "group2.comp2.x2"])

        # J11 = my_asm_jac["group1.comp1.x1", "group1.comp1.x1"][0]
        # J12 = my_asm_jac["group1.comp1.x1", "group1.comp1.x2"][0]
        # J21 = my_asm_jac["group2.comp2.x2", "group2.comp2.x1"][0]
        # J22 = my_asm_jac["group2.comp2.x2", "group2.comp2.x2"][0]

        # extract the first and second subsystems
        subsys1, _ = system._subsystems_allprocs["group1"]
        subsys2, _ = system._subsystems_allprocs["group2"]

        # mode = self._orig_mode
        if system.comm.rank == 0:
            print("States:")
            print(subsys2._outputs.asarray())

            print("\nResiduals:")
            print(subsys2._residuals.asarray(), flush=True)

        # ideally, we are solving for all outputs in subsys2
        # so this will be our jacobian size
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

        if self._modeNS == "fwd":
            print("FWD")
            # backup the vectors we are working with
            rvec = system._vectors["residual"]["linear"]
            ovec = system._vectors["output"]["linear"]
            ivec = system._vectors["input"]["linear"]

            r_data = rvec.asarray(copy=True)
            o_data = ovec.asarray(copy=True)
            # i_data = ivec.asarray(copy=True)
            print(rvec, ovec, ivec)
            # set the ovec to zeros
            ovec.set_val(np.zeros(len(ovec)))

            for ii, var in enumerate(vars_to_solve):
                # set the linear seed of the variable we want to solve for in subsys 2
                ovec[f"{subsys2.name}.{var}"] = 1.0

                # transfer this seed to the first subsystem
                system._transfer("linear", "fwd", subsys1.name)

                # run the jac-vec computation in the first subsystem
                scope_out, scope_in = system._get_matvec_scope(subsys1)

                subsys1._apply_linear(None, ["linear"], "fwd", scope_out, scope_in)
                if system.comm.rank == 0:
                    print(f"\nComputing Jacobian columns for {var}")
                    print(f"B[:,{ii}]                    =", subsys1._vectors["output"]["linear"].asarray())
                    print(
                        f"|B[:,{ii}]|                  =",
                        np.linalg.norm(subsys1._vectors["output"]["linear"].asarray()),
                        flush=True,
                    )

                # using the result from this jac-vec product, solve the RHS for this subsystem
                subsys1._solve_linear("fwd", ["linear"], ContainsAll())
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
                system._transfer("linear", "fwd")

                # run the apply linear. we do it on the complete system here
                scope_out, scope_in = system._get_matvec_scope()
                system._apply_linear(None, ["linear"], "fwd", scope_out, scope_in)

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

        elif self._modeNS == "rev":  # rev mode

            # backup the vectors we are working with
            rvec = system._vectors["residual"]["linear"]
            ovec = system._vectors["output"]["linear"]
            ivec = system._vectors["input"]["linear"]

            r_data = rvec.asarray(copy=True)
            o_data = ovec.asarray(copy=True)
            # i_data = ivec.asarray(copy=True)
            # set the rvec to zeros
            # print(rvec, subsys2._outputs, ivec)
            rvec.set_val(np.zeros(len(rvec)))
            # ovec.set_val(np.zeros(len(rvec)))
            # ovec.set_val(np.ones(len(rvec)))
            # ovec.set_vec(system._vectors["output"]["nonlinear"])
            print(vars_to_solve)
            for ii, var in enumerate(resd_to_solve):
                # set the linear seed of the variable we want to solve for in subsys 2

                rvec[f"{subsys2.name}.{var}"] = 1.0
                print(
                    "subsys2",
                    subsys2._vectors["residual"]["linear"],
                    subsys2._vectors["output"]["linear"],
                    subsys2._vectors["input"]["linear"],
                )
                # system._transfer("linear", "rev", subsys1.name)
                # run the jac-vec computation in the first subsystem
                # scope_out, scope_in = system._get_matvec_scope()
                # system._apply_linear(None, ["linear"], "rev", scope_out, scope_in)
                # print(
                #     "subsys2",
                #     subsys2._vectors["residual"]["linear"],
                #     subsys2._vectors["output"]["linear"],
                #     subsys2._vectors["input"]["linear"],
                # )
                # print(
                #     "A subsys1",
                #     subsys1._vectors["residual"]["linear"],
                #     subsys1._vectors["output"]["linear"],
                #     subsys1._vectors["input"]["linear"],
                # )
                # system._transfer("linear", "rev")
                scope_out, scope_in = system._get_matvec_scope()
                # print(scope_out, scope_in)
                system._apply_linear(None, ["linear"], "rev", scope_out, scope_in)

                # print("jac:", subsys2._jacobian["jac"])
                if system.comm.rank == 0:
                    print(f"\nComputing Jacobian columns for {var}")
                    print(f"C[{ii},:]                    =", subsys2._vectors["input"]["linear"].asarray())
                    print(
                        f"|C[{ii},:]|                  =",
                        np.linalg.norm(subsys2._vectors["input"]["linear"].asarray()),
                        flush=True,
                    )
                print(
                    "subsys2",
                    subsys2._vectors["residual"]["linear"],
                    subsys2._vectors["output"]["linear"],
                    subsys2._vectors["input"]["linear"],
                )
                print(
                    "B subsys1",
                    subsys1._vectors["residual"]["linear"],
                    subsys1._vectors["output"]["linear"],
                    subsys1._vectors["input"]["linear"],
                )

                system._transfer("linear", "rev", subsys1.name)
                # # run the jac-vec computation in the first subsystem
                print(
                    "A subsys1",
                    subsys1._vectors["residual"]["linear"],
                    subsys1._vectors["output"]["linear"],
                    subsys1._vectors["input"]["linear"],
                )
                subsys1._solve_linear("rev", ["linear"], ContainsAll())
                print(
                    "subsys1",
                    subsys1._vectors["residual"]["linear"],
                    subsys1._vectors["output"]["linear"],
                    subsys1._vectors["input"]["linear"],
                )
                # print(subsys2._vectors["residual"]["linear"])
                if system.comm.rank == 0:
                    # print(f"\nComputing Jacobian columns for {var}")
                    print(f"C[{ii},:] A^-1                    =", subsys1._vectors["residual"]["linear"].asarray())
                    print(
                        f"|C[{ii},:] A^-1|                  =",
                        np.linalg.norm(subsys1._vectors["residual"]["linear"].asarray()),
                        flush=True,
                    )
                # print(subsys1._vectors["residual"]["linear"])
                subsys1._vectors["residual"]["linear"] *= -1.0

                # print(subsys1._vectors["residual"]["linear"])
                scope_out, scope_in = system._get_matvec_scope(subsys1)
                # print(subsys1._vectors["input"]["linear"])
                subsys1._apply_linear(None, ["linear"], "rev", scope_out, scope_in)
                print(
                    "subsys1",
                    subsys1._vectors["residual"]["linear"],
                    subsys1._vectors["output"]["linear"],
                    subsys1._vectors["input"]["linear"],
                )
                # print(subsys1._vectors["input"]["linear"])
                if system.comm.rank == 0:
                    print(f"C[{ii},:] A^-1 B               =", subsys1._vectors["input"]["linear"].asarray())
                    print(
                        f"|C[{ii},:] A^-1 B|             =",
                        np.linalg.norm(subsys1._vectors["input"]["linear"].asarray()),
                        flush=True,
                    )
                subsys1._solve_linear("rev", ["linear"], ContainsAll())
                # do another mat-mult with the solution of this linear system, we want to get the final
                # jacobian using the schur method here, so we will need to do a bit more math
                # transfer the outputs to inputs
                system._transfer("linear", "rev")
                # # subsys2._solve_linear("rev", ["linear"], ContainsAll())
                # scope_out, scope_in = system._get_matvec_scope()
                # print(
                #     "subsys2",
                #     subsys2._vectors["residual"]["linear"],
                #     subsys2._vectors["output"]["linear"],
                #     subsys2._vectors["input"]["linear"],
                # )
                # print(
                #     "subsys1",
                #     subsys1._vectors["residual"]["linear"],
                #     subsys1._vectors["output"]["linear"],
                #     subsys1._vectors["input"]["linear"],
                # )
                # # print(scope_out, scope_in)
                # system._apply_linear(None, ["linear"], "rev", scope_out, scope_in)
                if system.comm.rank == 0:
                    print("seed for C | D            =", system._vectors["residual"]["linear"].asarray(), flush=True)

                # transfer the outputs to inputs
                # system._transfer("linear", "fwd")

                # run the apply linear. we do it on the complete system here
                # scope_out, scope_in = system._get_matvec_scope()
                # system._apply_linear(None, ["linear"], "rev", scope_out, scope_in)
                # subsys2._solve_linear("rev", ["linear"], ContainsAll())
                # scope_out, scope_in = system._get_matvec_scope()
                # system._apply_linear(None, ["linear"], "rev", scope_out, scope_in)
                # the result is the final jacobian for this using the schur complement method
                if system.comm.rank == 0:
                    print(
                        f"D[{ii},:] - C A^-1 B[{ii},:]    =",
                        subsys2._vectors["output"]["linear"].asarray(),
                        flush=True,
                    )
                print(
                    "subsys2",
                    subsys2._vectors["residual"]["linear"],
                    subsys2._vectors["output"]["linear"],
                    subsys2._vectors["input"]["linear"],
                )
                print(
                    "subsys1",
                    subsys1._vectors["residual"]["linear"],
                    subsys1._vectors["output"]["linear"],
                    subsys1._vectors["input"]["linear"],
                )
                # quit()

                # put this value into the jacobian.
                schur_jac[ii, :] = subsys2._vectors["output"]["linear"].asarray()

                # set back the seed to zero for the next vector
                rvec[f"{subsys2.name}.{var}"] = 0.0
        # print(subsys1._residuals, ovec)

        # put back the vectors
        rvec.set_val(r_data)
        ovec.set_val(o_data)
        print(
            "subsys1",
            subsys1._vectors["residual"]["linear"],
            subsys1._vectors["output"]["linear"],
            subsys1._vectors["input"]["linear"],
        )
        print(
            "subsys2",
            subsys2._vectors["residual"]["linear"],
            subsys2._vectors["output"]["linear"],
            subsys2._vectors["input"]["linear"],
        )
        # quit()
        # ivec.set_val(i_data)

        # we now have the schur complement of the jacobian for the second component.
        # do a newton update with it!
        subsys2._vectors["residual"]["linear"].set_vec(subsys2._residuals)
        subsys2._vectors["output"]["linear"].set_vec(subsys2._outputs)
        subsys2._vectors["residual"]["linear"] *= -1.0
        # print(subsys2._residuals)
        # quit()
        # custom_jac = np.zeros((3,3))
        # custom_jac[0, :] = np.array([
        #     -0.00940017,
        #     -0.00261924,
        #     0.1737319
        # ])
        # custom_jac[1, :] = np.array([
        #     -0.00152532,
        #     -0.00149961,
        #     7.83679413e-05
        # ])
        # custom_jac[2, :] = np.array([
        #     -2.43617362,
        #     -0.5398966,
        #     -45.166525
        # ])

        if system.comm.rank == 0:
            print("Schur Jacobian:\n", schur_jac, flush=True)
            # print("My    Jacobian:\n", custom_jac, flush=True)
        # system.comm.barrier()

        d_subsys2 = scipy.linalg.solve(schur_jac, subsys2._vectors["residual"]["linear"].asarray())

        if system.comm.rank == 0:
            print("update vector:", d_subsys2, flush=True)

        # # the R1 should already be zero from the solve subsystems call
        # # R1 = system._residuals['group1.x1'][0]
        # R2 = system._residuals['group2.x2'][0]

        # # compute the RHS for x2
        # # because R1 is zero, we dont need to add its contribution here!
        # rhs2 = -R2  # + J21 * (1. / J11) * R1

        # # compute the LHS for x2
        # # this is the tricky bit, we need to modify the jacobian of the system we are solving for,
        # # using the information from the other subsystem (J11) and their coupling (J12 and J21)
        # lhs2 = J22 - J21 * (1. / J11) * J12

        # # print(lhs2, schur_jac)

        # # update for x2
        # # this will be replaced by a solve linear call
        # dx2 = rhs2 / lhs2

        # quit()

        # RHS for x1
        # again, no need to solve for x1, we want to purely rely on the subsystem solve  for that
        # rhs1 = -R1 - J12 * dx2
        # lhs1 = J11
        # dx1 = rhs1 / lhs1
        # system._outputs["group1.x1"] += dx1

        # take the update for x2. this will include a line search as well!
        # system._outputs["group2.x2"] += dx2

        # take the update
        # if self.linesearch:
        #     self.linesearch._do_subsolve = do_subsolve
        #     self.linesearch.solve()
        # else:
        #     system._outputs += system._vectors['output']['linear']

        # loop over the variables just to be safe with the ordering
        for ii, var in enumerate(vars_to_solve):
            system._outputs[f"{subsys2.name}.{var}"] += d_subsys2[ii]

        self._solver_info.pop()

        # Hybrid newton support.
        if do_subsolve:
            with Recording("Newton_subsolve", 0, self):
                self._solver_info.append_solver()
                self._gs_iter()
                self._solver_info.pop()

        # Enable local fd
        system._owns_approx_jac = approx_status

    def _set_complex_step_mode(self, active):
        """
        Turn on or off complex stepping mode.

        Recurses to turn on or off complex stepping mode in all subsystems and their vectors.

        Parameters
        ----------
        active : bool
            Complex mode flag; set to True prior to commencing complex step.
        """
        if self.linear_solver is not None:
            self.linear_solver._set_complex_step_mode(active)
            if self.linear_solver._assembled_jac is not None:
                self.linear_solver._assembled_jac.set_complex_step_mode(active)

    def cleanup(self):
        """
        Clean up resources prior to exit.
        """
        super().cleanup()

        if self.linear_solver:
            self.linear_solver.cleanup()
        if self.linesearch:
            self.linesearch.cleanup()
