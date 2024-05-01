"""LinearSolver that uses PetSC KSP to solve for a system's derivatives."""

import numpy as np
import os
import sys

from openmdao.solvers.solver import LinearSolver
from openmdao.solvers.linear.linear_rhs_checker import LinearRHSChecker
from openmdao.utils.mpi import check_mpi_env

use_mpi = check_mpi_env()
if use_mpi is not False:
    try:
        import petsc4py
        from petsc4py import PETSc
    except ImportError:
        PETSc = None
        if use_mpi is True:
            raise ImportError("Importing petsc4py failed and OPENMDAO_USE_MPI is true.")
else:
    PETSc = None

KSP_TYPES = [
    "richardson",
    "chebyshev",
    "cg",
    "groppcg",
    "pipecg",
    "pipecgrr",
    "cgne",
    "nash",
    "stcg",
    "gltr",
    "fcg",
    "pipefcg",
    "gmres",
    "pipefgmres",
    "fgmres",
    "lgmres",
    "dgmres",
    "pgmres",
    "tcqmr",
    "bcgs",
    "ibcgs",
    "fbcgs",
    "fbcgsr",
    "bcgsl",
    "cgs",
    "tfqmr",
    "cr",
    "pipecr",
    "lsqr",
    "preonly",
    "qcg",
    "bicg",
    "minres",
    "symmlq",
    "lcd",
    "python",
    "gcr",
    "pipegcr",
    "tsirm",
    "cgls"
]


def _get_petsc_vec_array_new(vec):
    """
    Get the array of values for the given PETSc vector.

    Helper function to handle a petsc backwards incompatibility.

    Parameters
    ----------
    vec : petsc vector
        Vector whose data is being requested.

    Returns
    -------
    ndarray
        A readonly copy of the array of values from vec.
    """
    return vec.getArray(readonly=True)


def _get_petsc_vec_array_old(vec):
    """
    Get the array of values for the given PETSc vector.

    Helper function to handle a petsc backwards incompatibility.

    Parameters
    ----------
    vec : petsc vector
        Vector whose data is being requested.

    Returns
    -------
    ndarray
        An array of values from vec.
    """
    return vec.getArray()


if PETSc:
    try:
        petsc_version = petsc4py.__version__
    except AttributeError:  # hack to fix doc-tests
        petsc_version = "3.5"


if PETSc and int((petsc_version).split('.')[1]) >= 6:
    _get_petsc_vec_array = _get_petsc_vec_array_new
else:
    _get_petsc_vec_array = _get_petsc_vec_array_old


class Monitor(object):
    """
    Prints output from PETSc's KSP solvers.

    Callable object given to KSP as a callback for printing the residual.

    Parameters
    ----------
    solver : object
        The openmdao solver.

    Attributes
    ----------
    _solver : _solver
        The openmdao solver.
    _norm : float
        The current norm.
    _norm0 : float
        The norm for the first iteration.
    """

    def __init__(self, solver):
        """
        Store pointer to the openmdao solver and initialize norms.
        """
        self._solver = solver
        self._norm = 1.0
        self._norm0 = 1.0

    def __call__(self, ksp, counter, norm):
        """
        Store norm if first iteration, and print norm.

        Parameters
        ----------
        ksp : object
            the KSP solver.
        counter : int
            the counter.
        norm : float
            the norm.
        """
        if counter == 0 and norm != 0.0:
            self._norm0 = norm
        self._norm = norm

        self._solver._mpi_print(counter, norm, norm / self._norm0)
        self._solver._iter_count += 1


class PETScKrylov(LinearSolver):
    """
    LinearSolver that uses PetSC KSP to solve for a system's derivatives.

    Parameters
    ----------
    **kwargs : dict
        Dictionary of options set by the instantiating class/script.

    Attributes
    ----------
    precon : Solver
        Preconditioner for linear solve. Default is None for no preconditioner.
    _ksp : dist
        Dictionary of KSP instances (keyed on vector name).
    _lin_rhs_checker : LinearRHSChecker or None
        Object for checking the right-hand side of the linear solve.
    """

    SOLVER = 'LN: PETScKrylov'

    def __init__(self, **kwargs):
        """
        Declare the solver options.
        """
        super().__init__(**kwargs)

        if PETSc is None:
            raise RuntimeError(f"{self.msginfo}: PETSc is not available. "
                               "Set shell variable OPENMDAO_USE_MPI=1 to detect earlier.")

        self._ksp = None
        self.precon = None
        self._lin_rhs_checker = None

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        super()._declare_options()

        self.options.declare('ksp_type', default='fgmres', values=KSP_TYPES,
                             desc="KSP algorithm to use. Default is 'fgmres'.")

        self.options.declare('restart', default=1000, types=int,
                             desc='Number of iterations between restarts. Larger values increase '
                             'iteration cost, but may be necessary for convergence')

        self.options.declare('precon_side', default='right', values=['left', 'right'],
                             desc='Preconditioner side, default is right.')

        self.options.declare('rhs_checking', types=(bool, dict),
                             default=False,
                             desc="If True, check RHS vs. cache and/or zero to avoid some solves."
                             "Can also be set to a dict of options for the LinearRHSChecker to "
                             "allow finer control over it. Allowed options are: "
                             f"{LinearRHSChecker.options}")

        # changing the default maxiter from the base class
        self.options['maxiter'] = 100

    def _assembled_jac_solver_iter(self):
        """
        Return a generator of linear solvers using assembled jacs.
        """
        if self.options['assemble_jac']:
            yield self
        if self.precon is not None:
            for s in self.precon._assembled_jac_solver_iter():
                yield s

    def use_relevance(self):
        """
        Return True if relevance should be active.

        Returns
        -------
        bool
            True if relevance should be active.
        """
        return False

    def _setup_solvers(self, system, depth):
        """
        Assign system instance, set depth, and optionally perform setup.

        Parameters
        ----------
        system : <System>
            pointer to the owning system.
        depth : int
            depth of the current system (already incremented).
        """
        super()._setup_solvers(system, depth)

        if self.precon is not None:
            self.precon._setup_solvers(self._system(), self._depth + 1)

        self._lin_rhs_checker = LinearRHSChecker.create(self._system(),
                                                        self.options['rhs_checking'])

    def _set_solver_print(self, level=2, type_='all'):
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

        if self.precon is not None and type_ != 'NL':
            self.precon._set_solver_print(level=level, type_=type_)

    def mult(self, mat, in_vec, result):
        """
        Apply Jacobian matrix (KSP Callback).

        The following attributes must be defined when solve is called to
        provide information used in this callback:

        _system : System
            pointer to the owning system.
        _vec_name : str
            the right-hand-side (RHS) vector name.
        _mode : str
            'fwd' or 'rev'.

        Parameters
        ----------
        mat : PETSc.Mat
            PETSc matrix object.
        in_vec : PetSC Vector
            Incoming vector.
        result : PetSC Vector
            Empty array into which we place the matrix-vector product.
        """
        # assign x and b vectors based on mode
        system = self._system()

        if self._mode == 'fwd':
            x_vec = system._doutputs
            b_vec = system._dresiduals
        else:  # rev
            x_vec = system._dresiduals
            b_vec = system._doutputs

        # set value of x vector to KSP provided value
        x_vec.set_val(_get_petsc_vec_array(in_vec))

        # apply linear
        scope_out, scope_in = system._get_matvec_scope()
        system._apply_linear(self._assembled_jac, self._mode, scope_out, scope_in)

        # stuff resulting value of b vector into result for KSP
        result.array[:] = b_vec.asarray()

    def _linearize_children(self):
        """
        Return a flag that is True when we need to call linearize on our subsystems' solvers.

        Returns
        -------
        bool
            Flag for indicating child linerization
        """
        precon = self.precon
        return (precon is not None) and (precon._linearize_children())

    def _linearize(self):
        """
        Perform any required linearization operations such as matrix factorization.
        """
        if self.precon is not None:
            self.precon._linearize()

        if self._lin_rhs_checker is not None:
            self._lin_rhs_checker.clear()

    def solve(self, mode, rel_systems=None):
        """
        Solve the linear system for the problem in self._system().

        The full solution vector is returned.

        Parameters
        ----------
        mode : str
            Derivative mode, can be 'fwd' or 'rev'.
        rel_systems : set of str
            Names of systems relevant to the current solve.  Deprecated.
        """
        self._mode = mode

        system = self._system()
        options = self.options

        if system.under_complex_step:
            raise RuntimeError('{}: PETScKrylov solver is not supported under '
                               'complex step.'.format(self.msginfo))

        maxiter = options['maxiter']
        atol = options['atol']
        rtol = options['rtol']

        # assign x and b vectors based on mode
        if self._mode == 'fwd':
            x_vec = system._doutputs
            b_vec = system._dresiduals
        else:  # rev
            x_vec = system._dresiduals
            b_vec = system._doutputs

            if self._lin_rhs_checker is not None:
                sol_array, is_zero = self._lin_rhs_checker.get_solution(b_vec.asarray(), system)
                if is_zero:
                    x_vec.set_val(0.0)
                    return
                if sol_array is not None:
                    x_vec.set_val(sol_array)
                    return

        rhs_array = b_vec.asarray(copy=True)
        sol_array = x_vec.asarray(copy=True)

        # create PETSc vectors from numpy arrays
        sol_petsc_vec = PETSc.Vec().createWithArray(sol_array, comm=system.comm)
        rhs_petsc_vec = PETSc.Vec().createWithArray(rhs_array, comm=system.comm)

        # run PETSc solver
        self._iter_count = 0
        ksp = self._get_ksp_solver(system)
        ksp.setTolerances(max_it=maxiter, atol=atol, rtol=rtol)
        ksp.solve(rhs_petsc_vec, sol_petsc_vec)

        # stuff the result into the x vector
        x_vec.set_val(sol_array)

        # as of petsc4py v3.20, the 'converged' attribute has been renamed to 'is_converged'
        if hasattr(self._ksp, 'is_converged'):
            if not self._ksp.is_converged:
                self._convergence_failure()
        elif not self._ksp.converged:
            self._convergence_failure()

        sol_petsc_vec = rhs_petsc_vec = None

        if not system.under_complex_step and self._lin_rhs_checker is not None and mode == 'rev':
            self._lin_rhs_checker.add_solution(rhs_array, sol_array, copy=False)

    def apply(self, mat, in_vec, result):
        """
        Apply preconditioner.

        Parameters
        ----------
        mat : PETSc.Mat
            PETSc matrix object.
        in_vec : PETSc.Vector
            Incoming vector.
        result : PETSc.Vector
            Empty vector in which the preconditioned in_vec is stored.
        """
        if self.precon:
            system = self._system()
            mode = self._mode

            # Need to clear out any junk from the inputs.
            system._dinputs.set_val(0.0)

            # assign x and b vectors based on mode
            if mode == 'fwd':
                x_vec = system._doutputs
                b_vec = system._dresiduals
            else:  # rev
                x_vec = system._dresiduals
                b_vec = system._doutputs

            # set value of b vector to KSP provided value
            b_vec.set_val(_get_petsc_vec_array(in_vec))

            # call the preconditioner
            self._solver_info.append_precon()
            self.precon.solve(mode)
            self._solver_info.pop()

            # stuff resulting value of x vector into result for KSP
            result.array[:] = x_vec.asarray()
        else:
            # no preconditioner, just pass back the incoming vector
            result.array[:] = _get_petsc_vec_array(in_vec)

    def _get_ksp_solver(self, system):
        """
        Get an instance of the KSP solver in `system`.

        Instances will be created on first request and cached for future use.

        Parameters
        ----------
        system : `System`
            Parent `System` object.

        Returns
        -------
        KSP
            the KSP solver instance.
        """
        # use cached instance if available
        if self._ksp:
            return self._ksp

        iproc = system.comm.rank
        lsize = np.sum(system._var_sizes['output'][iproc, :])
        size = np.sum(system._var_sizes['output'])

        jac_mat = PETSc.Mat().createPython([(lsize, size), (lsize, size)],
                                           comm=system.comm)
        jac_mat.setPythonContext(self)
        jac_mat.setUp()

        ksp = self._ksp = PETSc.KSP().create(comm=system.comm)

        ksp.setOperators(jac_mat)
        ksp.setType(self.options['ksp_type'])
        ksp.setGMRESRestart(self.options['restart'])
        if self.options['precon_side'] == 'left':
            ksp.setPCSide(PETSc.PC.Side.LEFT)
        else:
            ksp.setPCSide(PETSc.PC.Side.RIGHT)
        ksp.setMonitor(Monitor(self))
        ksp.setInitialGuessNonzero(True)

        pc_mat = ksp.getPC()
        pc_mat.setType('python')
        pc_mat.setPythonContext(self)

        return ksp
