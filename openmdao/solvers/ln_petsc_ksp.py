"""LinearSolver that uses PetSC KSP to solve for a system's derivatives.

This solver can be used under MPI.
"""
from __future__ import division, print_function

import petsc4py
from petsc4py import PETSc

from openmdao.solvers.solver import LinearSolver

from pprint import pprint


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
    """Helper function to handle a petsc backwards incompatibility."""
    return vec.getArray(readonly=True)


def _get_petsc_vec_array_old(vec):
    """Helper function to handle a petsc backwards incompatibility."""
    return vec.getArray()


try:
    petsc_version = petsc4py.__version__
except AttributeError:  # hack to fix doc-tests
    petsc_version = "3.5"


if int((petsc_version).split('.')[1]) >= 6:
    _get_petsc_vec_array = _get_petsc_vec_array_new
else:
    _get_petsc_vec_array = _get_petsc_vec_array_old


class Monitor(object):
    """Prints output from PETSc's KSP solvers.

    Callable object given to KSP as a callback object for printing the residual.
    """

    def __init__(self, ksp):
        """Store pointer to the ksp solver.

        Args
        ----
        ksp : object
            the KSP solver
        """
        self._ksp = ksp
        self._norm = 1.0
        self._norm0 = 1.0

    def __call__(self, ksp, counter, norm):
        """Store norm if first iteration, and print norm.

        Args
        ----
        ksp : object
            the KSP solver

        counter : int
            the counter

        norm : float
            the norm
        """
        print('---------------------------')
        print('ln_petsc_ksp Monitor, counter =', counter, 'norm:', norm)
        if counter == 0 and norm != 0.0:
            self._norm0 = norm
        self._norm = norm

        self._ksp._mpi_print(counter, norm / self._norm0, norm)
        self._ksp._iter_count += 1
        print('---------------------------')


class PetscKSP(LinearSolver):
    """LinearSolver that uses PetSC KSP to solve for a system's derivatives.

    Options
    -------
    options['ksp_type'] :  str('fgmres')
        KSP algorithm to use. Default is 'fgmres'.
    """

    SOLVER = 'LN: PetscKSP'

    def __init__(self, **kwargs):
        """Declare the solver options.

        Args
        ----
        kwargs : {}
            dictionary of options set by the instantiating class/script.
        """
        super(PetscKSP, self).__init__(**kwargs)

        self._print_name = 'KSP'

        # additional option to specify KSP algorithm to use
        opt = self.options
        opt.declare('ksp_type', value='fgmres', values=KSP_TYPES,
                    desc="KSP algorithm to use. Default is 'fgmres'.")

        # initialize dictionary of KSP instances (keyed on vector name)
        self.ksp = {}

        # TODO: User can specify another linear solver to use as a preconditioner
        self.preconditioner = None

    def mult(self, mat, in_vec, result):
        """Apply Jacobian matrix (KSP Callback).

        The following attributes must be defined when solve is called to provide
        info used in this callback:

        _system : System
            pointer to the owning system.
        _vec_name : str
            the right-hand-side (RHS) vector name.
        _mode : str
            'fwd' or 'rev'.

        Args
        ----
        in_vec : PetSC Vector
            Incoming vector

        result : PetSC Vector
            Empty array into which we place the matrix-vector product.
        """
        print('---------------------------')
        print('ln_petsc_ksp mult')

        # assign x and b vectors based on mode
        system = self._system
        vec_name = self._vec_name

        if self._mode == 'fwd':
            x_vec = system._vectors['output'][vec_name]
            b_vec = system._vectors['residual'][vec_name]
        elif self._mode == 'rev':
            x_vec = system._vectors['residual'][vec_name]
            b_vec = system._vectors['output'][vec_name]

        # set value of x vector to KSP provided value
        x_vec.set_data(_get_petsc_vec_array(in_vec))

        # apply linear
        ind1, ind2 = system._variable_allprocs_range['output']
        var_inds = [ind1, ind2, ind1, ind2]
        system._apply_linear([vec_name], self._mode, var_inds)

        # return resulting value of b vector to KSP
        b_vec.get_data(result.array)
        print('ln_petsc_ksp mult result:\n', result.array)
        print('---------------------------')

    def __call__(self, vec_names, mode):
        """Solve the linear system for the problem in self._system.

        The full solution vector is returned.

        Args
        ----
        vec_names : list of vector names

        mode : string
            Derivative mode, can be 'fwd' or 'rev'.
        """
        print('ln_ksp __call__', vec_names, mode)
        self._vec_names = vec_names
        self._mode = mode

        system = self._system
        options = self.options

        maxiter = options['maxiter']
        atol = options['atol']
        rtol = options['rtol']

        print('ln_ksp __call__ vec_names=', self._vec_names)

        # for vec_name, rhs in iteritems(rhs_mat):
        for vec_name in self._vec_names:
            self._vec_name = vec_name

            print('ln_ksp __call__ path_name=', system.path_name, "vec_name =", vec_name)

            # assign x and b vectors based on mode
            if self._mode == 'fwd':
                x_vec = system._vectors['output'][vec_name]
                b_vec = system._vectors['residual'][vec_name]
            elif self._mode == 'rev':
                x_vec = system._vectors['residual'][vec_name]
                b_vec = system._vectors['output'][vec_name]

            # create numpy arrays to interface with Petsc
            sol_array = x_vec.get_data()
            rhs_array = b_vec.get_data()

            # create Petsc vectors on top of numpy arrays
            self.sol_petsc_vec = PETSc.Vec().createWithArray(sol_array, comm=system.comm)
            self.rhs_petsc_vec = PETSc.Vec().createWithArray(rhs_array, comm=system.comm)

            # run Petsc solver
            self._iter_count = 0
            ksp = self._get_ksp_solver(system, vec_name)
            ksp.setTolerances(max_it=maxiter, atol=atol, rtol=rtol)
            ksp.solve(self.rhs_petsc_vec, self.sol_petsc_vec)

            # stuff the result into the x vector
            print('result:')
            pprint(sol_array)
            x_vec.set_data(sol_array)

    def apply(self, mat, in_vec, result):
        """Apply preconditioner.

        Args
        ----
        in_vec : PetSC Vector
            Incoming vector

        result : PetSC Vector
            Empty vector into which we return the preconditioned in_vec
        """
        if self.preconditioner is None:
            result.array[:] = _get_petsc_vec_array(in_vec)
            return
        else:
            # TODO
            pass

    def _get_ksp_solver(self, system, vec_name):
        """Get an instance of the KSP solver for `vec_name` in `system`.

        Instances will be created on first request and cached for future use.

        Args
        ----
        system : `System`
            Parent `System` object.

        vec_name : string
            name of vector
        """
        # use cached instance if available
        if vec_name in self.ksp:
            return self.ksp[vec_name]

        rank = system.comm.rank + system._mpi_proc_range[0]

        lsizes = system._sys_assembler._variable_sizes_all['output'][rank]
        sizes = sum(system._sys_assembler._variable_sizes_all['output'])

        lsize = sum(lsizes)
        size = sum(sizes)

        jac_mat = PETSc.Mat().createPython([(lsize, size), (lsize, size)],
                                           comm=system.comm)
        jac_mat.setPythonContext(self)
        jac_mat.setUp()

        ksp = self.ksp[vec_name] = PETSc.KSP().create(comm=system.comm)

        ksp.setOperators(jac_mat)
        ksp.setType(self.options['ksp_type'])
        ksp.setGMRESRestart(1000)
        ksp.setPCSide(PETSc.PC.Side.RIGHT)
        ksp.setMonitor(Monitor(self))

        pc_mat = ksp.getPC()
        pc_mat.setType('python')
        pc_mat.setPythonContext(self)

        return ksp
