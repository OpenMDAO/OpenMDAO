"""LinearSolver that uses PetSC KSP to solve for a system's derivatives.

This solver can be used under MPI.
"""
from __future__ import division, print_function

import os

# TODO: Do we have to make this solver with a factory?
import petsc4py
from petsc4py import PETSc
import numpy as np
from collections import OrderedDict

from openmdao.solvers.solver import LinearSolver

trace = os.environ.get("OPENMDAO_TRACE")

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


#
# Class object is given to KSP as a callback object for printing the residual.
#
class Monitor(object):
    """Prints output from PETSc's KSP solvers."""

    def __init__(self, ksp):
        """Store pointer to the ksp solver.

        Args
        ----
        ksp : object
            the KSP solver
        """
        self._ksp = ksp
        self._norm0 = 1.0
        self._norm = 1.0

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
        if counter == 0 and norm != 0.0:
            self._norm0 = norm
        self._norm = norm

        ksp = self._ksp
        ksp.iter_count += 1

        if ksp.options['iprint'] == 2:
            ksp.print_norm(ksp.print_name, ksp.system, ksp.iter_count,
                           norm, self._norm0, indent=1, solver='LN')


class PetscKSP(LinearSolver):
    """LinearSolver that uses PetSC KSP to solve for a system's derivatives.

    This solver can be used under MPI.

    Options
    -------
    options['atol'] :  float(1e-12)
        Absolute convergence tolerance.
    options['err_on_maxiter'] : bool(False)
        If True, raise an Error if not converged at maxiter.
    options['iprint'] :  int(0)
        Set to 0 to print only failures, set to 1 to print iteration totals to
        stdout, set to 2 to print the residual each iteration to stdout,
        or -1 to suppress all printing.
    options['ksp_type'] :  str('fgmres')
        KSP algorithm to use. Default is 'fgmres'.
    options['maxiter'] :  int(100)
        Maximum number of iterations.
    options['mode'] :  str('auto')
        Derivative calculation mode 'fwd' (forward), 'rev' (reverse), 'auto'.
    options['rtol'] :  float(1e-12)
        Relative convergence tolerance.
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

        opt = self.options
        opt.declare('ksp_type', value='fgmres', values=KSP_TYPES,
                    desc="KSP algorithm to use. Default is 'fgmres'.")

        # These are defined whenever we call solve to provide info we need in
        # the callback.
        # self.system = None
        # self.voi = None
        # self.mode = None

        self.ksp = {}
        self.print_name = 'KSP'

        # User can specify another linear solver to use as a preconditioner
        self.preconditioner = None

    def setup(self, system):
        """Setup petsc problem just once.

        Args
        ----
        system : `System`
            Parent `System` object.
        """
        if not system.is_active():
            return

        # allocate and cache the ksp problem for each voi
        for voi in system.dumat:
            sizes = system._local_unknown_sizes[voi]
            lsize = np.sum(sizes[system.comm.rank, :])
            size = np.sum(sizes)

            if trace:
                print("creating petsc matrix of size (%d,%d)" % (lsize, size))
            jac_mat = PETSc.Mat().createPython([(lsize, size), (lsize, size)],
                                               comm=system.comm)
            if trace:
                print("petsc matrix creation DONE for %s" % voi)
            jac_mat.setPythonContext(self)
            jac_mat.setUp()

            if trace:  # pragma: no cover
                print("creating KSP object for system", system.path_name)

            ksp = self.ksp[voi] = PETSc.KSP().create(comm=system.comm)
            if trace:
                print("KSP creation DONE")

            ksp.setOperators(jac_mat)
            ksp.setType(self.options['ksp_type'])
            ksp.setGMRESRestart(1000)
            ksp.setPCSide(PETSc.PC.Side.RIGHT)
            ksp.setMonitor(Monitor(self))

            if trace:  # pragma: no cover
                print("ksp.getPC()")
                print("b_vec, x_vec size: %d" % lsize)
            pc_mat = ksp.getPC()
            pc_mat.setType('python')
            pc_mat.setPythonContext(self)

        if trace:  # pragma: no cover
            print("ksp setup done")

        if self.preconditioner:
            self.preconditioner.setup(system)

    def print_all_convergence(self, level=2):
        """Turn on iprint for this solver and all subsolvers.

        Override if your solver has subsolvers.

        Args
        ----
        level : int(2)
            iprint level. Set to 2 to print residuals each iteration; set to 1
            to print just the iteration totals.
        """
        self.options['iprint'] = level
        if self.preconditioner:
            self.preconditioner.print_all_convergence(level)

    def mult(self, mat, in_vec, result):
        """KSP Callback: applies Jacobian matrix. Mode is determined by the system.

        Args
        ----
        in_vec : PetSC Vector
            Incoming vector

        result : PetSC Vector
            Empty array into which we place the matrix-vector product.
        """
        print('---------------------------')
        print('ln_petsc_ksp mult')
        vec_name = self._vec_name
        system = self._system
        ind1, ind2 = system._variable_allprocs_range['output']

        if self._mode == 'fwd':
            x_vec = system._vectors['output'][vec_name]
            b_vec = system._vectors['residual'][vec_name]
        elif self._mode == 'rev':
            x_vec = system._vectors['residual'][vec_name]
            b_vec = system._vectors['output'][vec_name]

        print('x_vec:', x_vec)
        print('b_vec:', b_vec)

        # Set incoming vector
        in_data = _get_petsc_vec_array(in_vec)
        print('in_data:', in_data)
        x_vec.vec.set_data(in_data)

        print('x_vec:', x_vec)
        print('b_vec:', b_vec)

        var_inds = [
            system._variable_allprocs_range['output'][0],
            system._variable_allprocs_range['output'][1],
            system._variable_allprocs_range['output'][0],
            system._variable_allprocs_range['output'][1],
        ]

        # Start with a clean slate
        b_vec.vec[:] = 0.0
        # system.clear_dparams()

        # system._sys_apply_linear(mode, self._system._do_apply, vois=(voi,),
        #                          rel_inputs=self.rel_inputs)
        system._apply_linear([vec_name], self._mode, var_inds)
        print('ln_petsc_ksp _mat_vec result:\n', b_vec.get_data())
        print('---------------------------')

        return b_vec.get_data()

    def __call__(self, vec_names, mode):
        """Solve the linear system for the problem in self._system.

        The full solution vector is returned.

        Args
        ----
        rhs_mat : dict of ndarray
            Dictionary containing one ndarry per top level quantity of
            interest. Each array contains the right-hand side for the linear
            solve.

        mode : string
            Derivative mode, can be 'fwd' or 'rev'.

        Returns
        -------
        dict of ndarray : Solution vectors
        """
        print('ln_ksp __call__', vec_names, mode)
        self._vec_names = vec_names
        self._mode = mode

        system = self._system
        options = self.options

        # print('ln_ksp __call__ system var names:')
        # pprint(system._variable_allprocs_names)
        # print('ln_ksp __call__ system var names, proc', system.comm.rank)
        # pprint(system._variable_myproc_names)
        # print('ln_ksp __call__ system vectors:')
        # pprint(system._vectors)

        maxiter = options['maxiter']
        iprint = options['iprint']
        atol = options['atol']
        rtol = options['rtol']

        print('ln_ksp __call__ vec_names=', self._vec_names)

        # for vec_name, rhs in iteritems(rhs_mat):
        for vec_name in self._vec_names:
            print('ln_ksp __call__', system.path_name, vec_name)

            self._vec_name = vec_name

            if self._mode == 'fwd':
                x_vec = system._vectors['output'][vec_name].get_data()
                b_vec = system._vectors['residual'][vec_name].get_data()
            elif self._mode == 'rev':
                x_vec = system._vectors['residual'][vec_name].get_data()
                b_vec = system._vectors['output'][vec_name].get_data()

            print('x_vec:', x_vec)
            print('b_vec:', b_vec)

            self.x_vec_petsc = PETSc.Vec().createWithArray(x_vec, comm=system.comm)
            self.b_vec_petsc = PETSc.Vec().createWithArray(b_vec, comm=system.comm)

            # Petsc can only handle one right-hand-side at a time for now
            self.iter_count = 0

            ksp = self._get_ksp_solver(system, vec_name)
            ksp.setTolerances(max_it=maxiter, atol=atol, rtol=rtol)
            ksp.solve(self.b_vec_petsc, self.x_vec_petsc)

            # Final residual print if you only want the last one
            if iprint == 1:
                mon = ksp.getMonitor()[0][0]
                self._mpi_print(self.iter_count, mon._norm, mon._norm0)

            if self.iter_count >= maxiter:
                msg = 'FAILED to converge in %d iterations' % self.iter_count
                fail = True
            else:
                msg = 'Converged in %d iterations' % self.iter_count
                fail = False

            if iprint > 0 or (fail and iprint > -1):
                self._mpi_print(self.iter_count, 0, 0)

            # x_vec.set_data()

            if fail and self.options['err_on_maxiter']:
                raise Exception("Solve in '%s': PetscKSP %s" %
                                (system.path_name, msg))

            # print system.name, 'Linear solution vec', d_unknowns

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

        system = self._system
        mode = self.mode

        voi = self.voi
        if mode == 'fwd':
            x_vec, b_vec = system.dumat[voi], system.drmat[voi]
        else:
            x_vec, b_vec = system.drmat[voi], system.dumat[voi]

        # Set incoming vector
        b_vec.vec[:] = _get_petsc_vec_array(in_vec)

        # Start with a clean slate
        system.clear_dparams()

        dumat = OrderedDict()
        dumat[voi] = system.dumat[voi]
        drmat = OrderedDict()
        drmat[voi] = system.drmat[voi]

        with system._dircontext:
            precon = self.preconditioner
            system._probdata.precon_level += 1
            if precon.options['iprint'] > 0:
                precon.print_norm(precon.print_name, system, precon.iter_count,
                                  0, 0, indent=1, solver='LN',
                                  msg='Start Preconditioner')

            system.solve_linear(dumat, drmat, (voi, ), mode=mode,
                                solver=precon, rel_inputs=self.rel_inputs)

            if precon.options['iprint'] > 0:
                precon.print_norm(precon.print_name, system, precon.iter_count,
                                  0, 0, indent=1, solver='LN',
                                  msg='End Preconditioner')
            system._probdata.precon_level -= 1

        result.array[:] = x_vec.vec

    def _get_ksp_solver(self, system, vec_name):
        """Setup petsc problem.

        Args
        ----
        system : `System`
            Parent `System` object.

        vec_name : string
            name of vector (variable of interest)
        """
        # use cached instance if available
        if vec_name in self.ksp:
            return self.ksp[vec_name]

        print('=====================')
        rank = system.comm.rank + system._mpi_proc_range[0]
        lsizes = system._sys_assembler._variable_sizes_all['output'][rank]
        sizes = sum(system._sys_assembler._variable_sizes_all['output'])
        print('_get_ksp_solver() lsizes =', lsizes, 'sizes =', sizes)

        lsize = sum(lsizes)
        size = sum(sizes)
        print('_get_ksp_solver() lsize =', lsize, 'size =', size)

        if trace:
            print("creating petsc matrix of size (%d,%d)" % (lsize, size))
        jac_mat = PETSc.Mat().createPython([(lsize, size), (lsize, size)],
                                           comm=system.comm)
        if trace:
            print("petsc matrix creation DONE for %s" % vec_name)
        jac_mat.setPythonContext(self)
        jac_mat.setUp()

        if trace:  # pragma: no cover
            print("creating KSP object for system", system.path_name)

        ksp = self.ksp[vec_name] = PETSc.KSP().create(comm=system.comm)
        if trace:
            print("KSP creation DONE")

        ksp.setOperators(jac_mat)
        ksp.setType(self.options['ksp_type'])
        ksp.setGMRESRestart(1000)
        ksp.setPCSide(PETSc.PC.Side.RIGHT)
        ksp.setMonitor(Monitor(self))

        if trace:  # pragma: no cover
            print("ksp.getPC()")
            print("b_vec, x_vec size: %d" % lsize)
        pc_mat = ksp.getPC()
        pc_mat.setType('python')
        pc_mat.setPythonContext(self)

        print('=====================')

        return ksp
