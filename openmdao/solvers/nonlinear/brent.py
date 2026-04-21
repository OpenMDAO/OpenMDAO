"""
Define the Brent class.

Based on implementation of the Brent algorithm in OpenMDAO 2.0 using brentq from scipy.
"""
import os

import networkx as nx
import numpy as np
from scipy.optimize import brentq

from openmdao.recorders.recording_iteration_stack import Recording
from openmdao.solvers.solver import NonlinearSolver
from openmdao.utils.om_warnings import issue_warning


CITATION = """@BOOK{Brent1973-dm,
  title     = "Algorithms for Minimization Without Derivatives",
  author    = "Brent, R P",
  publisher = "Prentice-Hall",
  pages     = "3-4",
  year      =  1973,
  address   = "Englewood Cliffs, NJ"
}"""


class BrentSolver(NonlinearSolver):
    """
    Brent solver.

    Root finding using Brent's method. This is a specialized solver that can only converge a single
    scalar residual. You must specify the name of the implicit state-variable via the `state_target`
    option. You must specify `lower_bound` and `upper_bound` for the upper and lower.

    Parameters
    ----------
    **kwargs : dict
        Options dictionary.

    Parameters
    ----------
    state_target : str
        Relative openmdao varpath to the state.
    upper_target : str or None
        Relative openmdao varpath to the upper bound. Only used if the lower bound is computed
        somewhere in the model.
    lower_target : str or None
        Relative openmdao varpath to the lower bound.  Only used if the lower bound is computed
        somewhere in the model.
    """

    SOLVER = 'NL: BRENT'

    def __init__(self, **kwargs):
        """
        Initialize all attributes.
        """
        super().__init__(**kwargs)

        self.state_target = None
        self.upper_target = None
        self.lower_target = None

        self.norm = 1.0

        self.cite = CITATION

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        super()._declare_options()

        self.options.declare('state_target', types=str, allow_none=True, default=None,
                             desc='Name of the implicit state to be solved')

        self.options.declare('lower_bound', default=0.0,
                             desc='Lower bound for the root search')
        self.options.declare('upper_bound', default=100.0,
                             desc='Upper bound for the root search')

        self.options.declare('lower_bound_target', allow_none=True, default=None,
                             desc='Openmdao path to the lower bound. When specified, this takes '
                             'precedence over the value specified in lower_bound.')
        self.options.declare('upper_bound_target', allow_none=True, default=None,
                             desc='Openmdao path to the upper bound. When specified, this takes '
                             'precedence over the value specified in upper_bound.')

        # Remove unsupported options.
        self.options.undeclare('stall_limit')
        self.options.undeclare('stall_tol')
        self.options.undeclare('stall_tol_type')

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

        if self.options['state_target'] is None:
            msg = f"{self.msginfo}: 'state_target' option in Brent solver must be specified."
            raise ValueError(msg)

        self.state_target = self.options['state_target']

        if self.state_target not in system._outputs:
            msg = f"{self.msginfo}: 'state_target' variable '{self.state_target}' not found."
            raise ValueError(msg)

        self.upper_target = upper = self.options['upper_bound_target']
        if upper is not None and upper not in system._outputs:
            msg = f"{self.msginfo}: 'upper_bound_target' variable '{upper}' not found."
            raise ValueError(msg)

        self.lower_target = lower = self.options['lower_bound_target']
        if lower is not None and lower not in system._outputs:
            msg = f"{self.msginfo}: 'lower_bound_target' variable '{lower}' not found."
            raise ValueError(msg)

        # Make sure we only have one state.
        valid = True
        n_imp = 0
        from openmdao.core.implicitcomponent import ImplicitComponent
        for sub in system.system_iter(recurse=False, typ=ImplicitComponent):
            if n_imp > 0:
                # Found more than 1 implicitcomponent
                valid = False
                break
            n_imp = len(sub._outputs)
            if n_imp > 1:
                # This implicitcomponent has more than 1 state
                valid = False
                break

        if not valid:
            msg = f"{self.msginfo}: Brent can only solve 1 single implicit state."
            raise ValueError(msg)

        # Check for cycles without an implicit state. Such a cycle could work in theory if the
        # user knew where openmdao broke the cycle, but it is better to break the cycle and
        # insert a balance.
        graph = system.compute_sys_graph()
        if not nx.is_directed_acyclic_graph(graph) and n_imp == 0:
            msg = f"{self.msginfo}: Brent does not support cycles."
            raise ValueError(msg)

    def _solve(self):
        """
        Run the iterative solver.

        The base class is overridden because scipy brentq controls the iteration.
        """
        system = self._system()

        atol = self.options['atol']
        rtol = self.options['rtol']
        iprint = self.options['iprint']

        self._mpi_print_header()

        self._iter_count = 0
        norm0, norm = self._iter_initialize()

        self._norm0 = norm0

        self._mpi_print(self._iter_count, norm, norm / norm0)

        # Brentq is external and controls convergence.
        self._run_all_iterations()

        # flag for the print statements. we only print on root if USE_PROC_FILES is not set to True
        print_flag = system.comm.rank == 0 or os.environ.get('USE_PROC_FILES')

        prefix = self._solver_info.prefix + self.SOLVER
        norm = self.norm

        # Solver terminated early because a Nan in the norm doesn't satisfy the while-loop
        # conditionals.
        if np.isinf(norm) or np.isnan(norm):
            self._inf_nan_failure()

        # Solver hit maxiter without meeting desired tolerances.
        elif norm > atol and norm / norm0 > rtol:
            self._convergence_failure()

        # Solver converged
        elif print_flag:
            if iprint == 1:
                print(prefix + ' Converged in {} iterations'.format(self._iter_count))
            elif iprint == 2:
                print(prefix + ' Converged')

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
        if self.options['debug_print']:
            self._err_cache['inputs'] = system._inputs._copy_vars()
            self._err_cache['outputs'] = system._outputs._copy_vars()

        with Recording(type(self).__name__, 0, self) as rec:
            self._solver_info.append_solver()

            # should call the subsystems solve before computing the first residual
            self._gs_iter()

            self._solver_info.pop()

            self._run_apply()
            norm = self._iter_get_norm()

            rec.abs = norm
            norm0 = norm if norm != 0.0 else 1.0
            rec.rel = norm / norm0

        return norm0, norm

    def _run_all_iterations(self):
        """
        Perform the operations in the iteration loop.
        """
        system = self._system()

        lower = self.options['lower_bound']
        if self.lower_target is not None:
            lower = system._outputs[self.lower_target]

        upper = self.options['upper_bound']
        if self.upper_target is not None:
            upper = system._outputs[self.upper_target]

        kwargs = {
            'maxiter': self.options['maxiter'],
            'a': lower,
            'b': upper,
            'full_output': False, # False, because we don't use the info, so just wastes operations
            'args': (system),
            'xtol': self.options['atol'],
            'rtol': self.options['rtol'],
        }

        try:
            xstar = brentq(self._eval, **kwargs)

            # Run the final point because last brentq point is a bracketing point.
            self._eval(xstar, system)

        except RuntimeError as err:
            # Let OpenMDAO handle nonconvergence.
            # Print actual error text in case it has some useful info from scipy.
            issue_warning(f"RuntimError from scipy brentq.  Error was: {err}")

    def _eval(self, x, system):
        """Callback function for evaluating f(x)"""

        system._outputs[self.state_target] = x
        norm0 = self._norm0

        # Run the model.
        with Recording(type(self).__name__, self._iter_count, self) as rec:
            self._solver_info.append_solver()
            self._gs_iter()
            self._solver_info.pop()

            self._iter_count += 1
            self._run_apply()
            norm = self._iter_get_norm()

            # Save the norm values in the context manager so they can also be recorded.
            rec.abs = norm
            if norm0 == 0:
                norm0 = 1
            rec.rel = norm / norm0

        self.norm = norm
        self._mpi_print(self._iter_count, norm, norm / norm0)

        return system._residuals[self.state_target]
