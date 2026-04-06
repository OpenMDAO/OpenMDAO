"""
Define the Brent class.

Based on implementation of the Brent algorithm in OpenMDAO 2.0 using brentq from scipy.
"""
import numpy as np
from scipy.optimize import brentq

from openmdao.recorders.recording_iteration_stack import Recording
from openmdao.solvers.solver import NonlinearSolver


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
    """

    SOLVER = 'NL: BRENT'

    def __init__(self, **kwargs):
        """
        Initialize all attributes.
        """
        super().__init__(**kwargs)

        self.state_target = ''
        self.norm = 1.0

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        super()._declare_options()

        self.options.declare('state_target', types=str, default='',
                             desc='Name of the implicit state to be solved')

        self.options.declare('lower_bound', default=0.0,
                             desc='Lower bound for the root search')
        self.options.declare('upper_bound', default=100.0,
                             desc='Upper bound for the root search')

        self.options.declare('lower_bound_target', default='',
                             desc='Openmdao path to the lower bound')
        self.options.declare('upper_bound_target', default='',
                             desc='Openmdao path to the upper bound')

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

        if self.options['state_target'].strip() == '':
            msg = f"{self.msginfo}: 'state_target' option in Brent solver of {pathname} must be specified."
            raise ValueError(msg)

        self.state_target = self.options['state_target']

        #self.var_lower_bound = None
        #var_lower_bound = self.options['var_lower_bound']
        #if var_lower_bound.strip() != '':
            #for var_name, meta in iteritems(sub.params):
                #if meta['top_promoted_name'] == var_lower_bound:
                    #self.var_lower_bound = var_name
                    #break
            #if self.var_lower_bound is None:
                #raise(ValueError("'var_lower_bound' variable '%s' was not found as a parameter on any component in %s"%(var_lower_bound, sub.pathname)))

        #self.var_upper_bound = None
        #var_upper_bound = self.options['var_upper_bound']
        #if var_upper_bound.strip() != '':
            #for var_name, meta in iteritems(sub.params):
                #if meta['top_promoted_name'] == var_upper_bound:
                    #self.var_upper_bound = var_name
                    #break
            #if self.var_upper_bound is None:
                #raise(ValueError("'var_lower_bound' variable '%s' was not found as a parameter on any component in %s"%(var_upper_bound, sub.pathname)))

    def _solve(self):
        """
        Run the iterative solver.
        """
        system = self._system()

        maxiter = self.options['maxiter']
        atol = self.options['atol']
        rtol = self.options['rtol']
        iprint = self.options['iprint']
        stall_limit = self.options['stall_limit']
        stall_tol = self.options['stall_tol']
        stall_tol_type = self.options['stall_tol_type']

        self._mpi_print_header()

        self._iter_count = 0
        norm0, norm = self._iter_initialize()

        self._norm0 = norm0

        self._mpi_print(self._iter_count, norm, norm / norm0)

        stalled = False
        stall_count = 0
        if stall_limit > 0:
            stall_norm = norm0

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

        # solver stalled.
        elif stalled:
            msg = (f"Solver '{self.SOLVER}' on system '{system.pathname}' stalled after "
                   f"{self._iter_count} iterations.")
            self.report_failure(msg)

        # Solver hit maxiter without meeting desired tolerances.
        elif norm > atol and norm / norm0 > rtol:
            self._convergence_failure()

        # Solver converged
        elif print_flag:
            if iprint == 1:
                print(prefix + ' Converged in {} iterations'.format(self._iter_count))
            elif iprint == 2:
                print(prefix + ' Converged')

    def _run_all_iterations(self):
        """
        Perform the operations in the iteration loop.
        """
        system = self._system()

        kwargs = {
            'maxiter': self.options['maxiter'],
            'a': self.options['lower_bound'],
            'b': self.options['upper_bound'],
            'full_output': False, # False, because we don't use the info, so just wastes operations
            'args': (system),
            'xtol': self.options['atol'],
            'rtol': self.options['rtol'],
        }

        xstar = brentq(self._eval, **kwargs)

        # Run the final point because last brentq point is a bracketing point.
        self._eval(xstar, system)

    def _eval(self, x, system):
        """Callback function for evaluating f(x)"""

        system.set_val(self.state_target, val=x)
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

    def zzsolve(self, params, unknowns, resids, system, metadata=None):
        """ Solves the system using the Brent Method.

        Args
        ----
        params : `VecWrapper`
            `VecWrapper` containing parameters. (p)

        unknowns : `VecWrapper`
            `VecWrapper` containing outputs and states. (u)

        resids : `VecWrapper`
            `VecWrapper` containing residuals. (r)

        system : `System`
            Parent `System` object.

        metadata : dict, optional
            Dictionary containing execution metadata (e.g. iteration coordinate).
        """
        self.sys = system
        self.metadata = metadata
        self.local_meta = create_local_meta(self.metadata, self.sys.pathname)
        self.sys.ln_solver.local_meta = self.local_meta
        idx = self.options['state_var_idx']

        if self.var_lower_bound is not None:
            lower = params[self.var_lower_bound]
        else:
            lower = self.options['lower_bound']

        if self.var_upper_bound is not None:
            upper = params[self.var_upper_bound]
        else:
            upper = self.options['upper_bound']

        kwargs = {'maxiter': self.options['maxiter'],
                  'a': lower,
                  'b': upper,
                  'full_output': False, # False, because we don't use the info, so just wastes operations
                  'args': (params, unknowns, resids)
                  }

        if self.options['xtol']:
            kwargs['xtol'] = self.options['xtol']
        if self.options['rtol'] > 0:
            kwargs['rtol'] = self.options['rtol']

        # Brent's method
        self.iter_count = 0

        # initial run to compute initial_norm
        self.sys.children_solve_nonlinear(self.local_meta)
        self.recorders.record_iteration(system, self.local_meta)

        # Evaluate Norm
        self.sys.apply_nonlinear(params, unknowns, resids)
        self.basenorm = resid_norm_0 = abs(resids._dat[self.s_var_name].val[idx])

        failed = False
        try:
            xstar = brentq(self._eval, **kwargs)
        except RuntimeError as err:
            msg = str(err)
            if 'different signs' in msg:
                raise
            failed = True

        self.sys = None

        resid_norm = abs(resids._dat[self.s_var_name].val[idx])

        if self.options['iprint'] > 0:

            if not failed:
                msg = 'Converged'

            self.print_norm(self.print_name, system, self.iter_count,
                            resid_norm, resid_norm_0, msg=msg)

        if failed and self.options['err_on_maxiter']:
            raise AnalysisError(msg)

    def _zzeval(self, x, params, unknowns, resids):
        """Callback function for evaluating f(x)"""

        idx = self.options['state_var_idx']
        self.iter_count += 1
        update_local_meta(self.local_meta, (self.iter_count, ))

        unknowns._dat[self.s_var_name].val[idx] = x

        self.sys.children_solve_nonlinear(self.local_meta)
        self.sys.apply_nonlinear(params, unknowns, resids)

        self.recorders.record_iteration(self.sys, self.local_meta)

        if self.options['iprint'] == 2:
            normval = abs(resids._dat[self.s_var_name].val[idx])
            self.print_norm(self.print_name, self.sys, self.iter_count, normval,
                            self.basenorm)

        return resids._dat[self.s_var_name].val[idx]