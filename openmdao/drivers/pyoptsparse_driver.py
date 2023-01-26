"""
OpenMDAO Wrapper for pyoptsparse.

pyoptsparse is based on pyOpt, which is an object-oriented framework for
formulating and solving nonlinear constrained optimization problems, with
additional MPI capability.
"""

import sys
import json
import signal
from packaging.version import Version

import numpy as np
from scipy.sparse import coo_matrix

try:
    import pyoptsparse
    Optimization = pyoptsparse.Optimization
except ImportError:
    pyoptsparse = None
except Exception as err:
    pyoptsparse = err

from openmdao.core.constants import INT_DTYPE
from openmdao.core.analysis_error import AnalysisError
from openmdao.core.driver import Driver, RecordingDebugging
import openmdao.utils.coloring as c_mod
from openmdao.utils.class_util import WeakMethodWrapper
from openmdao.utils.mpi import FakeComm, MPI
from openmdao.utils.general_utils import _src_or_alias_name
from openmdao.utils.om_warnings import issue_warning

# what version of pyoptspare are we working with
if pyoptsparse and hasattr(pyoptsparse, '__version__'):
    pyoptsparse_version = Version(pyoptsparse.__version__)
else:
    pyoptsparse_version = None

# All optimizers in pyoptsparse
optlist = {'ALPSO', 'CONMIN', 'IPOPT', 'NLPQLP', 'NSGA2', 'ParOpt', 'PSQP', 'SLSQP', 'SNOPT'}

if pyoptsparse_version is None or pyoptsparse_version < Version('2.6.0'):
    optlist.add('NOMAD')

if pyoptsparse_version is None or pyoptsparse_version < Version('2.1.2'):
    optlist.update({'FSQP', 'NLPY_AUGLAG'})

# names of optimizers that use gradients
grad_drivers = optlist.intersection({'CONMIN', 'FSQP', 'IPOPT', 'NLPQLP', 'PSQP',
                                     'SLSQP', 'SNOPT', 'NLPY_AUGLAG', 'ParOpt'})

# names of optimizers that allow multiple objectives
multi_obj_drivers = {'NSGA2'}

# All optimizers that require an initial run
run_required = {'NSGA2', 'ALPSO'}

# The pyoptsparse API provides for an optional 'fail' flag in the return value of
# objective and gradient functions, but this flag is only used by a subset of the
# available optimizers. If the flag is not respected by the optimizer, we have to
# return NaN values to indicate a bad evaluation.
respects_fail_flag = {
    # Currently supported optimizers (v2.9.0)
    'ALPSO': False,
    'CONMIN': False,
    'IPOPT': False,
    'NLPQLP': False,
    'NSGA2': False,
    'PSQP': False,
    'ParOpt': True,
    'SLSQP': False,
    'SNOPT': True,           # as of v2.0.0, requires SNOPT 7.7
    'FSQP': False,           # no longer supported as of v2.1.2
    'NLPY_AUGLAG': False,    # no longer supported as of v2.1.2
    'NOMAD': False           # no longer supported as of v2.6.0
}

DEFAULT_OPT_SETTINGS = {}
DEFAULT_OPT_SETTINGS['IPOPT'] = {
    'hessian_approximation': 'limited-memory',
    'nlp_scaling_method': 'user-scaling',
    'linear_solver': 'mumps'
}

CITATIONS = """@article{Wu_pyoptsparse_2020,
    author = {Neil Wu and Gaetan Kenway and Charles A. Mader and John Jasa and
     Joaquim R. R. A. Martins},
    title = {{pyOptSparse:} A {Python} framework for large-scale constrained
     nonlinear optimization of sparse systems},
    journal = {Journal of Open Source Software},
    volume = {5},
    number = {54},
    month = {October},
    year = {2020},
    pages = {2564},
    doi = {10.21105/joss.02564},
    publisher = {The Open Journal},
}

@article{Hwang_maud_2018
    author = {Hwang, John T. and Martins, Joaquim R.R.A.},
    title = "{A Computational Architecture for Coupling Heterogeneous
             Numerical Models and Computing Coupled Derivatives}",
    journal = "{ACM Trans. Math. Softw.}",
    volume = {44},
    number = {4},
    month = jun,
    year = {2018},
    pages = {37:1--37:39},
    articleno = {37},
    numpages = {39},
    doi = {10.1145/3182393},
    publisher = {ACM},
}
"""

DEFAULT_SIGNAL = None


class UserRequestedException(Exception):
    """
    User Requested Exception.

    This exception indicates that the user has requested that SNOPT/pyoptsparse ceases
    model execution and reports to SNOPT that execution should be terminated.
    """

    pass


class pyOptSparseDriver(Driver):
    """
    Driver wrapper for pyoptsparse.

    Pyoptsparse is based on pyOpt, which
    is an object-oriented framework for formulating and solving nonlinear
    constrained optimization problems, with additional MPI capability.
    pypptsparse has interfaces to the following optimizers:
    ALPSO, CONMIN, FSQP, IPOPT, NLPQLP, NSGA2, PSQP, SLSQP,
    SNOPT, NLPY_AUGLAG, NOMAD, ParOpt.
    Note that some of these are not open source and therefore not included
    in the pyoptsparse source code.

    pyOptSparseDriver supports the following:
        equality_constraints

        inequality_constraints

        two_sided_constraints

    Parameters
    ----------
    **kwargs : dict of keyword arguments
        Keyword arguments that will be mapped into the Driver options.

    Attributes
    ----------
    fail : bool
        Flag that indicates failure of most recent optimization.
    hist_file : str or None
        File location for saving pyopt_sparse optimization history.
        Default is None for no output.
    hotstart_file : str
        Optional file to hot start the optimization.
    opt_settings : dict
        Dictionary for setting optimizer-specific options.
    pyopt_solution : Solution
        Pyopt_sparse solution object.
    _fill_NANs : bool
        Used internally to control when to return NANs for a bad evaluation.
    _check_jac : bool
        Used internally to control when to perform singular checks on computed total derivs.
    _exc_info : 3 item tuple
        Storage for exception and traceback information for exception that was raised in the
        _objfunc or _gradfunc callbacks.
    _in_user_function :bool
        This is set to True at the start of a pyoptsparse callback to _objfunc and _gradfunc, and
        restored to False at the finish of each callback.
    _indep_list : list
        List of design variables.
    _quantities : list
        Contains the objectives plus nonlinear constraints.
    _signal_cache : <Function>
        Cached function pointer that was assigned as handler for signal defined in option
        user_terminate_signal.
    _user_termination_flag : bool
        This is set to True when the user sends a signal to terminate the job.
    """

    def __init__(self, **kwargs):
        """
        Initialize pyopt.
        """
        if pyoptsparse is None:
            # pyoptsparse is not installed
            raise RuntimeError('pyOptSparseDriver is not available, pyOptsparse is not installed.')

        if isinstance(pyoptsparse, Exception):
            # there is some other issue with the pyoptsparse installation
            raise pyoptsparse

        super().__init__(**kwargs)

        # What we support
        self.supports['optimization'] = True
        self.supports['inequality_constraints'] = True
        self.supports['equality_constraints'] = True
        self.supports['multiple_objectives'] = True
        self.supports['two_sided_constraints'] = True
        self.supports['linear_constraints'] = True
        self.supports['simultaneous_derivatives'] = True
        self.supports['total_jac_sparsity'] = True

        # What we don't support yet
        self.supports['active_set'] = False
        self.supports['integer_design_vars'] = False
        self.supports['distributed_design_vars'] = False
        self.supports._read_only = True

        # The user places optimizer-specific settings in here.
        self.opt_settings = {}

        # The user can set a file name here to store history
        self.hist_file = None

        # The user can set a file here to hot start the optimization
        # with a history file
        self.hotstart_file = None

        # We save the pyopt_solution so that it can be queried later on.
        self.pyopt_solution = None

        # we have to return NANs in order for some optimizers that don't respect
        # the fail flag (e.g. IPOPT) to recognize a bad point and respond accordingly
        self._fill_NANs = False

        self._indep_list = []
        self._quantities = []
        self.fail = False
        self._signal_cache = None
        self._user_termination_flag = False
        self._in_user_function = False
        self._check_jac = False
        self._exc_info = None
        self._total_jac_format = 'dict'

        self.cite = CITATIONS

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        self.options.declare('optimizer', default='SLSQP', values=optlist,
                             desc='Name of optimizers to use')
        self.options.declare('title', default='Optimization using pyOpt_sparse',
                             desc='Title of this optimization run')
        self.options.declare('print_opt_prob', types=bool, default=False,
                             desc='Print the opt problem summary before running the optimization')
        self.options.declare('print_results', types=bool, default=True,
                             desc='Print pyOpt results if True')
        self.options.declare('gradient_method', default='openmdao',
                             values={'openmdao', 'pyopt_fd', 'snopt_fd'},
                             desc='Finite difference implementation to use')
        self.options.declare('user_terminate_signal', default=DEFAULT_SIGNAL, allow_none=True,
                             desc='OS signal that triggers a clean user-termination. '
                                  'Only SNOPT supports this option.')
        self.options.declare('singular_jac_behavior', default='warn',
                             values=['error', 'warn', 'ignore'],
                             desc='Defines behavior of a zero row/col check after first call to'
                                  'compute_totals:'
                                  'error - raise an error.'
                                  'warn - raise a warning.'
                                  "ignore - don't perform check.")
        self.options.declare('singular_jac_tol', default=1e-16,
                             desc='Tolerance for zero row/column check.')

    def _setup_driver(self, problem):
        """
        Prepare the driver for execution.

        This is the final thing to run during setup.

        Parameters
        ----------
        problem : <Problem>
            Pointer to the containing problem.
        """
        super()._setup_driver(problem)

        self.supports._read_only = False
        self.supports['gradients'] = self.options['optimizer'] in grad_drivers
        self.supports._read_only = True

        if len(self._objs) > 1 and self.options['optimizer'] not in multi_obj_drivers:
            raise RuntimeError('Multiple objectives have been added to pyOptSparseDriver'
                               ' but the selected optimizer ({0}) does not support'
                               ' multiple objectives.'.format(self.options['optimizer']))

        self._setup_tot_jac_sparsity()

    def get_driver_objective_calls(self):
        """
        Return number of objective evaluations made during a driver run.

        Returns
        -------
        int
            Number of objective evaluations made during a driver run.
        """
        return self.pyopt_solution.userObjCalls if self.pyopt_solution else None

    def get_driver_derivative_calls(self):
        """
        Return number of derivative evaluations made during a driver run.

        Returns
        -------
        int
            Number of derivative evaluations made during a driver run.
        """
        return self.pyopt_solution.userSensCalls if self.pyopt_solution else None

    def run(self):
        """
        Excute pyOptsparse.

        Note that pyOpt controls the execution, and the individual optimizers
        (e.g., SNOPT) control the iteration.

        Returns
        -------
        bool
            Failure flag; True if failed to converge, False is successful.
        """
        problem = self._problem()
        model = problem.model
        relevant = model._relevant
        self.pyopt_solution = None
        self._total_jac = None
        self.iter_count = 0
        fwd = problem._mode == 'fwd'
        self._quantities = []

        optimizer = self.options['optimizer']
        self._fill_NANs = not respects_fail_flag[self.options['optimizer']]

        self._check_for_missing_objective()
        self._check_for_invalid_desvar_values()
        self._check_jac = self.options['singular_jac_behavior'] in ['error', 'warn']

        # Only need initial run if we have linear constraints or if we are using an optimizer that
        # doesn't perform one initially.
        con_meta = self._cons
        model_ran = False
        if optimizer in run_required or np.any([con['linear'] for con in self._cons.values()]):
            with RecordingDebugging(self._get_name(), self.iter_count, self) as rec:
                # Initial Run
                model.run_solve_nonlinear()
                rec.abs = 0.0
                rec.rel = 0.0
                model_ran = True
            self.iter_count += 1

        # compute dynamic simul deriv coloring or just sparsity if option is set
        coloring = self._get_coloring(run_model=not model_ran)

        comm = None if isinstance(problem.comm, FakeComm) else problem.comm
        opt_prob = Optimization(self.options['title'], WeakMethodWrapper(self, '_objfunc'),
                                comm=comm)

        # Add all design variables
        dv_meta = self._designvars
        self._indep_list = indep_list = list(dv_meta)
        input_vals = self.get_design_var_values()

        for name, meta in dv_meta.items():
            size = meta['global_size'] if meta['distributed'] else meta['size']
            if pyoptsparse_version is None or pyoptsparse_version < Version('2.6.1'):
                opt_prob.addVarGroup(name, size, type='c',
                                     value=input_vals[name],
                                     lower=meta['lower'], upper=meta['upper'])
            else:
                opt_prob.addVarGroup(name, size, varType='c',
                                     value=input_vals[name],
                                     lower=meta['lower'], upper=meta['upper'])

        if pyoptsparse_version is None or pyoptsparse_version < Version('2.5.1'):
            opt_prob.finalizeDesignVariables()
        else:
            opt_prob.finalize()

        # Add all objectives
        objs = self.get_objective_values()
        for name in objs:
            opt_prob.addObj(name)
            self._quantities.append(name)

        # Calculate and save derivatives for any linear constraints.
        lcons = [key for (key, con) in con_meta.items() if con['linear']]
        if len(lcons) > 0:
            _lin_jacs = self._compute_totals(of=lcons, wrt=indep_list,
                                             return_format=self._total_jac_format)
            _con_vals = self.get_constraint_values(lintype='linear')
            # convert all of our linear constraint jacs to COO format. Otherwise pyoptsparse will
            # do it for us and we'll end up with a fully dense COO matrix and very slow evaluation
            # of linear constraints!
            _y_intercepts = {}
            for name, jacdct in _lin_jacs.items():
                _y_intercepts[name] = _con_vals[name]
                for n, subjac in jacdct.items():
                    if isinstance(subjac, np.ndarray):
                        _y_intercepts[name] -= subjac.dot(input_vals[n])
                        # we can safely use coo_matrix to automatically convert the ndarray
                        # since our linear constraint jacs are constant, so zeros won't become
                        # nonzero during the optimization.
                        mat = coo_matrix(subjac)
                        if mat.row.size > 0:
                            # convert to 'coo' format here to avoid an emphatic warning
                            # by pyoptsparse.
                            jacdct[n] = {'coo': [mat.row, mat.col, mat.data], 'shape': mat.shape}

        # Add all equality constraints
        for name, meta in con_meta.items():
            if meta['equals'] is None:
                continue
            size = meta['global_size'] if meta['distributed'] else meta['size']
            lower = upper = meta['equals']
            path = meta['source']
            if fwd:
                wrt = [v for v in indep_list if path in relevant[dv_meta[v]['source']]]
            else:
                rels = relevant[path]
                wrt = [v for v in indep_list if dv_meta[v]['source'] in rels]

            if not wrt:
                issue_warning(f"Equality constraint '{name}' does not depend on any design "
                              "variables and was not added to the optimization.")
                continue

            if meta['linear']:
                jac = {w: _lin_jacs[name][w] for w in wrt}
                opt_prob.addConGroup(name, size,
                                     lower=lower - _y_intercepts[name],
                                     upper=upper - _y_intercepts[name],
                                     linear=True, wrt=wrt, jac=jac)
            else:
                if name in self._res_subjacs:
                    resjac = self._res_subjacs[name]
                    jac = {n: resjac[dv_meta[n]['source']] for n in wrt}
                else:
                    jac = None

                opt_prob.addConGroup(name, size, lower=lower, upper=upper, wrt=wrt, jac=jac)
                self._quantities.append(name)

        # Add all inequality constraints
        for name, meta in con_meta.items():
            if meta['equals'] is not None:
                continue
            size = meta['global_size'] if meta['distributed'] else meta['size']

            # Bounds - double sided is supported
            lower = meta['lower']
            upper = meta['upper']

            path = meta['source']

            if fwd:
                wrt = [v for v in indep_list if path in relevant[dv_meta[v]['source']]]
            else:
                rels = relevant[path]
                wrt = [v for v in indep_list if dv_meta[v]['source'] in rels]

            if not wrt:
                issue_warning(f"Inequality constraint '{name}' does not depend on any design "
                              "variables and was not added to the optimization.")
                continue

            if meta['linear']:
                jac = {w: _lin_jacs[name][w] for w in wrt}
                opt_prob.addConGroup(name, size,
                                     upper=upper - _y_intercepts[name],
                                     lower=lower - _y_intercepts[name],
                                     linear=True, wrt=wrt, jac=jac)
            else:
                if name in self._res_subjacs:
                    resjac = self._res_subjacs[name]
                    jac = {n: resjac[dv_meta[n]['source']] for n in wrt}
                else:
                    jac = None
                opt_prob.addConGroup(name, size, upper=upper, lower=lower, wrt=wrt, jac=jac)
                self._quantities.append(name)

        # Instantiate the requested optimizer
        try:
            _tmp = __import__('pyoptsparse', globals(), locals(), [optimizer], 0)
            opt = getattr(_tmp, optimizer)()

        except Exception as err:
            # Change whatever pyopt gives us to an ImportError, give it a readable message,
            # but raise with the original traceback.
            msg = "Optimizer %s is not available in this installation." % optimizer
            raise ImportError(msg)

        # Process any default optimizer-specific settings.
        if optimizer in DEFAULT_OPT_SETTINGS:
            for name, value in DEFAULT_OPT_SETTINGS[optimizer].items():
                if name not in self.opt_settings:
                    self.opt_settings[name] = value

        # Set optimization options
        for option, value in self.opt_settings.items():
            opt.setOption(option, value)

        # Print the pyoptsparse optimization problem summary before running the optimization.
        # This allows users to confirm their optimization setup.
        if self.options['print_opt_prob']:
            if not MPI or model.comm.rank == 0:
                print(opt_prob)

        self._exc_info = None
        try:

            # Execute the optimization problem
            if self.options['gradient_method'] == 'pyopt_fd':

                # Use pyOpt's internal finite difference
                # TODO: Need to get this from OpenMDAO
                # fd_step = problem.model.deriv_options['step_size']
                fd_step = 1e-6
                sol = opt(opt_prob, sens='FD', sensStep=fd_step, storeHistory=self.hist_file,
                          hotStart=self.hotstart_file)

            elif self.options['gradient_method'] == 'snopt_fd':
                if self.options['optimizer'] == 'SNOPT':

                    # Use SNOPT's internal finite difference
                    # TODO: Need to get this from OpenMDAO
                    # fd_step = problem.model.deriv_options['step_size']
                    fd_step = 1e-6
                    sol = opt(opt_prob, sens=None, sensStep=fd_step, storeHistory=self.hist_file,
                              hotStart=self.hotstart_file)

                else:
                    msg = "SNOPT's internal finite difference can only be used with SNOPT"
                    self._exc_info = (Exception, Exception(msg), None)
            else:

                # Use OpenMDAO's differentiator for the gradient
                sol = opt(opt_prob, sens=WeakMethodWrapper(self, '_gradfunc'),
                          storeHistory=self.hist_file, hotStart=self.hotstart_file)

        except Exception as c:
            if not self._exc_info:
                raise

        if self._exc_info:
            if self._exc_info[2] is None:
                raise self._exc_info[1]
            raise self._exc_info[1].with_traceback(self._exc_info[2])

        # Print results
        if self.options['print_results']:
            if not MPI or model.comm.rank == 0:
                print(sol)

        # Pull optimal parameters back into framework and re-run, so that
        # framework is left in the right final state
        dv_dict = sol.getDVs()
        for name in indep_list:
            self.set_design_var(name, dv_dict[name])

        with RecordingDebugging(self._get_name(), self.iter_count, self) as rec:
            try:
                model.run_solve_nonlinear()
            except AnalysisError:
                model._clear_iprint()

            rec.abs = 0.0
            rec.rel = 0.0
        self.iter_count += 1

        # Save the most recent solution.
        self.pyopt_solution = sol

        try:
            exit_status = sol.optInform['value']
            self.fail = False

            # These are various failed statuses.
            if optimizer == 'IPOPT':
                if exit_status not in {0, 1}:
                    self.fail = True
            elif exit_status > 2:
                self.fail = True

        except KeyError:
            # optimizers other than pySNOPT may not populate this dict
            pass

        # revert signal handler to cached version
        sigusr = self.options['user_terminate_signal']
        if sigusr is not None:
            signal.signal(sigusr, self._signal_cache)
            self._signal_cache = None   # to prevent memory leak test from failing

        return self.fail

    def _objfunc(self, dv_dict):
        """
        Compute the objective function and constraints.

        This function is passed to pyOpt's Optimization object and is called
        from its optimizers.

        Parameters
        ----------
        dv_dict : dict
            Dictionary of design variable values.

        Returns
        -------
        func_dict : dict
            Dictionary of all functional variables evaluated at design point.

        fail : int
            0 for successful function evaluation
            1 for unsuccessful function evaluation
        """
        model = self._problem().model
        fail = 0

        # Note: we place our handler as late as possible so that codes that run in the
        # workflow can place their own handlers.
        sigusr = self.options['user_terminate_signal']
        if sigusr is not None and self._signal_cache is None:
            self._signal_cache = signal.getsignal(sigusr)
            signal.signal(sigusr, self._signal_handler)

        try:
            for name in self._indep_list:
                self.set_design_var(name, dv_dict[name])

            # print("Setting DV")
            # print(dv_dict)

            # Check if we caught a termination signal while SNOPT was running.
            if self._user_termination_flag:
                func_dict = self.get_objective_values()
                func_dict.update(self.get_constraint_values(lintype='nonlinear'))
                return func_dict, 2

            # Execute the model
            with RecordingDebugging(self._get_name(), self.iter_count, self) as rec:
                self.iter_count += 1
                try:
                    self._in_user_function = True
                    model.run_solve_nonlinear()

                # Let the optimizer try to handle the error
                except AnalysisError:
                    model._clear_iprint()
                    fail = 1

                # User requested termination
                except UserRequestedException:
                    model._clear_iprint()
                    fail = 2

                func_dict = self.get_objective_values()
                func_dict.update(self.get_constraint_values(lintype='nonlinear'))

                if fail > 0 and self._fill_NANs:
                    for name in func_dict:
                        func_dict[name].fill(np.NAN)

                # Record after getting obj and constraint to assure they have
                # been gathered in MPI.
                rec.abs = 0.0
                rec.rel = 0.0

        except Exception:
            self._exc_info = sys.exc_info()
            fail = 1
            func_dict = {}

        # print("Functions calculated")
        # print(dv_dict)
        # print(func_dict, flush=True)

        self._in_user_function = False
        return func_dict, fail

    def _gradfunc(self, dv_dict, func_dict):
        """
        Compute the gradient of the objective function and constraints.

        This function is passed to pyOpt's Optimization object and is called
        from its optimizers.

        Parameters
        ----------
        dv_dict : dict
            Dictionary of design variable values. All keys are sources.
        func_dict : dict
            Dictionary of all functional variables evaluated at design point. Keys are
            sources and aliases.

        Returns
        -------
        sens_dict : dict
            Dictionary of dictionaries for gradient of each dv/func pair

        fail : int
            0 for successful function evaluation
            1 for unsuccessful function evaluation
        """
        prob = self._problem()
        fail = 0

        try:

            # Check if we caught a termination signal while SNOPT was running.
            if self._user_termination_flag:
                return {}, 2

            try:
                self._in_user_function = True
                sens_dict = self._compute_totals(of=self._quantities,
                                                 wrt=self._indep_list,
                                                 return_format=self._total_jac_format)

                # First time through, check for zero row/col.
                if self._check_jac:
                    raise_error = self.options['singular_jac_behavior'] == 'error'
                    self._total_jac.check_total_jac(raise_error=raise_error,
                                                    tol=self.options['singular_jac_tol'])
                    self._check_jac = False

            # Let the optimizer try to handle the error
            except AnalysisError:
                prob.model._clear_iprint()
                fail = 1

            # User requested termination
            except UserRequestedException:
                prob.model._clear_iprint()
                fail = 2

            else:
                # if we don't convert to 'coo' here, pyoptsparse will do a
                # conversion of our dense array into a fully dense 'coo', which is bad.
                # TODO: look into getting rid of all of these conversions!
                new_sens = {}
                res_subjacs = self._res_subjacs
                for okey in func_dict:
                    new_sens[okey] = newdv = {}
                    for ikey in dv_dict:
                        ikey_src = self._designvars[ikey]['source']
                        if okey in res_subjacs and ikey_src in res_subjacs[okey]:
                            arr = sens_dict[okey][ikey]
                            coo = res_subjacs[okey][ikey_src]
                            row, col, _ = coo['coo']
                            coo['coo'][2] = arr[row, col].flatten()
                            newdv[ikey] = coo
                        elif okey in sens_dict:
                            newdv[ikey] = sens_dict[okey][ikey]
                sens_dict = new_sens

            if fail > 0:
                # We need to cobble together a sens_dict of the correct size.
                # Best we can do is return zeros.

                sens_dict = {}
                for okey, oval in func_dict.items():
                    sens_dict[okey] = {}
                    osize = len(oval)
                    for ikey, ival in dv_dict.items():
                        isize = len(ival)
                        sens_dict[okey][ikey] = np.zeros((osize, isize))
                        if self._fill_NANs:
                            sens_dict[okey][ikey].fill(np.NAN)

        except Exception:
            self._exc_info = sys.exc_info()
            fail = 1
            sens_dict = {}

        # print("Derivatives calculated")
        # print(dv_dict)
        # print(sens_dict, flush=True)
        self._in_user_function = False
        return sens_dict, fail

    def _get_name(self):
        """
        Get name of current optimizer.

        Returns
        -------
        str
            The name of the current optimizer.
        """
        return "pyOptSparse_" + self.options['optimizer']

    def _get_ordered_nl_responses(self):
        """
        Return the names of nonlinear responses in the order used by the driver.

        Default order is objectives followed by nonlinear constraints.  This is used for
        simultaneous derivative coloring and sparsity determination.

        Returns
        -------
        list of str
            The nonlinear response names in order.
        """
        nl_order = list(self._objs)
        neq_order = []
        for n, meta in self._cons.items():
            if 'linear' not in meta or not meta['linear']:
                if meta['equals'] is not None:
                    nl_order.append(n)
                else:
                    neq_order.append(n)

        nl_order.extend(neq_order)

        return nl_order

    def _setup_tot_jac_sparsity(self, coloring=None):
        """
        Set up total jacobian subjac sparsity.

        Parameters
        ----------
        coloring : Coloring or None
            Current coloring.
        """
        total_sparsity = None
        self._res_subjacs = {}
        coloring = coloring if coloring is not None else self._get_static_coloring()
        if coloring is not None:
            total_sparsity = coloring.get_subjac_sparsity()
            if self._total_jac_sparsity is not None:
                raise RuntimeError("Total jac sparsity was set in both _total_coloring"
                                   " and _total_jac_sparsity.")
        elif self._total_jac_sparsity is not None:
            if isinstance(self._total_jac_sparsity, str):
                with open(self._total_jac_sparsity, 'r') as f:
                    self._total_jac_sparsity = json.load(f)
            total_sparsity = self._total_jac_sparsity

        if total_sparsity is None:
            return

        for res, dvdict in total_sparsity.items():  # res are 'driver' names (prom name or alias)
            if res in self._objs:  # skip objectives
                continue
            # if res in self._responses and self._responses[res]['alias'] is not None:
            #     res = self._responses[res]['source']
            self._res_subjacs[res] = {}
            for dv, (rows, cols, shape) in dvdict.items():  # dvs are src names
                rows = np.array(rows, dtype=INT_DTYPE)
                cols = np.array(cols, dtype=INT_DTYPE)

                self._res_subjacs[res][dv] = {
                    'coo': [rows, cols, np.zeros(rows.size)],
                    'shape': shape,
                }

    def _signal_handler(self, signum, frame):
        # Subsystems (particularly external codes) may declare their own signal handling, so
        # execute the cached handler first.
        if self._signal_cache is not signal.Handlers.SIG_DFL:
            self._signal_cache(signum, frame)

        self._user_termination_flag = True
        if self._in_user_function:
            raise UserRequestedException('User requested termination.')
