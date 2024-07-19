"""
OpenMDAO Wrapper for pyoptsparse.

pyoptsparse is based on pyOpt, which is an object-oriented framework for
formulating and solving nonlinear constrained optimization problems, with
additional MPI capability.
"""
import pathlib
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

from openmdao.core.constants import INT_DTYPE, _DEFAULT_REPORTS_DIR, _ReprClass
from openmdao.core.analysis_error import AnalysisError
from openmdao.core.driver import Driver, RecordingDebugging, filter_by_meta
from openmdao.core.group import Group
from openmdao.utils.class_util import WeakMethodWrapper
from openmdao.utils.mpi import FakeComm, MPI
from openmdao.utils.om_warnings import issue_warning, warn_deprecation
from openmdao.utils.reports_system import get_reports_dir

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
run_required = {'NSGA2'}
if pyoptsparse_version is None or pyoptsparse_version < Version('2.9.4'):
    run_required.add('ALPSO')  # ALPSO bug fixed in v2.9.4

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
    _nl_responses : list
        Contains the objectives plus nonlinear constraints.
    _signal_cache : <Function>
        Cached function pointer that was assigned as handler for signal defined in option
        user_terminate_signal.
    _total_jac_sparsity : dict, str, or None
        Specifies sparsity of sub-jacobians of the total jacobian.
    _user_termination_flag : bool
        This is set to True when the user sends a signal to terminate the job.
    _model_ran : bool
        This is set to True after the full model has been run at least once.
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

        # We save the pyopt_solution so that it can be queried later on.
        self.pyopt_solution = None

        # we have to return NANs in order for some optimizers that don't respect
        # the fail flag (e.g. IPOPT) to recognize a bad point and respond accordingly
        self._fill_NANs = False

        self._nl_responses = []
        self.fail = False
        self._signal_cache = None
        self._user_termination_flag = False
        self._in_user_function = False
        self._check_jac = False
        self._exc_info = None
        self._total_jac_format = 'dict'
        self._total_jac_sparsity = None
        self._model_ran = False

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
        self.options.declare('print_results', types=(bool, str), default=True,
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
        self.options.declare('hist_file', types=str, default=None, allow_none=True,
                             desc='File location for saving pyopt_sparse optimization history. '
                                  'Default is None for no output.')
        self.options.declare('hotstart_file', types=str, default=None, allow_none=True,
                             desc='File location of a pyopt_sparse optimization history to use '
                                  'to hot start the optimization. Default is None.')
        self.options.declare('output_dir', types=(str, _ReprClass), default=_DEFAULT_REPORTS_DIR,
                             allow_none=True,
                             desc='Directory location of pyopt_sparse output files.'
                             'Default is ./reports_directory/problem_name.')

    @property
    def hist_file(self):
        """
        Get the 'hist_file' option for this driver.
        """
        warn_deprecation("The 'hist_file' attribute is deprecated. "
                         "Use the 'hist_file' option instead.")
        return self.options['hist_file']

    @hist_file.setter
    def hist_file(self, file_name):
        """
        Set the 'hist_file' option for this driver.
        """
        warn_deprecation("The 'hist_file' attribute is deprecated. "
                         "Use the 'hist_file' option instead.")
        self.options['hist_file'] = file_name

    @property
    def hotstart_file(self):
        """
        Get the 'hotstart_file' option for this driver.
        """
        warn_deprecation("The 'hotstart_file' attribute is deprecated. "
                         "Use the 'hotstart_file' option instead.")
        return self.options['hotstart_file']

    @hotstart_file.setter
    def hotstart_file(self, file_name):
        """
        Set the 'hotstart_file' option for this driver.
        """
        warn_deprecation("The 'hotstart_file' attribute is deprecated. "
                         "Use the 'hotstart_file' option instead.")
        self.options['hotstart_file'] = file_name

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

        self._model_ran = False
        self._setup_tot_jac_sparsity()

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
        self.result.reset()
        problem = self._problem()
        model = problem.model
        relevance = model._relevance

        self.pyopt_solution = None
        self._total_jac = None
        self._total_jac_linear = None
        self.iter_count = 0
        self._nl_responses = []

        optimizer = self.options['optimizer']

        self._fill_NANs = not respects_fail_flag[self.options['optimizer']]

        self._check_for_missing_objective()
        self._check_for_invalid_desvar_values()
        self._check_jac = self.options['singular_jac_behavior'] in ['error', 'warn']

        linear_constraints = [key for key, con in self._cons.items() if con['linear']]

        # Only need initial run if we have linear constraints or if we are using an optimizer that
        # doesn't perform one initially.
        model_ran = False
        if optimizer in run_required or linear_constraints:
            with RecordingDebugging(self._get_name(), self.iter_count, self) as rec:
                self._run_solve_nonlinear()
                rec.abs = 0.0
                rec.rel = 0.0
                model_ran = True
            self.iter_count += 1

        self._model_ran = model_ran
        self._coloring_info.run_model = not model_ran

        comm = None if isinstance(problem.comm, FakeComm) else problem.comm
        opt_prob = Optimization(self.options['title'], WeakMethodWrapper(self, '_objfunc'),
                                comm=comm)

        input_vals = self.get_design_var_values()

        for name, meta in self._designvars.items():
            # translate absolute var names to promoted names for pyoptsparse
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
            opt_prob.addObj(model._get_prom_name(name))
            self._nl_responses.append(name)

        # Calculate and save derivatives for any linear constraints.
        if linear_constraints:
            _lin_jacs = self._compute_totals(of=linear_constraints, wrt=list(self._lin_dvs),
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

        # # compute dynamic simul deriv coloring
        problem.get_total_coloring(self._coloring_info, run_model=not model_ran)

        bad_resps = [n for n in relevance._no_dv_responses if n in self._cons]
        bad_cons = [n for n, m in self._cons.items() if m['source'] in bad_resps]

        if bad_cons:
            issue_warning(f"Equality constraint(s) {sorted(bad_cons)} do not depend on any design "
                          "variables and were not added to the optimization.")

            for name in bad_cons:
                del self._cons[name]
                del self._responses[name]

        eqcons = {n: m for n, m in self._cons.items() if m['equals'] is not None}
        if eqcons:
            # Add all equality constraints
            for name, meta in eqcons.items():
                size = meta['global_size'] if meta['distributed'] else meta['size']
                lower = upper = meta['equals']

                # set equality constraints as reverse seeds to see what dvs are relevant
                with relevance.seeds_active(rev_seeds=meta['source']):

                    if meta['linear']:
                        wrts = [v for v in self._lin_dvs
                                if relevance.is_relevant(self._lin_dvs[v]['source'])]
                        jac = {w: _lin_jacs[name][w] for w in wrts}
                        opt_prob.addConGroup(name, size,
                                             lower=lower - _y_intercepts[name],
                                             upper=upper - _y_intercepts[name],
                                             linear=True, wrt=wrts, jac=jac)
                    else:
                        wrts = [v for v in self._nl_dvs
                                if relevance.is_relevant(self._nl_dvs[v]['source'])]

                        if name in self._con_subjacs:
                            resjac = self._con_subjacs[name]
                            jac = {n: resjac[n] for n in wrts}
                        else:
                            jac = None

                        opt_prob.addConGroup(name, size, lower=lower, upper=upper, wrt=wrts,
                                             jac=jac)
                        self._nl_responses.append(name)

        ineqcons = {n: m for n, m in self._cons.items() if m['equals'] is None}
        if ineqcons:
            # Add all inequality constraints
            for name, meta in ineqcons.items():
                size = meta['global_size'] if meta['distributed'] else meta['size']

                # Bounds - double sided is supported
                lower = meta['lower']
                upper = meta['upper']

                # set inequality constraints as reverse seeds to see what dvs are relevant
                with relevance.seeds_active(rev_seeds=(meta['source'],)):

                    if meta['linear']:
                        wrts = [v for v in self._lin_dvs
                                if relevance.is_relevant(self._lin_dvs[v]['source'])]
                        jac = {w: _lin_jacs[name][w] for w in wrts}
                        opt_prob.addConGroup(name, size,
                                             upper=upper - _y_intercepts[name],
                                             lower=lower - _y_intercepts[name],
                                             linear=True, wrt=wrts, jac=jac)
                    else:
                        wrts = [v for v in self._nl_dvs
                                if relevance.is_relevant(self._nl_dvs[v]['source'])]
                        if name in self._con_subjacs:
                            resjac = self._con_subjacs[name]
                            jac = {n: resjac[n] for n in wrts}
                        else:
                            jac = None

                        opt_prob.addConGroup(name, size, upper=upper, lower=lower,
                                             wrt=wrts, jac=jac)
                        self._nl_responses.append(name)

        # Instantiate the requested optimizer
        try:
            _tmp = __import__('pyoptsparse', globals(), locals(), [optimizer], 0)
            opt = getattr(_tmp, optimizer)()

        except Exception as err:
            # Change whatever pyopt gives us to an ImportError, give it a readable message,
            # but raise with the original traceback.
            msg = "Optimizer %s is not available in this installation." % optimizer
            raise ImportError(msg)

        # Need to tell optimizer where to put its .out files
        if self.options['output_dir'] is None:
            output_dir = "."
        elif self.options['output_dir'] == _DEFAULT_REPORTS_DIR:
            problem = self._problem()
            default_output_dir = pathlib.Path(get_reports_dir()).joinpath(problem._name)
            pathlib.Path(default_output_dir).mkdir(parents=True, exist_ok=True)
            output_dir = str(default_output_dir)
        else:
            output_dir = self.options['output_dir']

        optimizers_and_output_files = {
            # ALPSO uses a single option `filename` to determine name of both output files
            'ALPSO': [('filename', 'ALPSO.out')],
            'CONMIN': [('IFILE', 'CONMIN.out')],
            'IPOPT': [('output_file', 'IPOPT.out')],
            'PSQP': [('IFILE', 'PSQP.out')],
            'SLSQP': [('IFILE', 'SLSQP.out')],
            'SNOPT': [('Print file', 'SNOPT_print.out'), ('Summary file', 'SNOPT_summary.out')]
        }

        if optimizer in optimizers_and_output_files:
            for opt_setting_name, output_file_name in optimizers_and_output_files[optimizer]:
                if self.opt_settings.get(opt_setting_name) is None:
                    self.opt_settings[opt_setting_name] = f'{output_dir}/{output_file_name}'

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
                sol = opt(opt_prob, sens='FD', sensStep=fd_step,
                          storeHistory=self.options['hist_file'],
                          hotStart=self.options['hotstart_file'])

            elif self.options['gradient_method'] == 'snopt_fd':
                if self.options['optimizer'] == 'SNOPT':

                    # Use SNOPT's internal finite difference
                    # TODO: Need to get this from OpenMDAO
                    # fd_step = problem.model.deriv_options['step_size']
                    fd_step = 1e-6
                    sol = opt(opt_prob, sens=None, sensStep=fd_step,
                              storeHistory=self.options['hist_file'],
                              hotStart=self.options['hotstart_file'])

                else:
                    msg = "SNOPT's internal finite difference can only be used with SNOPT"
                    self._exc_info = (Exception, Exception(msg), None)
            else:

                # Use OpenMDAO's differentiator for the gradient
                sol = opt(opt_prob, sens=WeakMethodWrapper(self, '_gradfunc'),
                          storeHistory=self.options['hist_file'],
                          hotStart=self.options['hotstart_file'])

        except Exception as c:
            if self._exc_info is None:
                raise

        if self._exc_info is not None:
            exc_info = self._exc_info
            self._exc_info = None
            if exc_info[2] is None:
                raise exc_info[1]
            raise exc_info[1].with_traceback(exc_info[2])

        # Print results
        if self.options['print_results']:
            if not MPI or model.comm.rank == 0:
                if self.options['print_results'] == 'minimal':
                    if hasattr(sol, 'summary_str'):
                        print(sol.summary_str(minimal_print=True))
                    else:
                        print('minimal_print is not available for this solution')
                        print(sol)
                else:
                    print(sol)

        # Pull optimal parameters back into framework and re-run, so that
        # framework is left in the right final state
        dv_dict = sol.getDVs()
        for name in self._designvars:
            self.set_design_var(name, dv_dict[model._get_prom_name(name)])

        with RecordingDebugging(self._get_name(), self.iter_count, self) as rec:
            try:
                self._run_solve_nonlinear()
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
            else:
                # exit status may be the empty string for optimizers that don't support it
                if exit_status and exit_status > 2:
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
            for name in self._nl_dvs:
                self.set_design_var(name, dv_dict[model._get_prom_name(name)])

            # print("Setting DV")
            # print(dv_dict)

            # Check if we caught a termination signal while SNOPT was running.
            if self._user_termination_flag:
                func_dict = self.get_objective_values()
                func_dict.update(self.get_constraint_values(lintype='nonlinear'))
                # convert func_dict to use promoted names
                func_dict = model._prom_names_dict(func_dict)
                return func_dict, 2

            # Execute the model
            with RecordingDebugging(self._get_name(), self.iter_count, self) as rec:
                self.iter_count += 1
                try:
                    self._in_user_function = True
                    # deactivate the relevance if we haven't run the full model yet, so that
                    # the full model will run at least once.
                    with model._relevance.nonlinear_active('iter', active=self._model_ran):
                        self._run_solve_nonlinear()
                        self._model_ran = True

                # Let the optimizer try to handle the error
                except AnalysisError:
                    model._clear_iprint()
                    fail = 1

                # User requested termination
                except UserRequestedException:
                    model._clear_iprint()
                    fail = 2

                # Record after getting obj and constraint to assure they have
                # been gathered in MPI.
                rec.abs = 0.0
                rec.rel = 0.0

        except Exception:
            if self._exc_info is None:  # avoid overwriting an earlier exception
                self._exc_info = sys.exc_info()
            fail = 1

        func_dict = self.get_objective_values()
        func_dict.update(self.get_constraint_values(lintype='nonlinear'))

        if fail > 0 and self._fill_NANs:
            for name in func_dict:
                func_dict[name].fill(np.nan)

        # convert func_dict to use promoted names
        func_dict = model._prom_names_dict(func_dict)

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
        model = prob.model
        fail = 0
        sens_dict = {}

        try:

            # Check if we caught a termination signal while SNOPT was running.
            if self._user_termination_flag:
                return {}, 2

            try:
                self._in_user_function = True
                sens_dict = self._compute_totals(of=self._nl_responses,
                                                 wrt=self._nl_dvs,
                                                 return_format=self._total_jac_format)

                # First time through, check for zero row/col.
                if self._check_jac and self._total_jac is not None:
                    for subsys in model.system_iter(include_self=True, recurse=True, typ=Group):
                        if subsys._has_approx:
                            break
                    else:
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
                con_subjacs = self._con_subjacs

                for okey in self._nl_responses:
                    new_sens[okey] = newdv = {}
                    for ikey in self._nl_dvs.keys():
                        if okey in con_subjacs and ikey in con_subjacs[okey]:
                            arr = sens_dict[okey][ikey]
                            coo = con_subjacs[okey][ikey]
                            row, col, _ = coo['coo']
                            coo['coo'][2] = arr[row, col].flatten()
                            newdv[ikey] = coo
                        elif okey in sens_dict:
                            newdv[ikey] = sens_dict[okey][ikey]
                sens_dict = new_sens

        except Exception:
            if self._exc_info is None:  # avoid overwriting an earlier exception
                self._exc_info = sys.exc_info()
            fail = 1

        if fail > 0:
            # We need to cobble together a sens_dict of the correct size.
            # Best we can do is return zeros or NaNs.
            for okey in self._nl_responses:
                if okey not in sens_dict:
                    sens_dict[okey] = {}
                oval = func_dict[model._get_prom_name(okey)]
                osize = len(oval)
                for ikey in self._designvars.keys():
                    ival = dv_dict[model._get_prom_name(ikey)]
                    isize = len(ival)
                    if ikey not in sens_dict[okey] or self._fill_NANs:
                        sens_dict[okey][ikey] = np.zeros((osize, isize))
                        if self._fill_NANs:
                            sens_dict[okey][ikey].fill(np.nan)

        # convert sens_dict to use promoted names
        sens_dict = model._prom_names_jac(sens_dict)

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
        self._con_subjacs = {}
        coloring = coloring if coloring is not None else self._get_static_coloring()
        if coloring is not None:
            total_sparsity = coloring.get_subjac_sparsity()
            if self._total_jac_sparsity is not None:
                raise RuntimeError("Total jac sparsity was set in both _total_coloring"
                                   " and _setup_tot_jac_sparsity.")
        elif self._total_jac_sparsity is not None:
            if isinstance(self._total_jac_sparsity, str):
                with open(self._total_jac_sparsity, 'r') as f:
                    self._total_jac_sparsity = json.load(f)
            total_sparsity = self._total_jac_sparsity

        if total_sparsity is None:
            return

        use_approx = self._problem().model._owns_approx_of is not None

        # exclude linear cons
        for con, conmeta in filter_by_meta(self._cons.items(), 'linear', exclude=True):
            self._con_subjacs[con] = {}
            consrc = conmeta['source']
            for dv, dvmeta in self._designvars.items():
                if use_approx:
                    dvsrc = dvmeta['source']
                    rows, cols, shape = total_sparsity[consrc][dvsrc]
                else:
                    rows, cols, shape = total_sparsity[con][dv]
                self._con_subjacs[con][dv] = {
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
