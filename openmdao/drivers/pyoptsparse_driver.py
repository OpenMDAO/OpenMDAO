"""
OpenMDAO Wrapper for pyoptsparse.

pyoptsparse is based on pyOpt, which is an object-oriented framework for
formulating and solving nonlinear constrained optimization problems, with
additional MPI capability.
"""

from collections import OrderedDict
import json
import signal
import sys
import traceback

import numpy as np
from scipy.sparse import coo_matrix

try:
    from pyoptsparse import Optimization
except ImportError:
    Optimization = None

from openmdao.core.analysis_error import AnalysisError
from openmdao.core.driver import Driver, RecordingDebugging
import openmdao.utils.coloring as c_mod
from openmdao.utils.general_utils import simple_warning
from openmdao.utils.class_util import weak_method_wrapper
from openmdao.utils.mpi import FakeComm


# names of optimizers that use gradients
grad_drivers = {'CONMIN', 'FSQP', 'IPOPT', 'NLPQLP',
                'PSQP', 'SLSQP', 'SNOPT', 'NLPY_AUGLAG'}

# names of optimizers that allow multiple objectives
multi_obj_drivers = {'NSGA2'}

# All optimizers in pyoptsparse
optlist = ['ALPSO', 'CONMIN', 'FSQP', 'IPOPT', 'NLPQLP',
           'NSGA2', 'PSQP', 'SLSQP', 'SNOPT', 'NLPY_AUGLAG', 'NOMAD']

# All optimizers that require an initial run
run_required = ['NSGA2', 'ALPSO']

DEFAULT_OPT_SETTINGS = {}
DEFAULT_OPT_SETTINGS['IPOPT'] = {
    'hessian_approximation': 'limited-memory',
    'nlp_scaling_method': 'user-scaling',
    'linear_solver': 'mumps'
}

CITATIONS = """@article{Hwang_maud_2018
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
    SNOPT, NLPY_AUGLAG, NOMAD.
    Note that some of these are not open source and therefore not included
    in the pyoptsparse source code.

    pyOptSparseDriver supports the following:
        equality_constraints

        inequality_constraints

        two_sided_constraints

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

        Parameters
        ----------
        **kwargs : dict of keyword arguments
            Keyword arguments that will be mapped into the Driver options.
        """
        if Optimization is None:
            raise RuntimeError('pyOptSparseDriver is not available, pyOptsparse is not installed.')

        super(pyOptSparseDriver, self).__init__(**kwargs)

        # What we support
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

        self._indep_list = []
        self._quantities = []
        self.fail = False
        self._signal_cache = None
        self._user_termination_flag = False
        self._in_user_function = False

        self.cite = CITATIONS

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        self.options.declare('optimizer', default='SLSQP', values=optlist,
                             desc='Name of optimizers to use')
        self.options.declare('title', default='Optimization using pyOpt_sparse',
                             desc='Title of this optimization run')
        self.options.declare('print_results', types=bool, default=True,
                             desc='Print pyOpt results if True')
        self.options.declare('gradient method', default='openmdao',
                             values={'openmdao', 'pyopt_fd', 'snopt_fd'},
                             desc='Finite difference implementation to use')
        self.options.declare('user_terminate_signal', default=signal.SIGUSR1, allow_none=True,
                             desc='OS signal that triggers a clean user-termination. Only SNOPT'
                             'supports this option.')

        # Deprecated option
        self.options.declare('user_teriminate_signal', default=None, allow_none=True,
                             desc='OS signal that triggers a clean user-termination. Only SNOPT'
                             'supports this option.',
                             deprecation="The option 'user_teriminate_signal' was misspelled and "
                             "will be deprecated. Please use 'user_terminate_signal' instead.")

    def _setup_driver(self, problem):
        """
        Prepare the driver for execution.

        This is the final thing to run during setup.

        Parameters
        ----------
        problem : <Problem>
            Pointer to the containing problem.
        """
        super(pyOptSparseDriver, self)._setup_driver(problem)

        self.supports._read_only = False
        self.supports['gradients'] = self.options['optimizer'] in grad_drivers
        self.supports._read_only = True

        if len(self._objs) > 1 and self.options['optimizer'] not in multi_obj_drivers:
            raise RuntimeError('Multiple objectives have been added to pyOptSparseDriver'
                               ' but the selected optimizer ({0}) does not support'
                               ' multiple objectives.'.format(self.options['optimizer']))

        self._setup_tot_jac_sparsity()

        # Handle deprecated option.
        if self.options._dict['user_teriminate_signal']['value'] is not None:
            self.options['user_terminate_signal'] = \
                self.options._dict['user_teriminate_signal']['value']

    def run(self):
        """
        Excute pyOptsparse.

        Note that pyOpt controls the execution, and the individual optimizers
        (e.g., SNOPT) control the iteration.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        """
        problem = self._problem()
        model = problem.model
        relevant = model._relevant
        self.pyopt_solution = None
        self._total_jac = None
        self.iter_count = 0
        fwd = problem._mode == 'fwd'
        optimizer = self.options['optimizer']
        self._quantities = []

        self._check_for_missing_objective()

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
        if c_mod._use_total_sparsity:
            coloring = None
            if self._coloring_info['coloring'] is None and self._coloring_info['dynamic']:
                coloring = c_mod.dynamic_total_coloring(self, run_model=not model_ran,
                                                        fname=self._get_total_coloring_fname())

            if coloring is not None:
                # if the improvement wasn't large enough, don't use coloring
                pct = coloring._solves_info()[-1]
                info = self._coloring_info
                if info['min_improve_pct'] > pct:
                    info['coloring'] = info['static'] = None
                    simple_warning("%s: Coloring was deactivated.  Improvement of %.1f%% was less "
                                   "than min allowed (%.1f%%)." % (self.msginfo, pct,
                                                                   info['min_improve_pct']))

        comm = None if isinstance(problem.comm, FakeComm) else problem.comm
        opt_prob = Optimization(self.options['title'], weak_method_wrapper(self, '_objfunc'),
                                comm=comm)

        # Add all design variables
        param_meta = self._designvars
        self._indep_list = indep_list = list(param_meta)
        param_vals = self.get_design_var_values()

        for name, meta in param_meta.items():
            size = meta['global_size'] if meta['distributed'] else meta['size']
            opt_prob.addVarGroup(name, size, type='c',
                                 value=param_vals[name],
                                 lower=meta['lower'], upper=meta['upper'])

        opt_prob.finalizeDesignVariables()

        # Add all objectives
        objs = self.get_objective_values()
        for name in objs:
            opt_prob.addObj(name)
            self._quantities.append(name)

        # Calculate and save derivatives for any linear constraints.
        lcons = [key for (key, con) in con_meta.items() if con['linear']]
        if len(lcons) > 0:
            _lin_jacs = self._compute_totals(of=lcons, wrt=indep_list, return_format='dict')
            # convert all of our linear constraint jacs to COO format. Otherwise pyoptsparse will
            # do it for us and we'll end up with a fully dense COO matrix and very slow evaluation
            # of linear constraints!
            to_remove = []
            for jacdct in _lin_jacs.values():
                for n, subjac in jacdct.items():
                    if isinstance(subjac, np.ndarray):
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
            if fwd:
                wrt = [v for v in indep_list if name in relevant[param_meta[v]['ivc_source']]]
            else:
                rels = relevant[name]
                wrt = [v for v in indep_list if param_meta[v]['ivc_source'] in rels]

            if meta['linear']:
                jac = {w: _lin_jacs[name][w] for w in wrt}
                opt_prob.addConGroup(name, size, lower=lower, upper=upper,
                                     linear=True, wrt=wrt, jac=jac)
            else:
                if name in self._res_jacs:
                    resjac = self._res_jacs[name]
                    jac = {n: resjac[n] for n in wrt}
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

            if fwd:
                wrt = [v for v in indep_list if name in relevant[param_meta[v]['ivc_source']]]
            else:
                rels = relevant[name]
                wrt = [v for v in indep_list if param_meta[v]['ivc_source'] in rels]

            if meta['linear']:
                jac = {w: _lin_jacs[name][w] for w in wrt}
                opt_prob.addConGroup(name, size, upper=upper, lower=lower,
                                     linear=True, wrt=wrt, jac=jac)
            else:
                if name in self._res_jacs:
                    resjac = self._res_jacs[name]
                    jac = {n: resjac[n] for n in wrt}
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

        # Execute the optimization problem
        if self.options['gradient method'] == 'pyopt_fd':

            # Use pyOpt's internal finite difference
            # TODO: Need to get this from OpenMDAO
            # fd_step = problem.model.deriv_options['step_size']
            fd_step = 1e-6
            sol = opt(opt_prob, sens='FD', sensStep=fd_step, storeHistory=self.hist_file,
                      hotStart=self.hotstart_file)

        elif self.options['gradient method'] == 'snopt_fd':
            if self.options['optimizer'] == 'SNOPT':

                # Use SNOPT's internal finite difference
                # TODO: Need to get this from OpenMDAO
                # fd_step = problem.model.deriv_options['step_size']
                fd_step = 1e-6
                sol = opt(opt_prob, sens=None, sensStep=fd_step, storeHistory=self.hist_file,
                          hotStart=self.hotstart_file)

            else:
                raise Exception("SNOPT's internal finite difference can only be used with SNOPT")
        else:

            # Use OpenMDAO's differentiator for the gradient
            sol = opt(opt_prob, sens=weak_method_wrapper(self, '_gradfunc'),
                      storeHistory=self.hist_file, hotStart=self.hotstart_file)

        # Print results
        if self.options['print_results']:
            print(sol)

        # Pull optimal parameters back into framework and re-run, so that
        # framework is left in the right final state
        dv_dict = sol.getDVs()
        for name in indep_list:
            self.set_design_var(name, dv_dict[name])

        with RecordingDebugging(self._get_name(), self.iter_count, self) as rec:
            model.run_solve_nonlinear()
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

                # Record after getting obj and constraint to assure they have
                # been gathered in MPI.
                rec.abs = 0.0
                rec.rel = 0.0

        except Exception as msg:
            tb = traceback.format_exc()

            # Exceptions seem to be swallowed by the C code, so this
            # should give the user more info than the dreaded "segfault"
            print("Exception: %s" % str(msg))
            print(70 * "=", tb, 70 * "=")
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
            Dictionary of design variable values.
        func_dict : dict
            Dictionary of all functional variables evaluated at design point.

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
                                                 return_format='dict')
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
                new_sens = OrderedDict()
                res_jacs = self._res_jacs
                for okey in func_dict:
                    new_sens[okey] = newdv = OrderedDict()
                    for ikey in dv_dict:
                        if okey in res_jacs and ikey in res_jacs[okey]:
                            arr = sens_dict[okey][ikey]
                            coo = res_jacs[okey][ikey]
                            row, col, data = coo['coo']
                            coo['coo'][2] = arr[row, col].flatten()
                            newdv[ikey] = coo
                        elif okey in sens_dict:
                            newdv[ikey] = sens_dict[okey][ikey]
                sens_dict = new_sens

            if fail > 0:
                # We need to cobble together a sens_dict of the correct size.
                # Best we can do is return zeros.

                sens_dict = OrderedDict()
                for okey, oval in func_dict.items():
                    sens_dict[okey] = OrderedDict()
                    osize = len(oval)
                    for ikey, ival in dv_dict.items():
                        isize = len(ival)
                        sens_dict[okey][ikey] = np.zeros((osize, isize))

        except Exception as msg:
            tb = traceback.format_exc()

            # Exceptions seem to be swallowed by the C code, so this
            # should give the user more info than the dreaded "segfault"
            print("Exception: %s" % str(msg))
            print(70 * "=", tb, 70 * "=", flush=True)
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
        self._res_jacs = {}
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

        for res, resdict in total_sparsity.items():
            if res in self._objs:  # skip objectives
                continue
            self._res_jacs[res] = {}
            for dv, (rows, cols, shape) in resdict.items():
                rows = np.array(rows, dtype=int)
                cols = np.array(cols, dtype=int)

                self._res_jacs[res][dv] = {
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
