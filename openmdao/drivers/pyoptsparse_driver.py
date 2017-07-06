"""
OpenMDAO Wrapper for pyoptsparse.

pyoptsparse is based on pyOpt, which is an object-oriented framework for
formulating and solving nonlinear constrained optimization problems, with
additional MPI capability.
"""

from __future__ import print_function
from collections import OrderedDict
import traceback

from six import iteritems
from six.moves import range

import numpy as np
import scipy as sp

from pyoptsparse import Optimization

from openmdao.core.driver import Driver

# names of optimizers that use gradients
grad_drivers = {'CONMIN', 'FSQP', 'IPOPT', 'NLPQLP',
                'PSQP', 'SLSQP', 'SNOPT', 'NLPY_AUGLAG'}

# names of optimizers that allow multiple objectives
multi_obj_drivers = {'NSGA2'}


def _check_imports():
    """
    Dynamically remove optimizers we don't have.
    """
    optlist = ['ALPSO', 'CONMIN', 'FSQP', 'IPOPT', 'NLPQLP',
               'NSGA2', 'PSQP', 'SLSQP', 'SNOPT', 'NLPY_AUGLAG', 'NOMAD']

    for optimizer in optlist[:]:
        try:
            __import__('pyoptsparse', globals(), locals(), [optimizer], 0)
        except ImportError:
            optlist.remove(optimizer)

    return optlist


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

    Options
    -------
    options['optimizer'] :  str('SLSQP')
        Name of optimizers to use
    options['print_results'] :  bool(True)
        Print pyOpt results if True
    options['gradient method'] :  str('openmdao', 'pyopt_fd', 'snopt_fd')
        Finite difference implementation to use ('snopt_fd' may only be used with SNOPT)
    options['title'] :  str('Optimization using pyOpt_sparse')
        Title of this optimization run

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
    problem : <Problem>
        Pointer to the containing problem.
    supports : <OptionsDictionary>
        Provides a consistant way for drivers to declare what features they support.
    pyopt_solution : Solution
        Pyopt_sparse solution object.
    _cons : dict
        Contains all constraint info.
    _designvars : dict
        Contains all design variable info.
    _indep_list : list
        List of design variables.
    _objs : dict
        Contains all objective info.
    _quantities : list
        Contains the objectives plus nonlinear constraints.
    _responses : dict
        Contains all response info.
    """

    def __init__(self):
        """
        Initialize pyopt.
        """
        super(pyOptSparseDriver, self).__init__()

        # What we support
        self.supports['inequality_constraints'] = True
        self.supports['equality_constraints'] = True
        self.supports['multiple_objectives'] = True
        self.supports['two_sided_constraints'] = True
        self.supports['linear_constraints'] = True

        # What we don't support yet
        self.supports['active_set'] = False
        self.supports['integer_design_vars'] = False

        # User Options
        self.options.declare('optimizer', default='SLSQP', values=_check_imports(),
                             desc='Name of optimizers to use')
        self.options.declare('title', default='Optimization using pyOpt_sparse',
                             desc='Title of this optimization run')
        self.options.declare('print_results', type_=bool, default=True,
                             desc='Print pyOpt results if True')
        self.options.declare('gradient method', default='openmdao',
                             values={'openmdao', 'pyopt_fd', 'snopt_fd'},
                             desc='Finite difference implementation to use')

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

        self.supports['gradients'] = self.options['optimizer'] in grad_drivers

        if len(self._objs) > 1 and self.options['optimizer'] not in multi_obj_drivers:
            raise RuntimeError('Multiple objectives have been added to pyOptSparseDriver'
                               ' but the selected optimizer ({0}) does not support'
                               ' multiple objectives.'.format(self.options['optimizer']))

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
        problem = self._problem
        model = self._problem.model
        self.pyopt_solution = None
        self.iter_count = 0

        # Initial Run
        model._solve_nonlinear()

        opt_prob = Optimization(self.options['title'], self._objfunc)

        # Add all design variables
        param_meta = self._designvars
        self._indep_list = indep_list = list(param_meta)
        param_vals = self.get_design_var_values()

        for name, meta in iteritems(param_meta):
            opt_prob.addVarGroup(name, meta['size'], type='c',
                                 value=param_vals[name],
                                 lower=meta['lower'], upper=meta['upper'])

        opt_prob.finalizeDesignVariables()

        # Add all objectives
        objs = self.get_objective_values()
        for name in objs:
            opt_prob.addObj(name)
            self._quantities.append(name)

        # Calculate and save derivatives for any linear constraints.
        con_meta = self._cons
        lcons = [key for (key, con) in iteritems(con_meta) if con['linear'] is True]
        if len(lcons) > 0:
            _lin_jacs = problem._compute_total_derivs(of=lcons, wrt=indep_list,
                                                      return_format='dict')

        # Add all equality constraints
        self.active_tols = {}
        eqcons = OrderedDict((key, con) for (key, con) in iteritems(con_meta)
                             if con['equals'] is not None)
        for name, meta in iteritems(eqcons):
            size = meta['size']
            lower = upper = meta['equals']

            if meta['linear']:
                opt_prob.addConGroup(name, size, lower=lower, upper=upper,
                                     linear=True,
                                     jac=_lin_jacs[name])
            else:
                opt_prob.addConGroup(name, size, lower=lower, upper=upper)
                self._quantities.append(name)

        # Add all inequality constraints
        iqcons = OrderedDict((key, con) for (key, con) in iteritems(con_meta)
                             if con['equals'] is None)
        for name, meta in iteritems(iqcons):
            size = meta['size']

            # Bounds - double sided is supported
            lower = meta['lower']
            upper = meta['upper']

            if meta['linear']:
                opt_prob.addConGroup(name, size, upper=upper, lower=lower,
                                     linear=True,
                                     jac=_lin_jacs[name])
            else:
                opt_prob.addConGroup(name, size, upper=upper, lower=lower)
                self._quantities.append(name)

        # Instantiate the requested optimizer
        optimizer = self.options['optimizer']
        try:
            _tmp = __import__('pyoptsparse', globals(), locals(), [optimizer], 0)
            opt = getattr(_tmp, optimizer)()

        except ImportError:
            msg = "Optimizer %s is not available in this installation." % optimizer
            raise ImportError(msg)

        # Set optimization options
        for option, value in self.opt_settings.items():
            opt.setOption(option, value)

        self.opt_prob = opt_prob
        self.opt = opt

        # Execute the optimization problem
        if self.options['gradient method'] == 'pyopt_fd':

            # Use pyOpt's internal finite difference
            # TODO: Need to get this from OpenMDAO
            # fd_step = problem.root.deriv_options['step_size']
            fd_step = 1e-6
            sol = opt(opt_prob, sens='FD', sensStep=fd_step, storeHistory=self.hist_file,
                      hotStart=self.hotstart_file)

        elif self.options['gradient method'] == 'snopt_fd':
            if self.options['optimizer'] == 'SNOPT':

                # Use SNOPT's internal finite difference
                # TODO: Need to get this from OpenMDAO
                # fd_step = problem.root.deriv_options['step_size']
                fd_step = 1e-6
                sol = opt(opt_prob, sens=None, sensStep=fd_step, storeHistory=self.hist_file,
                          hotStart=self.hotstart_file)

            else:
                msg = "SNOPT's internal finite difference can only be used with SNOPT"
                raise Exception(msg)
        else:

            # Use OpenMDAO's differentiator for the gradient
            sol = opt(opt_prob, sens=self._gradfunc, storeHistory=self.hist_file,
                      hotStart=self.hotstart_file)

        # Print results
        if self.options['print_results']:
            print(sol)

        # Pull optimal parameters back into framework and re-run, so that
        # framework is left in the right final state
        dv_dict = sol.getDVs()
        for name in indep_list:
            val = dv_dict[name]
            self.set_design_var(name, val)

        model._solve_nonlinear()

        # Save the most recent solution.
        self.pyopt_solution = sol
        try:
            exit_status = sol.optInform['value']
            self.fail = False

            # These are various failed statuses.
            if exit_status > 2:
                self.fail = True

        except KeyError:
            # Nothing is here, so something bad happened!
            self.fail = True

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
        model = self._problem.model
        fail = 0

        try:
            for name in self._indep_list:
                self.set_design_var(name, dv_dict[name])

            # print("Setting DV")
            # print(dv_dict)

            # Execute the model
            self.iter_count += 1
            model._solve_nonlinear()

            func_dict = self.get_objective_values()
            func_dict.update(self.get_constraint_values(lintype='nonlinear'))

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
        # print(func_dict)
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
        prob = self._problem
        fail = 0

        try:

            sens_dict = self._compute_total_derivs(of=self._quantities,
                                                   wrt=self._indep_list,
                                                   return_format='dict')

        except Exception as msg:
            tb = traceback.format_exc()

            # Exceptions seem to be swallowed by the C code, so this
            # should give the user more info than the dreaded "segfault"
            print("Exception: %s" % str(msg))
            print(70 * "=", tb, 70 * "=")
            sens_dict = {}

        # print("Derivatives calculated")
        # print(dv_dict)
        # print(sens_dict)
        return sens_dict, fail
