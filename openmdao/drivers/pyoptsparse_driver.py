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

import scipy as sp
import numpy as np

from pyoptsparse import Optimization

from openmdao.core.driver import Driver

# names of optimizers that use gradients
grad_drivers = {'CONMIN', 'FSQP', 'IPOPT', 'NLPQLP',
                'PSQP', 'SLSQP', 'SNOPT', 'NLPY_AUGLAG'}

# names of optimizers that allow multiple objectives
multi_obj_drivers = {'NSGA2'}


def _check_imports():
    """ Dynamically remove optimizers we don't have.
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
    """ Driver wrapper for pyoptsparse.

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
    hist_file : string or None
        File location for saving pyopt_sparse optimization history.
        Default is None for no output.
    opt_settings : dict
        Dictionary for setting optimizer-specific options.
    problem : <Problem>
        Pointer to the containing problem.
    supports : <OptionsDictionary>
        Provides a consistant way for drivers to declare what features they support.
    _designvars : dict
        Contains all design variable info.
    _cons : dict
        Contains all constraint info.
    _objs : dict
        Contains all objective info.
    _responses : dict
        Contains all response info.
    """

    def __init__(self):
        """Initialize pyopt"""

        super(pyOptSparseDriver, self).__init__()

        # What we support
        self.supports['inequality_constraints'] = True
        self.supports['equality_constraints'] = True
        self.supports['multiple_objectives'] = True
        self.supports['two_sided_constraints'] = True
        self.supports['active_set'] = True

        # TODO: Support these
        # self.supports['linear_constraints'] = True
        # self.supports['integer_design_vars'] = False

        # User Options
        self.options.declare('optimizer', value='SLSQP', values=_check_imports(),
                             desc='Name of optimizers to use')
        self.options.declare('title', value='Optimization using pyOpt_sparse',
                             desc='Title of this optimization run')
        self.options.declare('print_results', type_=bool, value=True,
                             desc='Print pyOpt results if True')
        self.options.declare('gradient method', value='openmdao',
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

        # Cache the jacs of linear constraints
        self.lin_jacs = OrderedDict()

        # Support for active-set performance improvements.
        self.active_tols = {}

        # self.quantities = []
        # self.metadata = None
        # self.exit_flag = 0
        # self.sparsity = OrderedDict()
        # self.sub_sparsity = OrderedDict()

    def _setup_driver(self, problem):
        """Prepare the driver for execution.

        This is the final thing to run during setup.

        Parameters
        ----------
        problem : <`Problem`>
            Pointer to the containing problem.
        """
        super(pyOptSparseDriver, self)._setup_driver(problem)

        self.supports['gradients'] = self.options['optimizer'] in grad_drivers

        if len(self._objs) > 1 and self.options['optimizer'] not in multi_obj_drivers:
            raise RuntimeError('Multiple objectives have been added to pyOptSparseDriver'
                               ' but the selected optimizer ({0}) does not support'
                               ' multiple objectives.'.format(self.options['optimizer']))

    def run(self):
        """Excute pyOptsparse.

        Note that pyOpt controls the execution, and the individual optimizers
        (e.g., SNOPT) control the iteration.
        """
        problem = self.problem
        model = self.problem.model
        self.pyopt_solution = None
        self.iter_count = 0

        # Initial Run
        model._solve_nonlinear()

        opt_prob = Optimization(self.options['title'], self._objfunc)

        # Add all design variables
        param_meta = self._designvars
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

        # Calculate and save derivatives for any linear constraints.
        con_meta = self._cons
        lcons = OrderedDict((key, con) for (key, con) in iteritems(con_meta)
                            if con['linear']==True)
        if len(lcons) > 0:
            self.lin_jacs = problem.compute_total_derivs(of=lcons, wrt=indep_list,
                                                         return_format='dict')

        # Add all equality constraints
        self.active_tols = {}
        eqcons = OrderedDict((key, con) for (key, con) in iteritems(con_meta)
                            if con['equals'])
        for name, meta in iteritems(eqcons):
            size = meta['size']
            lower = upper = meta['equals']

            if meta['linear']:
                opt_prob.addConGroup(name, size, lower=lower, upper=upper,
                                     linear=True, #wrt=wrt,
                                     jac=self.lin_jacs[name])
            else:

                opt_prob.addConGroup(name, size, lower=lower, upper=upper)

            active_tol = meta.get('active_tol')
            if active_tol:
                self.active_tols[name] = active_tol

        # Add all inequality constraints
        iqcons = OrderedDict((key, con) for (key, con) in iteritems(con_meta)
                            if not con['equals'])
        for name, meta in iteritems(iqcons):
            size = meta['size']

            # Bounds - double sided is supported
            lower = meta['lower']
            upper = meta['upper']

            if meta['linear']:
                opt_prob.addConGroup(name, size, upper=upper, lower=lower,
                                     linear=True, #wrt=wrt,
                                     jac=self.lin_jacs[name])
            else:

                opt_prob.addConGroup(name, size, upper=upper, lower=lower)

            active_tol = meta.get('active_tol')
            if active_tol is not None:
                self.active_tols[name] = active_tol

        # Instantiate the requested optimizer
        optimizer = self.options['optimizer']
        try:
            _tmp = __import__('pyoptsparse', globals(), locals(), [optimizer], 0)
            opt = getattr(_tmp, optimizer)()
        except ImportError:
            msg = "Optimizer %s is not available in this installation." % \
                   optimizer
            raise ImportError(msg)

        #Set optimization options
        for option, value in self.opt_settings.items():
            opt.setOption(option, value)

        self.opt_prob = opt_prob

        # Execute the optimization problem
        if self.options['gradient method'] == 'pyopt_fd':

            # Use pyOpt's internal finite difference
            fd_step = problem.root.deriv_options['step_size']
            sol = opt(opt_prob, sens='FD', sensStep=fd_step, storeHistory=self.hist_file,
                      hotStart=self.hotstart_file)

        elif self.options['gradient method'] == 'snopt_fd':
            if self.options['optimizer']=='SNOPT':

                # Use SNOPT's internal finite difference
                fd_step = problem.root.deriv_options['step_size']
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
            self.set_desvar(name, val)

        with self.root._dircontext:
            self.root.solve_nonlinear(metadata=self.metadata)

        # Save the most recent solution.
        self.pyopt_solution = sol
        try:
            exit_status = sol.optInform['value']
            self.exit_flag = 1
            if exit_status > 2: # bad
                self.exit_flag = 0
        except KeyError: #nothing is here, so something bad happened!
            self.exit_flag = 0

    def _objfunc(self, dv_dict):
        """ Function that evaluates and returns the objective function and
        constraints. This function is passed to pyOpt's Optimization object
        and is called from its optimizers.

        Args
        ----
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

        fail = 0
        system = self.problem.model

        try:
            for name in self.indep_list:
                self.set_desvar(name, dv_dict[name])

            # Execute the model
            #print("Setting DV")
            #print(dv_dict)

            self.iter_count += 1

            model._solve_nonlinear()

            func_dict = self.get_objectives() # this returns a new OrderedDict
            func_dict.update(self.get_constraints())

            # Record after getting obj and constraint to assure they have
            # been gathered in MPI.
            self.recorders.record_iteration(system, metadata)

            # Get the double-sided constraint evaluations
            #for key, con in iteritems(self.get_2sided_constraints()):
            #    func_dict[name] = np.array(con.evaluate(self.parent))

        except Exception as msg:
            tb = traceback.format_exc()

            # Exceptions seem to be swallowed by the C code, so this
            # should give the user more info than the dreaded "segfault"
            print("Exception: %s" % str(msg))
            print(70*"=",tb,70*"=")
            fail = 1
            func_dict = {}

        #print("Functions calculated")
        #print(func_dict)
        return func_dict, fail

    def _gradfunc(self, dv_dict, func_dict):
        """ Function that evaluates and returns the gradient of the objective
        function and constraints. This function is passed to pyOpt's
        Optimization object and is called from its optimizers.

        Args
        ----
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

        fail = 0

        try:

            # Assemble inactive constraints
            inactives = {}
            if len(self.active_tols) > 0:
                for name, tols in iteritems(self.active_tols):
                    con = self.opt_prob.constraints[name]
                    inactive_idx = []
                    val = con.value
                    for j in range(len(val)):
                        if isinstance(tols, float):
                            tol = tols
                        else:
                            tol = tols[j]
                        lower, upper = con.lower[j], con.upper[j]
                        if lower is not None and val[j] > lower + tol:
                            inactive_idx.append(j)
                        if upper is not None and val[j] < upper - tol:
                            inactive_idx.append(j)

                    if inactive_idx:
                        inactives[name] = inactive_idx

            try:
                sens_dict = self.calc_gradient(dv_dict, self.quantities,
                                               return_format='dict',
                                               sparsity=self.sparsity,
                                               inactives=inactives)

            # Let the optimizer try to handle the error
            except AnalysisError:
                fail = 1

                # We need to cobble together a sens_dict of the correct size.
                # Best we can do is return zeros.

                sens_dict = OrderedDict()
                for okey, oval in iteritems(func_dict):
                    sens_dict[okey] = OrderedDict()
                    osize = len(oval)
                    for ikey, ival in iteritems(dv_dict):
                        isize = len(ival)
                        sens_dict[okey][ikey] = np.zeros((osize, isize))

            # Support for sub-index sparsity by returning the Jacobian in a
            # pyopt sparse format.
            for con, val1 in iteritems(self.sub_sparsity):
                for desvar, rel_idx in iteritems(val1):
                    coo = {}
                    jac = sens_dict[con][desvar]
                    nrow, ncol = jac.shape
                    coo['shape'] = [nrow, ncol]

                    row = []
                    col = []
                    data = []
                    ncol = len(rel_idx)
                    for i in range(nrow):
                        row.extend([i]*ncol)
                        col.extend(rel_idx)
                        data.extend(jac[i][rel_idx])

                    coo['coo'] = [np.array(row), np.array(col), np.array(data)]
                    sens_dict[con][desvar] = coo

        except Exception as msg:
            tb = traceback.format_exc()

            # Exceptions seem to be swallowed by the C code, so this
            # should give the user more info than the dreaded "segfault"
            print("Exception: %s" % str(msg))
            print(70*"=",tb,70*"=")
            sens_dict = {}

        #print("Derivatives calculated")
        #print(dv_dict)
        #print(sens_dict)
        return sens_dict, fail
