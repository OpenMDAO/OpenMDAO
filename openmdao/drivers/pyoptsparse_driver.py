"""
OpenMDAO Wrapper for pyoptsparse.
pyoptsparse is based on pyOpt, which is an object-oriented framework for
formulating and solving nonlinear constrained optimization problems, with
additional MPI capability.
"""

from __future__ import print_function

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

        # TODO: Support these
        #self.supports['active_set'] = True
        #self.supports['linear_constraints'] = True
        #self.supports['integer_design_vars'] = False

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

        self.pyopt_solution = None

        #self.lin_jacs = OrderedDict()
        #self.quantities = []
        #self.metadata = None
        #self.exit_flag = 0
        #self.sparsity = OrderedDict()
        #self.sub_sparsity = OrderedDict()
        #self.active_tols = {}

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
        (i.e., SNOPT) control the iteration.
        """
        model = self.problem.model
        self.pyopt_solution = None
        self.iter_count = 0

        # Initial Run
        model._solve_nonlinear()

        opt_prob = Optimization(self.options['title'], self._objfunc)

        # Add all parameters
        param_meta = self._designvars
        param_vals = self.get_design_var_values()

        for name, meta in iteritems(param_meta):
            opt_prob.addVarGroup(name, meta['size'], type='c',
                                 value=param_vals[name],
                                 lower=meta['lower'], upper=meta['upper'])

        opt_prob.finalizeDesignVariables()

        # Add all objectives
        objs = self.get_objectives()
        self.quantities = list(objs)
        self.sparsity = OrderedDict()
        self.sub_sparsity = OrderedDict()
        for name in objs:
            opt_prob.addObj(name)
            self.sparsity[name] = self.indep_list

        # Calculate and save gradient for any linear constraints.
        lcons = self.get_constraints(lintype='linear').keys()
        self._problem = problem
        if len(lcons) > 0:
            self.lin_jacs = self.calc_gradient(indep_list, lcons,
                                               return_format='dict')
            #print("Linear Gradient")
            #print(self.lin_jacs)

        # Add all equality constraints
        econs = self.get_constraints(ctype='eq', lintype='nonlinear')
        con_meta = self.get_constraint_metadata()
        self.quantities += list(econs)

        self.active_tols = {}
        for name in self.get_constraints(ctype='eq'):
            meta = con_meta[name]
            size = meta['size']
            lower = upper = meta['equals']

            # Sparsify Jacobian via relevance
            rels = rel.relevant[name]
            wrt = rels.intersection(indep_list)
            self.sparsity[name] = wrt

            if meta['linear']:
                opt_prob.addConGroup(name, size, lower=lower, upper=upper,
                                     linear=True, wrt=wrt,
                                     jac=self.lin_jacs[name])
            else:

                jac = self._build_sparse(name, wrt, size, param_vals,
                                         sub_param_conns, full_param_conns, rels)
                opt_prob.addConGroup(name, size, lower=lower, upper=upper,
                                     wrt=wrt, jac=jac)

            active_tol = meta.get('active_tol')
            if active_tol:
                self.active_tols[name] = active_tol

        # Add all inequality constraints
        incons = self.get_constraints(ctype='ineq', lintype='nonlinear')
        self.quantities += list(incons)

        for name in self.get_constraints(ctype='ineq'):
            meta = con_meta[name]
            size = meta['size']

            # Bounds - double sided is supported
            lower = meta['lower']
            upper = meta['upper']

            # Sparsify Jacobian via relevance
            rels = rel.relevant[name]
            wrt = rels.intersection(indep_list)
            self.sparsity[name] = wrt

            if meta['linear']:
                opt_prob.addConGroup(name, size, upper=upper, lower=lower,
                                     linear=True, wrt=wrt,
                                     jac=self.lin_jacs[name])
            else:

                jac = self._build_sparse(name, wrt, size, param_vals,
                                         sub_param_conns, full_param_conns, rels)
                opt_prob.addConGroup(name, size, upper=upper, lower=lower,
                                     wrt=wrt, jac=jac)

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

        self._problem = None

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

    def _build_sparse(self, name, wrt, consize, param_vals, sub_param_conns,
                      full_param_conns, rels):
        """ Build up the data structures that define a sparse Jacobian
        matrix. Called separately on each nonlinear constraint.

        Args
        ----
        name : str
            Constraint name.
        wrt : list
            List of relevant param names.
        consize : int
            Width of this constraint.
        param_vals : dict
            Dictionary of parameter values; used for sizing.
        sub_param_conns : dict
            Parameter subindex connection info.
        full_param_conns : dict
            Parameter full connection info.
        rels : set
            Set of relevant nodes for this connstraint.

        Returns
        -------
        pyoptsparse coo matrix or None
        """

        jac = None

        # Additional sparsity for index connections
        for param in wrt:

            sub_conns = sub_param_conns.get(param)
            if not sub_conns:
                continue

            # If we have a simultaneous full connection, then we move on
            full_conns = full_param_conns.get(param)
            if full_conns.intersection(rels):
                continue

            rel_idx = set()
            for target, idx in iteritems(sub_conns):

                # If a target of the indexed desvar connection is
                # in the relevant path for this constraint, then
                # those indices are relevant.
                if target in rels:
                    rel_idx.update(idx)

            nrel = len(rel_idx)
            if nrel > 0:

                if jac is None:
                    jac = {}

                if param not in jac:
                    # A coo matrix for the Jacobian
                    # mat = {'coo':[row, col, data],
                    #        'shape':[nrow, ncols]}
                    coo = {}
                    coo['shape'] = [consize, len(param_vals[param])]
                    jac[param] = coo

                row = []
                col = []
                for i in range(consize):
                    row.extend([i]*nrel)
                    col.extend(rel_idx)
                data = np.ones((len(row), ))

                jac[param]['coo'] = [np.array(row), np.array(col), data]

                if name not in self.sub_sparsity:
                    self.sub_sparsity[name] = {}
                self.sub_sparsity[name][param] = np.array(list(rel_idx))

        return jac

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
        metadata = self.metadata
        system = self.root

        try:
            for name in self.indep_list:
                self.set_desvar(name, dv_dict[name])

            # Execute the model
            #print("Setting DV")
            #print(dv_dict)

            self.iter_count += 1
            update_local_meta(metadata, (self.iter_count,))

            try:
                with self.root._dircontext:
                    system.solve_nonlinear(metadata=metadata)

            # Let the optimizer try to handle the error
            except AnalysisError:
                fail = 1

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
