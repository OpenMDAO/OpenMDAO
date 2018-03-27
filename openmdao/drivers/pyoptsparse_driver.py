"""
OpenMDAO Wrapper for pyoptsparse.

pyoptsparse is based on pyOpt, which is an object-oriented framework for
formulating and solving nonlinear constrained optimization problems, with
additional MPI capability.
"""

from __future__ import print_function
from collections import OrderedDict
import traceback

from six import iteritems, itervalues

import numpy as np
from scipy.sparse import coo_matrix

from pyoptsparse import Optimization

from openmdao.core.analysis_error import AnalysisError
from openmdao.core.driver import Driver, RecordingDebugging
from openmdao.utils.record_util import create_local_meta


# names of optimizers that use gradients
grad_drivers = {'CONMIN', 'FSQP', 'IPOPT', 'NLPQLP',
                'PSQP', 'SLSQP', 'SNOPT', 'NLPY_AUGLAG'}

# names of optimizers that allow multiple objectives
multi_obj_drivers = {'NSGA2'}


def _check_imports():
    """
    Dynamically remove optimizers we don't have.

    Returns
    -------
    list of str
        List of valid optimizer strings.
    """
    optlist = ['ALPSO', 'CONMIN', 'FSQP', 'IPOPT', 'NLPQLP',
               'NSGA2', 'PSQP', 'SLSQP', 'SNOPT', 'NLPY_AUGLAG', 'NOMAD']

    for optimizer in optlist[:]:
        try:
            __import__('pyoptsparse', globals(), locals(), [optimizer], 0)
        except ImportError:
            optlist.remove(optimizer)

    return optlist


CITATIONS = """
@phdthesis{hwang_thesis_2015,
  author       = {John T. Hwang},
  title        = {A Modular Approach to Large-Scale Design Optimization of Aerospace Systems},
  school       = {University of Michigan},
  year         = 2015
}
"""


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
        Name of optimizers to use.
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
        self.supports['simultaneous_derivatives'] = True

        # What we don't support yet
        self.supports['active_set'] = False
        self.supports['integer_design_vars'] = False

        # User Options
        self.options.declare('optimizer', default='SLSQP', values=_check_imports(),
                             desc='Name of optimizers to use')
        self.options.declare('title', default='Optimization using pyOpt_sparse',
                             desc='Title of this optimization run')
        self.options.declare('print_results', types=bool, default=True,
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

        self.cite = CITATIONS

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
        model = problem.model
        relevant = model._relevant
        self.pyopt_solution = None
        self.iter_count = 0
        fwd = problem._mode == 'fwd'
        optimizer = self.options['optimizer']

        # Metadata Setup
        self.metadata = create_local_meta(optimizer)

        # Only need initial run if we have linear constraints.
        con_meta = self._cons
        if np.any([con['linear'] for con in itervalues(self._cons)]):
            with RecordingDebugging(optimizer, self.iter_count, self) as rec:
                # Initial Run
                model._solve_nonlinear()
                rec.abs = 0.0
                rec.rel = 0.0
            self.iter_count += 1

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
        lcons = [key for (key, con) in iteritems(con_meta) if con['linear'] is True]
        if len(lcons) > 0:
            _lin_jacs = self._compute_totals(of=lcons, wrt=indep_list, return_format='dict')
            # convert all of our linear constraint jacs to COO format. Otherwise pyoptsparse will
            # do it for us and we'll end up with a fully dense COO matrix and very slow evaluation
            # of linear constraints!
            to_remove = []
            for oname, jacdct in iteritems(_lin_jacs):
                for n, subjac in iteritems(jacdct):
                    if isinstance(subjac, np.ndarray):
                        # we can safely use coo_matrix to automatically convert the ndarray
                        # since our linear constraint jacs are constant, so zeros won't become
                        # nonzero during the optimization.
                        mat = coo_matrix(subjac)
                        if mat.row.size > 0:
                            jacdct[n] = {'coo': [mat.row, mat.col, mat.data], 'shape': mat.shape}

        # Add all equality constraints
        self.active_tols = {}
        eqcons = OrderedDict((key, con) for (key, con) in iteritems(con_meta)
                             if con['equals'] is not None)
        for name, meta in iteritems(eqcons):
            size = meta['size']
            lower = upper = meta['equals']
            if fwd:
                wrt = [v for v in indep_list if name in relevant[v]]
            else:
                rels = relevant[name]
                wrt = [v for v in indep_list if v in rels]

            if meta['linear']:
                jac = {w: _lin_jacs[name][w] for w in wrt}
                opt_prob.addConGroup(name, size, lower=lower, upper=upper,
                                     linear=True, wrt=wrt, jac=jac)
            else:
                if name in self._res_jacs:
                    jac = self._res_jacs[name]
                else:
                    jac = None
                opt_prob.addConGroup(name, size, lower=lower, upper=upper, wrt=wrt, jac=jac)
                self._quantities.append(name)

        # Add all inequality constraints
        iqcons = OrderedDict((key, con) for (key, con) in iteritems(con_meta)
                             if con['equals'] is None)
        for name, meta in iteritems(iqcons):
            size = meta['size']

            # Bounds - double sided is supported
            lower = meta['lower']
            upper = meta['upper']

            if fwd:
                wrt = [v for v in indep_list if name in relevant[v]]
            else:
                rels = relevant[name]
                wrt = [v for v in indep_list if v in rels]

            if meta['linear']:
                jac = {w: _lin_jacs[name][w] for w in wrt}
                opt_prob.addConGroup(name, size, upper=upper, lower=lower,
                                     linear=True, wrt=wrt, jac=jac)
            else:
                if name in self._res_jacs:
                    jac = self._res_jacs[name]
                else:
                    jac = None
                opt_prob.addConGroup(name, size, upper=upper, lower=lower, wrt=wrt, jac=jac)
                self._quantities.append(name)

        # Instantiate the requested optimizer
        try:
            _tmp = __import__('pyoptsparse', globals(), locals(), [optimizer], 0)
            opt = getattr(_tmp, optimizer)()

        except ImportError:
            msg = "Optimizer %s is not available in this installation." % optimizer
            raise ImportError(msg)

        # Set optimization options
        for option, value in self.opt_settings.items():
            opt.setOption(option, value)

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

        with RecordingDebugging(self.options['optimizer'], self.iter_count, self) as rec:
            model._solve_nonlinear()
            rec.abs = 0.0
            rec.rel = 0.0
        self.iter_count += 1

        # Save the most recent solution.
        self.pyopt_solution = sol
        try:
            exit_status = sol.optInform['value']
            self.fail = False

            # These are various failed statuses.
            if exit_status > 2:
                self.fail = True

        except KeyError:
            # optimizers other than pySNOPT may not populate this dict
            pass

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
            with RecordingDebugging(self.options['optimizer'], self.iter_count, self) as rec:
                self.iter_count += 1
                try:
                    model._solve_nonlinear()

                # Let the optimizer try to handle the error
                except AnalysisError:
                    model._clear_iprint()
                    fail = 1

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

            try:
                sens_dict = self._compute_totals(of=self._quantities,
                                                 wrt=self._indep_list,
                                                 return_format='dict')
                # i = 0
                # print("col %d:" % i, prob.total_info.J[:,i])
            # Let the optimizer try to handle the error
            except AnalysisError:
                self._problem.model._clear_iprint()
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
            # else:
            #     if we don't convert to 'coo' here, pyoptsparse will do a
            #     conversion of our dense array into a fully dense 'coo', which is bad.
            #     TODO: look into getting rid of all of these conversions!
            #     new_sens = OrderedDict()
            #     res_jacs = self._res_jacs
            #     for okey in func_dict:
            #         new_sens[okey] = newdv = OrderedDict()
            #         for ikey in dv_dict:
            #             if okey in res_jacs and ikey in res_jacs[okey]:
            #                 arr = sens_dict[okey][ikey]
            #                 coo = res_jacs[okey][ikey]
            #                 row, col, data = coo['coo']
            #                 coo['coo'][2] = arr[row, col].flatten()
            #                 newdv[ikey] = coo
            #             else:
            #                 newdv[ikey] = sens_dict[okey][ikey]
            #     sens_dict = new_sens
            #     for name, dvdct in iteritems(self._res_jacs):
            #         for dv, coo in iteritems(dvdct):
            #             arr = sens_dict[name][dv]
            #             row, col, data = coo['coo']
            #             coo['coo'][2] = arr[row, col].flatten()
            #             sens_dict[name][dv] = coo

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

    def _get_name(self):
        """
        Get name of current optimizer.

        Returns
        -------
        str
            The name of the current optimizer.
        """
        return self.options['optimizer']

    def get_ordered_nl_responses(self):
        """
        Return the names of nonlinear responses in the order used by the driver.

        Default order is objectives followed by nonlinear constraints.  This is used for
        simultaneous derivative coloring and sparsity determination.

        Returns
        -------
        list of str
            The nonlinear response names in order.
        """
        order = list(self._objs)
        order.extend(n for n, meta in iteritems(self._cons) if meta['equals'] is not None
                     and not ('linear' in meta and meta['linear']))
        order.extend(n for n, meta in iteritems(self._cons) if meta['equals'] is None
                     and not ('linear' in meta and meta['linear']))
        return order

    def _setup_simul_coloring(self, mode='fwd'):
        """
        Set up metadata for simultaneous derivative solution.

        Parameters
        ----------
        mode : str
            Derivative direction, either 'fwd' or 'rev'.
        """
        super(pyOptSparseDriver, self)._setup_simul_coloring(mode)

        if self._simul_coloring_info is None:
            return

        column_lists, row_map, sparsity = self._simul_coloring_info

        if sparsity is None:
            raise RuntimeError("To use simul coloring with pyOptSparse you must include "
                               "sparsity structure. (run openmdao simul_coloring with -s option)")

        self._res_jacs = {}
        # for res, resdict in iteritems(sparsity):
        #     if res in self._objs:  # skip objectives
        #         continue
        #     self._res_jacs[res] = {}
        #     for dv, (rows, cols, shape) in iteritems(resdict):
        #         rows = np.array(rows, dtype=int)
        #         cols = np.array(cols, dtype=int)
        #
        #         # print("sparsity for %s, %s: %d of %s" % (res, dv, rows.size,
        #         #                                         (shape[0] * shape[1])))
        #
        #         self._res_jacs[res][dv] = {
        #             'coo': [rows, cols, np.zeros(rows.size)],
        #             'shape': [shape[0], shape[1]],
        #         }
