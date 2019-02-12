"""
OpenMDAO Wrapper for pyoptsparse.

pyoptsparse is based on pyOpt, which is an object-oriented framework for
formulating and solving nonlinear constrained optimization problems, with
additional MPI capability.
"""
from __future__ import print_function

from collections import OrderedDict
import json
import sys
import traceback

from six import iteritems, itervalues, string_types, reraise

import numpy as np
from scipy.sparse import coo_matrix

from pyoptsparse import Optimization

from openmdao.core.analysis_error import AnalysisError
from openmdao.core.driver import Driver, RecordingDebugging
import openmdao.utils.coloring as coloring_mod


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
    _indep_list : list
        List of design variables.
    _quantities : list
        Contains the objectives plus nonlinear constraints.
    """

    def __init__(self, **kwargs):
        """
        Initialize pyopt.

        Parameters
        ----------
        **kwargs : dict of keyword arguments
            Keyword arguments that will be mapped into the Driver options.
        """
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
        self.options.declare('dynamic_derivs_sparsity', default=False, types=bool,
                             desc='Compute derivative sparsity dynamically if True')
        self.options.declare('dynamic_simul_derivs', default=False, types=bool,
                             desc='Compute simultaneous derivative coloring dynamically if True')
        self.options.declare('dynamic_derivs_repeats', default=3, types=int,
                             desc='Number of compute_totals calls during dynamic computation of '
                                  'simultaneous derivative coloring or derivatives sparsity')

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

        self._setup_tot_jac_sparsity()

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
        self._total_jac = None
        self.iter_count = 0
        fwd = problem._mode == 'fwd'
        optimizer = self.options['optimizer']

        # Only need initial run if we have linear constraints or if we are using an optimizer that
        # doesn't perform one initially.
        con_meta = self._cons
        if optimizer in run_required or np.any([con['linear'] for con in itervalues(self._cons)]):
            with RecordingDebugging(optimizer, self.iter_count, self) as rec:
                # Initial Run
                model._solve_nonlinear()
                rec.abs = 0.0
                rec.rel = 0.0
            self.iter_count += 1

        # compute dynamic simul deriv coloring or just sparsity if option is set
        if coloring_mod._use_sparsity:
            if self.options['dynamic_simul_derivs']:
                coloring_mod.dynamic_simul_coloring(self, run_model=optimizer not in run_required,
                                                    do_sparsity=True)
            elif self.options['dynamic_derivs_sparsity']:
                coloring_mod.dynamic_sparsity(self)

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
        lcons = [key for (key, con) in iteritems(con_meta) if con['linear']]
        if len(lcons) > 0:
            _lin_jacs = self._compute_totals(of=lcons, wrt=indep_list, return_format='dict')
            # convert all of our linear constraint jacs to COO format. Otherwise pyoptsparse will
            # do it for us and we'll end up with a fully dense COO matrix and very slow evaluation
            # of linear constraints!
            to_remove = []
            for jacdct in itervalues(_lin_jacs):
                for n, subjac in iteritems(jacdct):
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
        for name, meta in iteritems(con_meta):
            if meta['equals'] is None:
                continue
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
                    resjac = self._res_jacs[name]
                    jac = {n: resjac[n] for n in wrt}
                else:
                    jac = None
                opt_prob.addConGroup(name, size, lower=lower, upper=upper, wrt=wrt, jac=jac)
                self._quantities.append(name)

        # Add all inequality constraints
        for name, meta in iteritems(con_meta):
            if meta['equals'] is not None:
                continue
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
            reraise(ImportError, ImportError(msg), sys.exc_info()[2])

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
            self.set_design_var(name, dv_dict[name])

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
            # Let the optimizer try to handle the error
            except AnalysisError:
                prob.model._clear_iprint()
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
                        else:
                            newdv[ikey] = sens_dict[okey][ikey]
                sens_dict = new_sens

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
        for n, meta in iteritems(self._cons):
            if 'linear' not in meta or not meta['linear']:
                if meta['equals'] is not None:
                    nl_order.append(n)
                else:
                    neq_order.append(n)

        nl_order.extend(neq_order)

        return nl_order

    def _setup_tot_jac_sparsity(self):
        """
        Set up total jacobian subjac sparsity.
        """
        if self._total_jac_sparsity is None:
            return

        if isinstance(self._total_jac_sparsity, string_types):
            with open(self._total_jac_sparsity, 'r') as f:
                self._total_jac_sparsity = json.load(f)

        self._res_jacs = {}
        for res, resdict in iteritems(self._total_jac_sparsity):
            if res in self._objs:  # skip objectives
                continue
            self._res_jacs[res] = {}
            for dv, (rows, cols, shape) in iteritems(resdict):
                rows = np.array(rows, dtype=int)
                cols = np.array(cols, dtype=int)

                self._res_jacs[res][dv] = {
                    'coo': [rows, cols, np.zeros(rows.size)],
                    'shape': shape,
                }
