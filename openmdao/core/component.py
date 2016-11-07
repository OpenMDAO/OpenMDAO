"""Define the core Component classes.

Classes
-------
Component - base Component class
ImplicitComponent - used to define output variables that are all implicit
ExplicitComponent - used to define output variables that are all explicit
IndepVarComp - used to define output variables that are all independent
"""

from __future__ import division

import collections

import numpy
from six import string_types

from openmdao.core.system import System

class Component(System):
    """Base Component class; not to be directly instantiated."""

    DEFAULTS = {
        'indices': [0],
        'shape': (1,),
        'units': '',
        'value': 1.0,
        'scale': 1.0,
        'lower': None,
        'upper': None,
        'var_set': 0,
    }


    def _add_variable(self, name, typ, val, kwargs):
        """Add an input/output variable to the component.

        Args
        ----
        name : str
            name of the variable in this component's namespace.

        typ : str
            either 'input' or 'output'

        val : object
            The value of the variable being added.

        **kwargs : dict
            variable metadata with DEFAULTS defined above.
        """
        metadata = self.DEFAULTS.copy()
        metadata.update(kwargs)
        metadata['value'] = val
        if isinstance(val, numpy.ndarray):
            metadata['shape'] = val.shape
            if typ == 'input' and 'indices' not in kwargs:
                metadata['indices'] = numpy.arange(0, val.size, dtype=int)

        if typ == 'input':
            metadata['indices'] = numpy.array(metadata['indices'])

        self._variable_allprocs_names[typ].append(name)
        self._variable_myproc_names[typ].append(name)
        self._variable_myproc_metadata[typ].append(metadata)

    def add_input(self, name, val=1.0, **kwargs):
        """See _add_variable."""
        self._add_variable(name, 'input', val, kwargs)

    def add_output(self, name, val=1.0, **kwargs):
        """See _add_variable."""
        self._add_variable(name, 'output', val, kwargs)

    def _setup_vector(self, vectors, vector_var_ids):
        super(Component, self)._setup_vector(vectors, vector_var_ids)

        # Components need to load their initial input values into the _inputs vector
        if vectors['input']._name is None:
            names = self._variable_myproc_names['input']
            inputs = self._inputs
            #TODO: I don't think we really need to do this for connected inputs, but
            # for unconnected ones, if we don't do this then the values set in add_input
            # won't end up in the inputs vector when the component runs
            for i, meta in enumerate(self._variable_myproc_metadata['input']):
                inputs[names[i]] = meta['value']


class ImplicitComponent(Component):
    """Class to inherit from when all output variables are implicit."""

    def _apply_nonlinear(self):
        """Compute residuals; call user's apply_nonlinear."""
        self.apply_nonlinear(self._inputs, self._outputs, self._residuals)

    def _solve_nonlinear(self):
        """Compute outputs; call user's solve_nonlinear or nonlinear solver."""
        if self._solvers_nonlinear is not None:
            self._solvers_nonlinear(self._inputs, self._outputs)
        else:
            self.solve_nonlinear(self._inputs, self._outputs)

    def _apply_linear(self, vec_names, mode, var_ind_range=None):
        """Compute jac-vector product; call user's / Jacobian's apply_linear."""
        if var_ind_range is None:
            var_ind_range = self._variable_allprocs_range['output']

        for vec_name in vec_names:
            tmp = self._get_vectors(vec_name, var_ind_range, mode)
            d_inputs, d_outputs, d_residuals = tmp
            self.apply_linear(self._inputs, self._outputs,
                              d_inputs, d_outputs, d_residuals, mode)
            self._jacobian._apply(d_inputs, d_outputs, d_residuals, mode)

    def _solve_linear(self, vec_names, mode):
        """Apply inverse jac product; linear solver / user's solve_linear."""
        if self._solvers_linear is not None:
            return self._solvers_linear(vec_names, mode)
        else:
            success = True
            for vec_name in vec_names:
                d_outputs = self._vectors['output'][vec_name]
                d_residuals = self._vectors['residual'][vec_name]
                tmp = self.solve_linear(d_outputs, d_residuals, mode)
                success = success and tmp
            return success

    def _linearize(self):
        """Compute jacobian / factorization; call user's linearize."""
        self._jacobian._system = self
        self.linearize(self._inputs, self._outputs, self._jacobian)

        if self._jacobian._top_name == self.path_name:
            self._jacobian._update()

    def apply_nonlinear(self, inputs, outputs, residuals):
        """Compute residuals given inputs and outputs.

        Args
        ----
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        outputs : Vector
            unscaled, dimensional output variables read via outputs[key]
        residuals : Vector
            unscaled, dimensional residuals written to via residuals[key]
        """
        pass

    def solve_nonlinear(self, inputs, outputs):
        """Compute outputs given inputs.

        Args
        ----
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        outputs : Vector
            unscaled, dimensional output variables read via outputs[key]
        """
        pass

    def apply_linear(self, inputs, outputs,
                     d_inputs, d_outputs, d_residuals, mode):
        """Compute jac-vector product.

        If mode is:
            'fwd': (d_inputs, d_outputs) \|-> d_residuals

            'rev': d_residuals \|-> (d_inputs, d_outputs)

        Args
        ----
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        outputs : Vector
            unscaled, dimensional output variables read via outputs[key]
        d_inputs : Vector
            see inputs; product must be computed only if var_name in d_inputs
        d_outputs : Vector
            see outputs; product must be computed only if var_name in d_outputs
        d_residuals : Vector
            see outputs
        mode : str
            either 'fwd' or 'rev'
        """
        self._jacobian._apply(d_inputs, d_outputs, d_residuals, mode)

    def solve_linear(self, d_outputs, d_residuals, mode):
        """Apply inverse jac product.

        If mode is:
            'fwd': d_residuals \|-> d_outputs
            
            'rev': d_outputs \|-> d_residuals

        Args
        ----
        d_outputs : Vector
            unscaled, dimensional quantities read via d_outputs[key]
        d_residuals : Vector
            unscaled, dimensional quantities read via d_residuals[key]
        mode : str
            either 'fwd' or 'rev'
        """
        pass

    def linearize(self, inputs, outputs, jacobian):
        """Compute sub-jacobian parts / factorization.

        Args
        ----
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        outputs : Vector
            unscaled, dimensional output variables read via outputs[key]
        jacobian : Jacobian
            sub-jac components written to jacobian[output_name, input_name]
        """
        pass


class ExplicitComponent(Component):
    """Class to inherit from when all output variables are explicit."""

    def _apply_nonlinear(self):
        """Compute residuals; wrap the 'compute' method."""
        inputs = self._inputs
        outputs = self._outputs
        residuals = self._residuals

        residuals.set_vec(outputs)
        self.compute(inputs, outputs)
        residuals -= outputs
        outputs += residuals

    def _solve_nonlinear(self):
        """Compute outputs; wrap the 'compute' method."""
        inputs = self._inputs
        outputs = self._outputs
        residuals = self._residuals

        residuals.set_val(0.0)
        self.compute(inputs, outputs)

    def _apply_linear(self, vec_names, mode, var_ind_range=None):
        """Compute jac-vector product.

        Wrap user's 'compute_jacvec_product' or Jacobian's apply method.
        """
        if var_ind_range is None:
            var_ind_range = self._variable_allprocs_range['output']

        if self._jacobian._top_name == self.path_name:
            for vec_name in vec_names:
                tmp = self._get_vectors(vec_name, var_ind_range, mode)
                d_inputs, d_outputs, d_residuals = tmp
                self._jacobian._apply(d_inputs, d_outputs, d_residuals, mode)
        else:
            if mode == 'fwd':
                for vec_name in vec_names:
                    tmp = self._get_vectors(vec_name, var_ind_range, mode)
                    d_inputs, d_outputs, d_residuals = tmp

                    self.compute_jacvec_product(self._inputs, self._outputs,
                                                d_inputs, d_residuals, mode)
                    d_residuals *= -1.0
                    d_residuals += d_outputs
            elif mode == 'rev':
                for vec_name in vec_names:
                    tmp = self._get_vectors(vec_name, var_ind_range, mode)
                    d_inputs, d_outputs, d_residuals = tmp

                    d_residuals *= -1.0
                    self.compute_jacvec_product(self._inputs, self._outputs,
                                                d_inputs, d_residuals, mode)
                    d_residuals *= -1.0
                    d_outputs.set_vec(d_residuals)

    def _solve_linear(self, vec_names, mode):
        """Apply inverse jac product; apply the identity inverse jacobian."""
        for vec_name in vec_names:
            d_outputs = self._vectors['output'][vec_name]
            d_residuals = self._vectors['residual'][vec_name]
            if mode == 'fwd':
                d_outputs.setvec(d_residuals)
            elif mode == 'rev':
                d_residuals.setvec(d_outputs)

    def _linearize(self):
        """Compute jacobian / factorization; wrap 'compute_jacobian'."""
        self.compute_jacobian(self._inputs, self._outputs, self._jacobian)

        for op_name in self._variable_myproc_names['output']:
            size = len(self._outputs[op_name])
            ones = numpy.ones(size)
            arange = numpy.arange(size)
            self._jacobian[op_name, op_name] = (ones, arange, arange)

        for op_name in self._variable_myproc_names['output']:
            for ip_name in self._variable_myproc_names['input']:
                if (op_name, ip_name) in self._jacobian:
                    self._jacobian._negate(op_name, ip_name)

        if self._jacobian._top_name == self.path_name:
            self._jacobian._update()

    def compute(self, inputs, outputs):
        """Compute outputs given inputs.

        Args
        ----
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        outputs : Vector
            unscaled, dimensional output variables read via outputs[key]
        """
        pass

    def compute_jacobian(self, inputs, outputs, jacobian):
        """Compute sub-jacobian parts / factorization.

        Args
        ----
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        outputs : Vector
            unscaled, dimensional output variables read via outputs[key]
        jacobian : Jacobian
            sub-jac components written to jacobian[output_name, input_name]
        """

        pass

    def compute_jacvec_product(self, inputs, outputs,
                               d_inputs, d_outputs, mode):
        """Compute jac-vector product.

        If mode is:
            'fwd': d_inputs \|-> d_outputs

            'rev': d_outputs \|-> d_inputs

        Args
        ----
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        outputs : Vector
            unscaled, dimensional output variables read via outputs[key]
        d_inputs : Vector
            see inputs; product must be computed only if var_name in d_inputs
        d_outputs : Vector
            see outputs; product must be computed only if var_name in d_outputs
        mode : str
            either 'fwd' or 'rev'
        """
        pass


class IndepVarComp(ExplicitComponent):
    """Class to inherit from when all output variables are independent."""

    def __init__(self, name, val=1.0, **kwargs):
        super(IndepVarComp, self).__init__(**kwargs)
        self._indep = (name, val)

        for illegal in ('promotes', 'promotes_inputs', 'promotes_outputs'):
            if illegal in kwargs:
                raise ValueError("IndepVarComp init: '%s' is not supported "
                                 "in IndepVarComp." % illegal)

    def initialize_variables(self):
        """Define the independent variables as output variables."""

        name, val = self._indep
        kwargs = self.kwargs

        if isinstance(name, string_types):
            self.add_output(name, val, **kwargs)

        elif isinstance(name, collections.Iterable):
            for tup in name:
                badtup = None
                if isinstance(tup, tuple):
                    if len(tup) == 3:
                        n, v, kw = tup
                    elif len(tup) == 2:
                        n, v = tup
                        kw = {}
                    else:
                        badtup = tup
                else:
                    badtup = tup
                if badtup:
                    if isinstance(badtup, string_types):
                        badtup = name
                    raise ValueError("IndepVarComp init: arg %s must be a tuple of the form "
                                     "(name, value) or (name, value, keyword_dict)." %
                                     str(badtup))
                self.add_output(n, v, **kw)
        else:
            raise ValueError("first argument to IndepVarComp init must be either of type "
                             "`str` or an iterable of tuples of the form (name, value) or "
                             "(name, value, keyword_dict).")
