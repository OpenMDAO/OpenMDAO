"""Define the Problem class and a FakeComm class for non-MPI users."""

from __future__ import division
from collections import OrderedDict
import sys
import warnings

from six import string_types
from six.moves import range

import numpy as np

from openmdao.assemblers.default_assembler import DefaultAssembler
from openmdao.vectors.default_vector import DefaultVector
from openmdao.error_checking.check_config import check_config


class FakeComm(object):
    """Fake MPI communicator class used if mpi4py is not installed.

    Attributes
    ----------
    rank : int
        index of current proc; value is 0 because there is only 1 proc.
    size : int
        number of procs in the comm; value is 1 since MPI is not available.
    """

    def __init__(self):
        """Initialize attributes."""
        self.rank = 0
        self.size = 1


class Problem(object):
    """Top-level container for the systems and drivers.

    Attributes
    ----------
    model : <System>
        pointer to the top-level <System> object (root node in the tree).
    comm : MPI.Comm or <FakeComm>
        the global communicator; the same as that of assembler and model.
    _assembler : <Assembler>
        pointer to the global <Assembler> object.
    _use_ref_vector : bool
        if True, allocate vectors to store ref. values.
    """

    def __init__(self, model=None, comm=None, assembler_class=None,
                 use_ref_vector=True):
        """Initialize attributes.

        Args
        ----
        model : <System> or None
            pointer to the top-level <System> object (root node in the tree).
        comm : MPI.Comm or <FakeComm> or None
            the global communicator; the same as that of assembler and model.
        assembler_class : <Assembler> or None
            pointer to the global <Assembler> object.
        use_ref_vector : bool
            if True, allocate vectors to store ref. values.
        """
        if comm is None:
            try:
                from mpi4py import MPI
                comm = MPI.COMM_WORLD
            except ImportError:
                comm = FakeComm()
        if assembler_class is None:
            assembler_class = DefaultAssembler

        self.model = model
        self.comm = comm
        self._assembler = assembler_class(comm)
        self._use_ref_vector = use_ref_vector

    # TODO: getitem/setitem need to properly handle scaling/units
    def __getitem__(self, name):
        """Get an output/input variable.

        Args
        ----
        name : str
            name of the variable in the root system's namespace.

        Returns
        -------
        float or ndarray
            the requested output/input variable.
        """
        try:
            self.model._outputs[name]
            ind = self.model._var_myproc_names['output'].index(name)
            c0, c1 = self.model._scaling_to_phys['output'][ind, :]
            return c0 + c1 * self.model._outputs[name]
        except KeyError:
            ind = self.model._var_myproc_names['input'].index(name)
            c0, c1 = self.model._scaling_to_phys['input'][ind, :]
            return c0 + c1 * self.model._inputs[name]

    def __setitem__(self, name, value):
        """Set an output/input variable.

        Args
        ----
        name : str
            name of the output/input variable in the root system's namespace.
        value : float or ndarray or list
            value to set this variable to.
        """
        try:
            self.model._outputs[name]
            ind = self.model._var_myproc_names['output'].index(name)
            c0, c1 = self.model._scaling_to_norm['output'][ind, :]
            self.model._outputs[name] = c0 + c1 * np.array(value)
        except KeyError:
            ind = self.model._var_myproc_names['input'].index(name)
            c0, c1 = self.model._scaling_to_norm['input'][ind, :]
            self.model._inputs[name] = c0 + c1 * np.array(value)

    def run_model(self):
        """Run the model by calling the root system's solve_nonlinear.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        float
            relative error.
        float
            absolute error.
        """
        return self.model._solve_nonlinear()

    def run_once(self):
        """Backward compatible call for run_model.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        float
            relative error.
        float
            absolute error.
        """
        warnings.simplefilter('always', DeprecationWarning)
        warnings.warn('This method provides backwards compabitibility with '
                      'OpenMDAO <= 1.x ; use run_driver instead.',
                      DeprecationWarning, stacklevel=2)
        warnings.simplefilter('ignore', DeprecationWarning)

        return self.run_model()

    def run(self):
        """Backward compatible call for run_driver.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        float
            relative error.
        float
            absolute error.
        """
        warnings.simplefilter('always', DeprecationWarning)
        warnings.warn('This method provides backwards compabitibility with '
                      'OpenMDAO <= 1.x ; use run_driver instead.',
                      DeprecationWarning, stacklevel=2)
        warnings.simplefilter('ignore', DeprecationWarning)

        return self.run_driver()

    def setup(self, vector_class=DefaultVector, check=True, logger=None,
              mode='auto'):
        """Set up everything (model, assembler, vector, solvers, drivers).

        Args
        ----
        vector_class : type (DefaultVector)
            reference to an actual <Vector> class; not an instance.
        check : boolean (True)
            whether to run error check after setup is complete.
        logger : object
            Object for logging config checks if check is True.
        mode : string
            Derivatives calculation mode, 'fwd' for forward, and 'rev' for
            reverse (adjoint). Default is 'auto', which lets OpenMDAO choose
            the best mode for your problem.

        Returns
        -------
        self : <Problem>
            this enables the user to instantiate and setup in one line.
        """
        model = self.model
        comm = self.comm
        assembler = self._assembler

        if mode not in ['fwd', 'rev', 'auto']:
            msg = "Unsupported mode: '%s'" % mode
            raise ValueError(msg)

        # TODO: Support automatic determination of mode
        if mode == 'auto':
            mode = 'rev'
        self._mode = mode

        # Recursive system setup
        model._setup_processors('', comm, {}, assembler, [0, comm.size])
        model._setup_variables()
        model._setup_variable_indices({'input': 0, 'output': 0})
        model._setup_connections()

        # Assembler setup: variable metadata and indices
        nvars = {typ: len(model._var_allprocs_names[typ])
                 for typ in ['input', 'output']}
        assembler._setup_variables(nvars, model._var_myproc_metadata,
                                   model._var_myproc_indices)

        # Assembler setup: variable connections
        assembler._setup_connections(model._var_connections_indices,
                                     model._var_allprocs_names)

        # Assembler setup: global transfer indices vector
        assembler._setup_src_indices(model._var_myproc_metadata['input'],
                                     model._var_myproc_indices['input'])

        # Assembler setup: compute data required for units/scaling
        assembler._setup_src_data(model._var_myproc_metadata['output'],
                                  model._var_myproc_indices['output'])

        # Set up scaling vectors
        model._setup_scaling()

        # Set up lower and upper bounds vectors
        lower_bounds = vector_class('lower', 'output', self.model)
        upper_bounds = vector_class('upper', 'output', self.model)
        model._setup_bounds_vectors(lower_bounds, upper_bounds, True)

        # Vector setup for the basic execution vector
        self.setup_vector('nonlinear', vector_class, self._use_ref_vector)

        # Vector setup for the linear vector
        self.setup_vector('linear', vector_class, self._use_ref_vector)

        # check to see if a global jacobian was set prior to setup
        for system in model.system_iter(include_self=True, recurse=True):
            if system._pre_setup_jac is not None:
                system.set_jacobian(system._pre_setup_jac)
                system._pre_setup_jac = None

        if check:
            check_config(self, logger)

        return self

    def setup_vector(self, vec_name, vector_class, use_ref_vector):
        """Set up the 'vec_name' <Vector>.

        Args
        ----
        vec_name : str
            name of the vector.
        vector_class : type
            reference to the actual <Vector> class.
        use_ref_vector : bool
            if True, allocate vectors to store ref. values.
        """
        model = self.model
        assembler = self._assembler

        vectors = {}
        for key in ['input', 'output', 'residual']:
            if key is 'residual':
                typ = 'output'
            else:
                typ = key

            vectors[key] = vector_class(vec_name, typ, self.model)

        # TODO: implement this properly
        ind1, ind2 = self.model._var_allprocs_range['output']
        import numpy
        vector_var_ids = numpy.arange(ind1, ind2)

        self.model._setup_vector(vectors, vector_var_ids, use_ref_vector)

    def compute_total_derivs(self, of=None, wrt=None, return_format='flat_dict'):
        """Compute derivatives of desired quantities with respect to desired inputs.

        Args
        ----
        of : list of variable name strings or None
            Variables whose derivatives will be computed. Default is None, which
            uses the driver's objectives and constraints.
        wrt : list of variable name strings or None
            Variables with respect to which the derivatives will be computed.
            Default is None, which uses the driver's desvars.
        return_format : string
            Format to return the derivatives. Default is a 'flat_dict', which
            returns them in a dictionary whose keys are tuples of form (of, wrt).

        Returns
        -------
        derivs : object
            Derivatives in form requested by 'return_format'.
        """
        model = self.model
        mode = self._mode
        vec_dinput = model._vectors['input']
        vec_doutput = model._vectors['output']
        vec_dresid = model._vectors['residual']

        # TODO - Pull 'of' and 'wrt' from driver if unspecified.
        if wrt is None:
            raise NotImplementedError("Need to specify 'wrt' for now.")
        if of is None:
            raise NotImplementedError("Need to specify 'of' for now.")

        # A number of features will need to be supported here as development
        # goes forward.
        # -------------------------------------------------------------------
        # TODO: Make sure we can function in parallel when some params or
        # functions are not local.
        # TODO: Support parallel adjoint and parallel forward derivatives
        #       Aside: how are they specified, and do we have to pick up any
        #       that are missed?
        # TODO: Handle driver scaling.
        # TODO: Might be some additional adjustments needed to set the 'one'
        #       into the PETSC vector.
        # TODO: support parmeter/constraint indices
        # TODO: Support for any other return_format we need.
        # TODO: Support constraint sparsity (i.e., skip in/out that are not
        #       relevant for this constraint) (desvars too?)
        # TODO: Don't calculate for inactive constraints
        # TODO: Support full-model FD. Don't know how this'll work, but we
        #       used to need a separate function for that.
        # TODO: poi_indices and qoi_indices requires special support
        # -------------------------------------------------------------------

        # Prepare model for calculation by cleaning out the derivatives
        # vectors.
        for subname in vec_dinput:

            # TODO: Do all three deriv vectors have the same keys?

            # Skip nonlinear because we don't need to mess with it?
            if subname == 'nonlinear':
                continue

            vec_dinput[subname].set_const(0.0)
            vec_doutput[subname].set_const(0.0)
            vec_dresid[subname].set_const(0.0)

        # Linearize Model
        model._linearize()

        # Create data structures (and possibly allocate space) for the total
        # derivatives that we will return.
        if return_format == 'flat_dict':

            totals = OrderedDict()

            for okeys in of:
                if isinstance(okeys, string_types):
                    okeys = (okeys,)
                for okey in okeys:
                    for ikeys in wrt:
                        if isinstance(ikeys, string_types):
                            ikeys = (ikeys,)
                        for ikey in ikeys:
                            totals[(okey, ikey)] = None

        else:
            msg = "Unsupported return format '%s." % return_format
            raise NotImplementedError(msg)

        if mode == 'fwd':
            input_list, output_list = wrt, of
            input_vec, output_vec = vec_dresid, vec_doutput
        else:
            input_list, output_list = of, wrt
            input_vec, output_vec = vec_doutput, vec_dresid

        # TODO : Parallel adjoint setup loop goes here.
        # NOTE : Until we support it, we will just limit ourselves to the
        # 'linear' vector.
        vecname = 'linear'
        dinputs = input_vec[vecname]
        doutputs = output_vec[vecname]

        # If Forward mode, solve linear system for each 'wrt'
        # If Adjoint mode, solve linear system for each 'of'
        for input_name in input_list:
            flat_view = dinputs._views_flat[input_name]
            n_in = len(flat_view)
            for idx in range(n_in):
                # Maybe we don't need to clean up so much at the beginning,
                # since we clean this every time.
                dinputs.set_const(0.0)

                # Dictionary access returns a scaler for 1d input, and we
                # need a vector for clean code, so use _views_flat.
                flat_view[idx] = 1.0

                # The root system solves here.
                model._solve_linear([vecname], mode)

                # Pull out the answers and pack them into our data structure.
                for output_name in output_list:

                    deriv_val = doutputs._views_flat[output_name]
                    len_val = len(deriv_val)

                    if return_format == 'flat_dict':
                        if mode == 'fwd':

                            key = (output_name, input_name)

                            if totals[key] is None:
                                totals[key] = np.zeros((len_val, n_in))
                            totals[key][:, idx] = deriv_val

                        else:

                            key = (input_name, output_name)

                            if totals[key] is None:
                                totals[key] = np.zeros((n_in, len_val))
                            totals[key][idx, :] = deriv_val

        return totals
