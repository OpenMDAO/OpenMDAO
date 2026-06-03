from collections import OrderedDict
from collections.abc import Mapping

import numpy as np

from openmdao.core.total_jac import _TotalJacInfo


class _FunctionalCallback:

    def __init__(self, prob, form, input_vars, output_vars,
                 ):
        """
        Initialize a _FunctionalCallback.

        Parameters
        ----------
        prob : Problem
            The Problem instance to wrap.
        form : str
            The form of the callback. Must be one of ``'f'``, ``'dfdx'``, or ``'fdfdx'``.
        input_vars : list of str or list of dict or None
            Variables to treat as inputs. See :meth:`Problem.get_callback` for the accepted
            format. Required when ``form='f'``; may be ``None`` for derivative forms to
            fall back to the driver's design variables.
        output_vars : list of str or list of dict or None
            Variables to treat as outputs. See :meth:`Problem.get_callback` for the accepted
            format. Required when ``form='f'``; may be ``None`` for derivative forms to
            fall back to the driver's responses.
        """
        self._problem = prob

        valid_forms = ("f", "dfdx", "fdfdx")
        if form not in valid_forms:
            msg = f"{self.msginfo}: Unsupported form='{form}'. Use one of {valid_forms}."
            raise ValueError(msg)
        self._form = form

        (self._total_jac_info, self._input_len, self._input_metadata, self._output_len,
         self._output_metadata) = self._process_vars(input_vars, output_vars,
                                                     # driver_scaling,
                                                     )

    def _process_vars(self, input_vars, output_vars,
                      # driver_scaling,
                      ):
        # First, decided on what names we'll be working with.
        if self.form == "f":
            # For the "f" form, we won't be working with derivatives and can't rely on
            # `_TotalJacInfo` to decided on inputs and outputs based on design variables and
            # responses.
            if (not input_vars) or (not output_vars):
                raise ValueError((f"{self.msginfo}: input_vars and output_vars arguments must "
                                  f"be provided when using form='{self.form}'"))
            # `"f"` form is not doing derivatives, so no need for the `_TotalJacInfo`.
            tji = None
        else:
            if input_vars:
                # Extract just the names, indices, and units to pass to `_TotalJacInfo` constructor.
                input_var_names = []
                input_var_indices = []
                input_var_units = {}
                for entry in input_vars:
                    if isinstance(entry, str):
                        input_var_names.append(entry)
                        input_var_indices.append(None)
                    elif isinstance(entry, Mapping):
                        for k, meta in entry.items():
                            input_var_names.append(k)
                            input_var_indices.append(meta.get("indices", None))
                            u = meta.get("units", None)
                            if u is not None:
                                input_var_units[k] = u
                    else:
                        raise ValueError((f"{self.msginfo}: invalid entry {entry} in "
                                           "input_vars argument"))
            else:
                input_var_names = None
                input_var_indices = None
                input_var_units = {}

            if output_vars:
                # Extract just the names, indices, and units to pass to `_TotalJacInfo` constructor.
                output_var_names = []
                output_var_indices = []
                output_var_units = {}
                for entry in output_vars:
                    if isinstance(entry, str):
                        output_var_names.append(entry)
                        output_var_indices.append(None)
                    elif isinstance(entry, Mapping):
                        for k, meta in entry.items():
                            output_var_names.append(k)
                            output_var_indices.append(meta.get("indices", None))
                            u = meta.get("units", None)
                            if u is not None:
                                output_var_units[k] = u
                    else:
                        raise ValueError((f"{self.msginfo}: invalid entry {entry} in "
                                           "output_vars argument"))
            else:
                output_var_names = None
                output_var_indices = None
                output_var_units = {}

            tji = _TotalJacInfo(self.problem, output_var_names, input_var_names,
                                of_indices=output_var_indices, wrt_indices=input_var_indices,
                                return_format='flat_dict',
                                driver_scaling=False,
                                _functional=True,
                                always_include_linear=True,
                                of_units=output_var_units or None,
                                wrt_units=input_var_units or None,
                                )

            if not input_vars:
                # User didn't provide any input var data, so create one from what `_TotalJacInfo`
                # decided.
                input_vars = [
                    {vname: {
                        "indices": vmeta["indices"],
                        "name": vmeta["name"],
                    } for vname, vmeta in tji.input_meta["fwd"].items()}
                ]
            if not output_vars:
                # User didn't provide any output var data, so create one from what `_TotalJacInfo`
                # decided.
                output_vars = [
                    {valias: {
                        "indices": vmeta["indices"],
                        "name": vmeta["name"],
                    } for valias, vmeta in tji.output_meta["fwd"].items()}
                ]

        input_metadata, input_len = self._process_var_arguments(input_vars)
        output_metadata, output_len = self._process_var_arguments(output_vars)

        return tji, input_len, input_metadata, output_len, output_metadata

    def _process_var_arguments(self, io_vars):
        offset = 0
        # `dict`s have been ordered since Python 3.7, but I think it's nice to be explicit.
        vmetas = OrderedDict()
        # I'm imagining that `var_args` will look like this:
        # [{"x": {"units": "m", "indices": [1, 2]}}, {"y": {"units": "s"}}, "z"]
        # ["x", {"t": {"units": "s"}}]
        for entry in io_vars:
            if isinstance(entry, Mapping):
                # Object is dict-like, so iterate over the mapping from variable names to variable
                # metadata.
                for valias, vmeta in entry.items():
                    vmetas[valias], offset = self._get_var_metadata_with_defaults(valias, vmeta,
                                                                                  offset)
            else:
                # Assume object is a string of a variable name, with no metadata.
                valias = entry
                vmeta = {}
                vmetas[valias], offset = self._get_var_metadata_with_defaults(valias, vmeta, offset)

        return vmetas, offset

    def _get_var_metadata_with_defaults(self, valias, vmeta, offset):
        problem = self.problem
        name = vmeta.get("name", valias)
        units = vmeta.get("units", None)

        indices = vmeta.get("indices", None)
        val = problem.get_val(name, units=units, indices=indices)
        try:
            # Assume val is an ndarray.
            shape = val.shape
            val_size = val.size
        except AttributeError:
            # Must be a scalar, so set the shape and size appropriately.
            shape = ()
            val_size = 1

        offset_new = offset + val_size
        offsets = (offset, offset_new)
        return {"name": name, "units": units, "indices": indices,
                "shape": shape, "offsets": offsets}, offset_new

    def _vector_to_problem(self, vec, metadata):
        problem = self._problem

        for valias, vmetadata in metadata.items():
            name = vmetadata["name"]
            units = vmetadata["units"]
            indices = vmetadata["indices"]
            shape = vmetadata["shape"]
            idx0, idx1 = vmetadata["offsets"]
            val = vec[idx0:idx1].reshape(shape)
            problem.set_val(name, val, indices=indices, units=units)

    def _problem_to_vector(self, vec, metadata):
        problem = self._problem

        for valias, vmetadata in metadata.items():
            name = vmetadata["name"]
            units = vmetadata["units"]
            indices = vmetadata["indices"]
            idx0, idx1 = vmetadata["offsets"]
            val = problem.get_val(name, indices=indices, units=units)
            vec[idx0:idx1] = val.flat

    def _totals_to_jacobian(self, J, totals):
        input_metadata = self._input_metadata
        output_metadata = self._output_metadata

        # Not sure about sparse Jacobians.
        # `_TotalJacInfo.compute_totals` appears to use constraint/objective
        # aliases, so no need to go from alias to name here.
        for (vout, vin), val in totals.items():
            offset_in0, offset_in1 = input_metadata[vin]["offsets"]
            offset_out0, offset_out1 = output_metadata[vout]["offsets"]

            J[offset_out0:offset_out1, offset_in0:offset_in1] = val

    def __call__(self, x, y=None, J=None):
        """
        Evaluate the model and return outputs and/or total derivatives.

        Sets the input variables from ``x``, calls ``Problem.run_model()``, and
        then reads back the requested outputs and/or computes total derivatives,
        depending on the ``form`` this callback was created with.

        Parameters
        ----------
        x : numpy.ndarray
            Flat 1-D input vector.  Must have length equal to the total size of
            all input variables (use :meth:`create_input_vector` to create a
            correctly-sized array pre-populated with the current problem values).
        y : numpy.ndarray or None, optional
            Pre-allocated flat 1-D output vector of length equal to the total
            size of all output variables.  If ``None`` a new array is allocated.
            Only used when ``form`` is ``'f'`` or ``'fdfdx'``.
        J : numpy.ndarray or None, optional
            Pre-allocated Jacobian matrix of shape ``(n_outputs, n_inputs)``.
            If ``None`` a new array is allocated.  Only used when ``form`` is
            ``'dfdx'`` or ``'fdfdx'``.

        Returns
        -------
        numpy.ndarray
            When ``form='f'``: the flat output vector ``y``.
        numpy.ndarray
            When ``form='dfdx'``: the Jacobian matrix ``J`` of shape
            ``(n_outputs, n_inputs)``.
        tuple of numpy.ndarray
            When ``form='fdfdx'``: the tuple ``(y, J)``.
        """
        problem = self.problem

        if len(x) != self._input_len:
            msg = (f"{self.msginfo}: expected x argument with length {self._input_len}, "
                   f"but found {len(x)}")
            raise ValueError(msg)

        if self.form in ("f", "fdfdx"):
            if y is not None:
                if len(y) != self._output_len:
                    msg = (f"{self.msginfo}: expected y argument with length {self._output_len}, "
                           f"but found {len(y)}")
                    raise ValueError(msg)
            else:
                y = np.zeros(self._output_len, dtype=x.dtype)

        if self.form in ("dfdx", "fdfdx"):
            if J is not None:
                if J.shape != (self._output_len, self._input_len):
                    msg = (f"{self.msginfo}: expected J argument with shape "
                           f"{(self._output_len, self._input_len)}, but found {J.shape}")
                    raise ValueError(msg)
            else:
                J = np.zeros((self._output_len, self._input_len), dtype=x.dtype)

        self._vector_to_problem(x, self._input_metadata)
        # Do I really have to do this every time?
        problem.run_model()

        if self.form in ("f", "fdfdx"):
            self._problem_to_vector(y, self._output_metadata)

        if self.form in ("dfdx", "fdfdx"):
            totals = self._total_jac_info.compute_totals()
            # `_TotalJacInfo.compute_totals` appears to use constraint/objective aliases.
            self._totals_to_jacobian(J, totals)

        if self.form == "f":
            return y
        elif self.form == "dfdx":
            return J
        elif self.form == "fdfdx":
            return y, J

    def get_input_val(self, name):
        """
        Return the current value of an input variable as a flat 1-D array.

        Parameters
        ----------
        name : str
            Name (or alias) of the input variable, as it was registered in
            ``input_vars``.

        Returns
        -------
        numpy.ndarray
            Current value of the variable, flattened to 1-D.  If ``indices``
            were specified for this variable, only the selected elements are
            returned.

        Raises
        ------
        ValueError
            If ``name`` is not among the registered input variables.
        """
        try:
            vmetadata = self._input_metadata[name]
        except KeyError:
            msg = f"{self.msginfo}: {name} is not an input variable"
            raise ValueError(msg)

        units = vmetadata["units"]
        indices = vmetadata["indices"]
        val = self.problem.get_val(name, units=units, indices=indices)
        val.shape = (-1,)
        return val

    def get_output_val(self, name):
        """
        Return the current value of an output variable as a flat 1-D array.

        Parameters
        ----------
        name : str
            Name (or alias) of the output variable, as it was registered in
            ``output_vars``.

        Returns
        -------
        numpy.ndarray
            Current value of the variable, flattened to 1-D.  If ``indices``
            were specified for this variable, only the selected elements are
            returned.

        Raises
        ------
        ValueError
            If ``name`` is not among the registered output variables.
        """
        try:
            vmetadata = self._output_metadata[name]
        except KeyError:
            msg = f"{self.msginfo}: {name} is not an output variable"
            raise ValueError(msg)

        units = vmetadata["units"]
        indices = vmetadata["indices"]
        val = self.problem.get_val(name, units=units, indices=indices).flatten()
        val.shape = (-1,)
        return val

    @property
    def form(self):
        """
        Return the form of this callback.

        Returns
        -------
        str
            One of ``'f'``, ``'dfdx'``, or ``'fdfdx'``.
        """
        return self._form

    @property
    def input_var_names(self):
        """
        Return input variable names.

        Returns
        -------
        list of str
            input variable names.
        """
        return list(self._input_metadata.keys())

    @property
    def output_var_names(self):
        """
        Return output variable names.

        Returns
        -------
        list of str
            output variable names.
        """
        return list(self._output_metadata.keys())

    @property
    def msginfo(self):
        """
        Return info to prepend to messages.

        Returns
        -------
        str
            Info to prepend to messages.
        """
        return type(self).__name__

    @property
    def problem(self):
        """
        Return the Problem associated with this callback.

        Returns
        -------
        Problem
            The Problem instance that was passed to the constructor.
        """
        return self._problem

    def create_input_vector(self, return_index_map=False):
        """
        Create a flat input vector pre-populated with the current problem values.

        Allocates a zero-filled 1-D array of the correct length and copies the
        current values of all registered input variables into it.  The returned
        array can be modified in-place and then passed directly to ``__call__``.

        If `return_index_map` is `True`, a `dict` mapping each input variable
        name to a `slice` object that indexes the variable is returned as the
        second value.

        Returns
        -------
        numpy.ndarray
            Flat 1-D array of length equal to the total size of all input variables,
            initialised with the current problem values.
        """
        x0 = np.zeros(self._input_len)
        self._problem_to_vector(x0, self._input_metadata)
        if return_index_map:
            index_map = {}
            for valias, vmeta in self._input_metadata.items():
                index_map[valias] = slice(*vmeta["offsets"])
            return x0, index_map
        else:
            return x0

    def create_output_vector(self, return_index_map=False):
        """
        Create a flat output vector pre-populated with the current problem values.

        Allocates a zero-filled 1-D array of the correct length and copies the
        current values of all registered output variables into it.  The returned
        array can be passed as the ``y`` argument to ``__call__`` to avoid
        allocating a new array on each call.

        If `return_index_map` is `True`, a `dict` mapping each output variable
        name to a `slice` object that indexes the variable is returned as the
        second value.

        Returns
        -------
        numpy.ndarray
            Flat 1-D array of length equal to the total size of all output variables,
            initialised with the current problem values.
        """
        y0 = np.zeros(self._output_len)
        self._problem_to_vector(y0, self._output_metadata)
        if return_index_map:
            index_map = {}
            for valias, vmeta in self._output_metadata.items():
                index_map[valias] = slice(*vmeta["offsets"])
            return y0, index_map
        return y0

    def create_jacobian_matrix(self, return_index_map=False):
        """
        Create a zero-filled Jacobian matrix of the correct shape.

        Allocates a 2-D array of zeros with shape ``(n_outputs, n_inputs)``.
        The returned array can be passed as the ``J`` argument to ``__call__``
        to avoid allocating a new matrix on each call.

        If `return_index_map` is `True`, a `dict` mapping a tuple of output-input variable
        name pairs to two `slice` objects that indexes the Jacobian is returned as the
        second value.

        Returns
        -------
        numpy.ndarray
            Zero-filled 2-D array of shape ``(n_outputs, n_inputs)``.
        """
        J0 = np.zeros((self._output_len, self._input_len))
        if return_index_map:
            index_map = {}
            for ovalias, ovmeta in self._output_metadata.items():
                for ivalias, ivmeta in self._input_metadata.items():
                    index_map[ovalias, ivalias] = (slice(*ovmeta['offsets']),
                                                   slice(*ivmeta['offsets']))
            return J0, index_map
        else:
            return J0


