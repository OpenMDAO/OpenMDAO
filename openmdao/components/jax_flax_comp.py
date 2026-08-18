from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
import openmdao.api as om


class JaxFlaxComp(om.JaxExplicitComponent):
    """
    OpenMDAO component wrapper for fixed-shape JAX/Flax neural-network models. 

    This component is intended as a generalization of a simple DenseNNComp:
    instead of hard-coding network layers inside the OpenMDAO component, the user
    provides a Flax model and trained parameter PyTree. OpenMDAO owns the input and
    output interface; Flax owns the model internals.

    Parameters
    ----------
    model : flax.linen.Module
        A Flax module instance. The component calls ``model.apply``.
    params : PyTree or Mapping
        Trained Flax parameters or a full Flax variables collection. If this is
        already a collection containing a ``"params"`` key, it is passed directly
        to ``model.apply``. Otherwise it is wrapped as ``{"params": params}``.
    input_shapes : dict[str, tuple[int, ...]]
        Mapping from OpenMDAO input names to array shapes.
    output_shapes : dict[str, tuple[int, ...]]
        Mapping from OpenMDAO output names to array shapes.
    input_units : dict[str, str | None], optional
        Optional units for each OpenMDAO input.
    output_units : dict[str, str | None], optional
        Optional units for each OpenMDAO output.
    input_defaults : dict[str, Any], optional
        Optional initial values for each input.
    output_defaults : dict[str, Any], optional
        Optional initial values for each output.
    """

    def initialize(self):
        self.options.declare("model")
        self.options.declare("params")
        self.options.declare("input_shapes", types=Mapping)
        self.options.declare("output_shapes", types=Mapping)
        self.options.declare("input_units", default=None, allow_none=True)
        self.options.declare("output_units", default=None, allow_none=True)
        self.options.declare("input_defaults", default=None, allow_none=True)
        self.options.declare("output_defaults", default=None, allow_none=True)

    def setup(self):
        input_shapes = dict(self.options["input_shapes"])
        output_shapes = dict(self.options["output_shapes"])
        input_units = self.options["input_units"] or {}
        output_units = self.options["output_units"] or {}
        input_defaults = self.options["input_defaults"] or {}
        output_defaults = self.options["output_defaults"] or {}

        self._input_names = tuple(input_shapes)
        self._output_names = tuple(output_shapes)

        for name, shape in input_shapes.items():
            val = input_defaults.get(name, jnp.zeros(shape))
            self.add_input(
                name,
                val=val,
                shape=shape,
                units=input_units.get(name),
            )

        for name, shape in output_shapes.items():
            val = output_defaults.get(name, jnp.zeros(shape))
            self.add_output(
                name,
                val=val,
                shape=shape,
                units=output_units.get(name),
            )

    def _variables(self):
        variables = self.options["params"]

        # Already a full Flax variables collection
        if isinstance(variables, Mapping) and "params" in variables:
            return variables

        # Raw parameter PyTree
        return {"params": variables}

    def _pack_model_inputs(self, inputs):
        return {name: jnp.asarray(inputs[name]) for name in self._input_names}
    
    def _unpack_model_outputs(self, raw_outputs):
        # Enable dict or array outputs
        if isinstance(raw_outputs, Mapping):
            return tuple(
                jnp.asarray(raw_outputs[name])
                for name in self._output_names
            )

        if len(self._output_names) == 1:
            return (jnp.asarray(raw_outputs),)
        
        raise ValueError(
            "Flax model returned a non-dict output, but multiple OpenMDAO "
            f"outputs were declared: {self._output_names}. "
            "For multiple outputs, the Flax model should return a dictionary."
        )

    def compute_primal(self, *openmdao_inputs):
        """Evaluate the Flax model using JAX-compatible arrays.

        '``om.JaxExplicitComponent`` passes OpenMDAO inputs positionally in the
        same order they were added in ``setup``. This method repacks them into
        a dictionary of named JAX arrays before calling ``model.apply``.
        """
        input_dict = {
            name: value for name, value in zip(self._input_names, openmdao_inputs)
        }

        model = self.options["model"]
        variables = self._variables()
        model_inputs = self._pack_model_inputs(input_dict)

        raw_outputs = model.apply(variables, model_inputs)

        return self._unpack_model_outputs(raw_outputs)