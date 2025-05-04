import numpy as np
import openmdao.api as om
import torch
import torch.nn as nn
import warnings

# Try importing torch.func for jacrev/jacfwd, handle if not available
try:
    import torch.func as functorch
    _has_functorch = True
except ImportError:
    functorch = None
    _has_functorch = False
    # Warning generated only if analytical partials requested later


class PyTorchComponent(om.ExplicitComponent):
    """
    An OpenMDAO Component that wraps a pre-loaded PyTorch model instance
    for inference and computes analytical partial derivatives using torch.func.

    Requires the user to provide the model instance and explicit
    input/output specifications (name, shape, etc.).
    """

    def initialize(self):
        """Declare options for the component."""
        # Model must be a pre-loaded instance (no default makes it mandatory)
        self.options.declare('model', types=(nn.Module),
                             desc='A pre-loaded PyTorch model instance (nn.Module).')
        # Specs are now mandatory (no default)
        self.options.declare('input_specs', types=list,
                             desc='List of tuples for inputs: [(name, shape, kwargs_dict), ...]')
        self.options.declare('output_specs', types=list,
                             desc='List of tuples for outputs: [(name, shape, kwargs_dict), ...]')

        # These options remain optional as they have defaults
        self.options.declare('model_expects_batch', types=bool, default=True,
                             desc='Set True if the PyTorch model forward pass expects a batch '
                             'dimension.')
        self.options.declare('device', types=str, default='cpu',
                             desc="Device model should run on ('cpu' or 'cuda'). Component assumes "
                             "model is already on this device.")
        self.options.declare('torch_dtype', default=torch.float64, # Default float64
                             desc="PyTorch dtype for tensor conversions (default: float64). Assumes"
                             " model parameters match.")
        self.options.declare('use_analytical_partials', types=bool, default=True,
                             desc='If True, attempt to use torch.func for analytical partials. '
                             'If False or torch.func unavailable, use FD.')

        # Internal storage
        self._pytorch_model = None
        self._input_names = []
        self._output_names = []
        self._input_shapes = {}
        self._output_shapes = {}
        self._input_name_to_idx = {}
        self._output_name_to_idx = {}
        self._analytic_partials = False # Initialize flag, will be set in setup_partials


    def setup(self):
        """Declare OpenMDAO inputs and outputs based on provided specs and model instance."""
        # Model instance is directly provided and mandatory
        self._pytorch_model = self.options['model']
        user_input_specs = self.options['input_specs']
        user_output_specs = self.options['output_specs']
        self.device = torch.device(self.options['device'])
        self.torch_dtype = self.options['torch_dtype']
        self.model_expects_batch = self.options['model_expects_batch']

        # Basic validation
        if not isinstance(self._pytorch_model, nn.Module):
             raise TypeError("The 'model' option must be an nn.Module instance.")
        if not user_input_specs:
            raise ValueError("input_specs cannot be empty.")
        if not user_output_specs:
            raise ValueError("output_specs cannot be empty.")

        # Assume model is already on the correct device and has the correct dtype (e.g., float64)
        print(f"Using provided model instance: {type(self._pytorch_model).__name__}")
        try:
            # Check device consistency if model has parameters
            model_device = next(self._pytorch_model.parameters()).device
            if model_device != self.device:
                 warnings.warn(f"Provided model parameters are on device '{model_device}' but "
                               f"component device is set to '{self.device}'. Ensure consistency.",
                               UserWarning)
            # Check dtype consistency if model has parameters
            model_dtype = next(self._pytorch_model.parameters()).dtype
            if model_dtype != self.torch_dtype:
                warnings.warn(f"Provided model parameters have dtype '{model_dtype}' but "
                              f"component torch_dtype is set to '{self.torch_dtype}'. Ensure "
                              "consistency.", UserWarning)

        except StopIteration:
             print("Model has no parameters to check device/dtype for.")

        # Ensure model is in evaluation mode
        self._pytorch_model.eval()

        # --- Populate internal specs from user input ---
        self._input_names = [spec[0] for spec in user_input_specs]
        self._output_names = [spec[0] for spec in user_output_specs]
        self._input_shapes = {spec[0]: spec[1] for spec in user_input_specs}
        self._output_shapes = {spec[0]: spec[1] for spec in user_output_specs}
        self._input_name_to_idx = {name: i for i, name in enumerate(self._input_names)}
        self._output_name_to_idx = {name: i for i, name in enumerate(self._output_names)}

        # --- Add OpenMDAO Inputs & Outputs ---
        print("Adding OpenMDAO variables from user specs...")
        for i, name in enumerate(self._input_names):
            shape = self._input_shapes[name]
            kwargs = {}
            if len(user_input_specs[i]) > 2:
                 kwargs = user_input_specs[i][2].copy() # Use copy

            default_val = kwargs.pop('val', np.ones(shape))
            if np.isscalar(default_val):
                default_val = float(default_val)
            else:
                default_val = default_val.astype(float) # Ensure OpenMDAO float type

            self.add_input(name, shape=shape, val=default_val, **kwargs)
            print(f"  Added input: name='{name}', shape={shape}")

        for i, name in enumerate(self._output_names):
            shape = self._output_shapes[name]
            kwargs = {}
            if len(user_output_specs[i]) > 2:
                 kwargs = user_output_specs[i][2].copy() # Use copy

            self.add_output(name, shape=shape, **kwargs)
            print(f"  Added output: name='{name}', shape={shape}")


    def setup_partials(self):
        """Declare partial derivatives using analytical method if enabled and available."""
        use_analytical = self.options['use_analytical_partials']
        declared_method = 'fd' # Default to finite difference

        if use_analytical and _has_functorch:
            declared_method = 'exact'
            print("Declared partials using analytical method ('exact') via torch.func.")
        elif use_analytical and not _has_functorch:
            warnings.warn("torch.func (functorch) not found, but analytical partials requested. "
                          "Falling back to finite differencing ('fd'). Install PyTorch >= 1.12+ for"
                           " analytical partials.", UserWarning)
            declared_method = 'fd'
        else:
             print(f"Declared partials using finite differencing ('fd'). Analytical partials "
                   f"requested: {use_analytical}")
             declared_method = 'fd'

        self.declare_partials('*', '*', method=declared_method)
        # Store flag based on actual declaration for compute_partials
        self._analytic_partials = (declared_method == 'exact')


    def _prepare_input_tensors(self, inputs, requires_grad=False):
        """Helper to convert OpenMDAO inputs to PyTorch tensors."""
        input_tensors = []
        for name in self._input_names:
            om_input_np = inputs[name]
            # Use component's target dtype for conversion from NumPy
            target_dtype = self.torch_dtype
            input_tensor = torch.from_numpy(om_input_np).to(dtype=target_dtype, device=self.device)

            if requires_grad:
                input_tensor.requires_grad_(True)

            if self.model_expects_batch:
                input_shape_from_spec = self._input_shapes[name]
                if len(input_tensor.shape) == len(input_shape_from_spec):
                    input_tensor = input_tensor.unsqueeze(0)
            input_tensors.append(input_tensor)
        return tuple(input_tensors)


    def _process_output_tensors(self, model_output_tensors):
        """Helper to process model outputs back to dict of NumPy arrays."""
        if not isinstance(model_output_tensors, tuple):
            model_output_tensors = (model_output_tensors,)

        if len(model_output_tensors) != len(self._output_names):
             raise ValueError(f"Model produced {len(model_output_tensors)} output(s), "
                              f"but {len(self._output_names)} output name(s) were specified.")

        processed_outputs = {}
        for i, name in enumerate(self._output_names):
            output_tensor = model_output_tensors[i]
            if self.model_expects_batch:
                oshape = self._output_shapes[name]
                if len(output_tensor.shape) == len(oshape) + 1 and output_tensor.shape[0] == 1:
                    output_tensor = output_tensor.squeeze(0)

            output_np = output_tensor.detach().cpu().numpy()
            # Ensure float64 for OpenMDAO
            processed_outputs[name] = output_np.astype(float)

        return processed_outputs


    def compute(self, inputs, outputs):
        """Run the PyTorch model inference."""
        self._pytorch_model.eval()
        input_tensors = self._prepare_input_tensors(inputs, requires_grad=False)

        with torch.no_grad():
             if len(input_tensors) == 1:
                 model_output = self._pytorch_model(input_tensors[0])
             else:
                 model_output = self._pytorch_model(*input_tensors)

        processed_outputs = self._process_output_tensors(model_output)
        for name, value in processed_outputs.items():
            outputs[name][:] = value


    def _model_forward_wrapper(self, *input_tensors_with_grad):
        """ A wrapper around the model's forward pass suitable for jacrev/jacfwd. """
        processed_inputs = []
        if self.model_expects_batch:
            for i, tensor in enumerate(input_tensors_with_grad):
                 spec_shape = self._input_shapes[self._input_names[i]]
                 if len(tensor.shape) == len(spec_shape):
                     processed_inputs.append(tensor.unsqueeze(0))
                 else:
                     processed_inputs.append(tensor)
            processed_inputs = tuple(processed_inputs)
        else:
             processed_inputs = input_tensors_with_grad

        if len(processed_inputs) == 1:
            model_outputs = self._pytorch_model(processed_inputs[0])
        else:
            model_outputs = self._pytorch_model(*processed_inputs)

        if not isinstance(model_outputs, tuple):
            model_outputs = (model_outputs,)

        final_outputs = []
        if self.model_expects_batch:
            for i, tensor in enumerate(model_outputs):
                 spec_shape = self._output_shapes[self._output_names[i]]
                 if len(tensor.shape) == len(spec_shape) + 1 and tensor.shape[0] == 1:
                     final_outputs.append(tensor.squeeze(0))
                 else:
                     final_outputs.append(tensor)
            final_outputs = tuple(final_outputs)
        else:
            final_outputs = model_outputs

        return final_outputs


    def compute_partials(self, inputs, partials):
        """
        Compute analytical partial derivatives if configured, otherwise OpenMDAO handles FD.
        """
        # Check the flag set during setup_partials based on the actual method declared
        if self._analytic_partials:
            if not _has_functorch:
                # This case should ideally be prevented by setup_partials fallback. defensive check
                raise RuntimeError("Analytical partials configured but torch.func not available.")

            self._pytorch_model.eval() # Ensure eval mode

            input_tensors_unbatched = []
            total_input_size = 0
            for name in self._input_names:
                 om_input_np = inputs[name]
                 # Use component's target dtype (should be float64)
                 tensor = torch.from_numpy(om_input_np).to(dtype=self.torch_dtype,
                                                           device=self.device)
                 tensor.requires_grad_(True)
                 input_tensors_unbatched.append(tensor)
                 total_input_size += tensor.numel()
            input_tensors_tuple = tuple(input_tensors_unbatched)

            total_output_size = 0
            for name in self._output_names:
                total_output_size += np.prod(self._output_shapes[name])

            # Choose jacrev or jacfwd based on total sizes
            if total_output_size <= total_input_size:
                jacobian_func = functorch.jacrev
                # mode_selected = "reverse (jacrev)"
            else:
                jacobian_func = functorch.jacfwd
                # mode_selected = "forward (jacfwd)"
            # print(f"DEBUG: Using {mode_selected}") # Optional

            argnums_to_diff = tuple(range(len(input_tensors_tuple)))
            # Compute all Jacobians: structure is result[input_idx][output_idx] = d(output)/d(input)
            all_jacobians_nested = jacobian_func(self._model_forward_wrapper,
                                                 argnums=argnums_to_diff)(*input_tensors_tuple)

            # Populate partials dictionary for the keys OpenMDAO expects for this comp/input pair
            for (of_name, wrt_name) in partials:
                if wrt_name not in self._input_names:
                    continue # Skip if wrt isn't a component input

                of_idx = self._output_name_to_idx[of_name]
                wrt_idx = self._input_name_to_idx[wrt_name]

                # Extract the specific Jacobian tensor: result[wrt_idx][of_idx]
                jacobian_tensor = all_jacobians_nested[wrt_idx][of_idx]

                # Reshape Jacobian to OpenMDAO's 2D format: (output_flat_size, input_flat_size)
                of_shape = self._output_shapes[of_name]
                wrt_shape = self._input_shapes[wrt_name]
                of_size = np.prod(of_shape)
                wrt_size = np.prod(wrt_shape)

                try:
                    # Check element count first for safety
                    if jacobian_tensor.numel() != (of_size * wrt_size):
                         raise ValueError("Jacobian element count mismatch for "
                                          f"({of_name}/{wrt_name}): Got {jacobian_tensor.numel()}, "
                                          f"Expected {of_size * wrt_size}")
                    # Reshape directly into the pre-allocated NumPy array if possible?
                    # No, OpenMDAO gives us the 'partials' dict-like object. Assign to it.
                    jacobian_tensor_reshaped = jacobian_tensor.reshape(of_size, wrt_size)

                except Exception as e:
                     raise RuntimeError(f"Could not reshape Jacobian for ({of_name}/{wrt_name}) "
                                        f"from {jacobian_tensor.shape} to ({of_size}, {wrt_size}). "
                                        f"Error: {e}") from e

                # Convert to NumPy array (detaching first), ensure float64, and store
                jacobian_np = jacobian_tensor_reshaped.detach().cpu().numpy().astype(float)
                partials[of_name, wrt_name] = jacobian_np # Assign to the array provided by OpenMDAO

        # else: If method is 'fd', OpenMDAO's default compute_partials or the one
        # from ExplicitComponent handles it based on component methods provided.


# ================== EXAMPLE USAGE (Simplified) ==================

if __name__ == "__main__":

    # --- Common Model Definition ---
    class SimpleMLP(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.layer1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.layer2 = nn.Linear(hidden_size, output_size)
        def forward(self, x):
            x = self.layer1(x)
            x = self.relu(x)
            x = self.layer2(x)
            return x

    input_features = 3
    hidden_features = 10
    output_features = 2

    # --- User prepares the model instance BEFORE creating the component ---
    print("\n" + "="*30)
    print(" Example: User loads model, provides instance and specs ")
    print("="*30)

    # 1. Create the model instance
    prepared_model = SimpleMLP(input_features, hidden_features, output_features)
    # 2. Set desired dtype (float64 for OpenMDAO compatibility)
    prepared_model.to(dtype=torch.float64)
    # 3. Set device
    device = torch.device('cpu') # or 'cuda'
    prepared_model.to(device)
    # 4. Ensure eval mode
    prepared_model.eval()
    print(f"Prepared model instance: type={type(prepared_model).__name__}, "
          f"dtype={next(prepared_model.parameters()).dtype}, "
          f"device={next(prepared_model.parameters()).device}")

    # --- Define specs EXPLICITLY ---
    # List of tuples: (name, shape, kwargs_for_add_input/output)
    input_spec = [('input_vec', (input_features,), {'units': None})]
    output_spec = [('output_val', (output_features,), {'units': None})]

    # --- Create OpenMDAO Problem ---
    prob = om.Problem()

    # Instantiate component, passing the PREPARED model instance and specs
    pytorch_wrapper = PyTorchComponent(model=prepared_model,          # Pass instance
                                       input_specs=input_spec,      # Provide specs
                                       output_specs=output_spec,     # Provide specs
                                       model_expects_batch=True,
                                       device=str(device),          # Pass device name as string
                                       torch_dtype=torch.float64,   # Consistent dtype
                                       use_analytical_partials=_has_functorch)

    ivc = om.IndepVarComp()
    # Use float64 for the input value numpy array
    ivc.add_output('model_input_val', val=np.array([0.5, -1.0, 1.5], dtype=np.float64),
                   shape=(input_features,))
    prob.model.add_subsystem('indep_vars', ivc)

    # Use names defined in user_input_spec for promotion
    prob.model.add_subsystem('pytorch_model', pytorch_wrapper,
                             promotes_inputs=[('input_vec', 'model_input_val')])

    # Setup and Run
    prob.setup()
    print("\nRunning OpenMDAO Problem...")
    prob.run_model()
    print("Run complete.")
    print("\nInput:", prob.get_val('pytorch_model.input_vec'))
    print("Output:", prob.get_val('pytorch_model.output_val'))

    # Check partials
    print("\nChecking partials...")
    # Use a smaller step for float64 checks if using FD method for comparison
    check_data = prob.check_partials(compact_print=True, method='fd', step=1e-7)

    print("\nScript finished.")