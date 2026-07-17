from numpy import load
import jax.numpy as jnp
import openmdao.api as om
import jax
    
class DenseNNComp(om.JaxExplicitComponent):
    '''
    Fixed-weight dense neural network implemented as an OpenMDAO
    JaxExplicitComponent.

    This component loads a feedforward neural network from a saved
    weights file and evaluates the network using JAX operations so
    that OpenMDAO can automatically compute derivatives through the
    neural network.

    The component currently supports fully-connected dense layers with
    a configurable activation function applied after each hidden layer.

    A separate activation function may optionally be applied to the
    output layer.

    Supported activation functions:
        - sigmoid
        - relu
        - tanh

    The weights file is expected to contain:
        layer_sizes : array describing the number of neurons in each layer
        W0, b0      : weights and biases for layer 0
        W1, b1      : weights and biases for layer 1
        ...

    Example architecture:
        layer_sizes = [2, 4, 3, 1]

    corresponds to:
        2 inputs
            -> 4-neuron hidden layer
            -> 3-neuron hidden layer
            -> 1 output

    The component supports both:
        - single-point evaluations
        - vectorized evaluations with vmap for uncertainty quantification

    Vectorized mode is controlled using vec_size > 1.

    Notes
    -----
    By default, the output layer is linear (no activation function).
    This behavior can be modified using the output_activation option.

    This component represents a simple feedforward dense neural network.
    It does not currently support:
        - convolutional layers
        - recurrent layers
        - transformers
        - attention mechanisms
        - branching computation graphs
        - normalization layers
        - pooling layers
    '''

    def initialize(self):
        # Number of vectorized samples to evaluate simultaneously.
        self.options.declare("vec_size", default=1, types=int)

        # Path to the neural-network weights and layers .npz file.
        self.options.declare("weights_file", types=str)

        # Activation function applied after each dense layer.
        self.options.declare(
            "activation",
            default="sigmoid",
            values=("sigmoid", "relu", "tanh"),
        )

        # Optional activation function applied to the output layer.
        # If None, the output layer remains linear.
        self.options.declare(
            "output_activation",
            default=None,
            values=(None, "sigmoid", "relu", "tanh"),
        )

    def setup(self):
        n = self.options["vec_size"]

        layer_sizes, self._weights, self._biases = self._load_weights()

        num_inputs = layer_sizes[0]
        num_outputs = layer_sizes[-1]

        if n == 1:
            self.add_input("x", shape=(num_inputs,))
            self.add_output("y", shape=(num_outputs,))
        else:
            self.add_input("x", shape=(n, num_inputs))
            self.add_output("y", shape=(n, num_outputs))

    def _load_weights(self):
        data = load(self.options["weights_file"])

        layer_sizes = data["layer_sizes"].astype(int)

        weights = []
        biases = []

        for i in range(len(layer_sizes) - 1):
            W = data[f"W{i}"]
            b = data[f"b{i}"]

            expected_W_shape = (layer_sizes[i], layer_sizes[i + 1])
            expected_b_shape = (layer_sizes[i + 1],)

            if W.shape != expected_W_shape:
                raise ValueError(
                    f"W{i} has shape {W.shape}, expected {expected_W_shape}."
                )

            if b.shape != expected_b_shape:
                raise ValueError(
                    f"b{i} has shape {b.shape}, expected {expected_b_shape}."
                )

            weights.append(jnp.array(W))
            biases.append(jnp.array(b))

        return layer_sizes, weights, biases
    
    def _activate(self, z):
        activation = self.options["activation"]

        if activation == "sigmoid":
            return 1.0 / (1.0 + jnp.exp(-z))

        elif activation == "relu":
            return jnp.maximum(0.0, z)

        else:
            return jnp.tanh(z)
    
    def _evaluate_single(self, x):
        z = x

        for W, b in zip(self._weights[:-1], self._biases[:-1]):
            z = self._activate(jnp.dot(z, W) + b)

        y = jnp.dot(z, self._weights[-1]) + self._biases[-1]

        if self.options["output_activation"] is not None:
            y = self._activate(y)

        return y

    def compute_primal(self, x):
        if self.options["vec_size"] == 1:
            y = self._evaluate_single(x)
        else:
            y = jax.vmap(self._evaluate_single)(x)

        return (y,)