
import numpy as np
from scipy.sparse import coo_matrix, issparse

try:
    import jax
    jax.config.update("jax_enable_x64", True)  # jax by default uses 32 bit floats
    import jax.numpy as jnp
    from jax.experimental import sparse as jsparse
except ImportError:
    jax = None
    jnp = np

from openmdao.core.problem import Problem
from openmdao.components.jax_explicit_comp import JaxExplicitComponent
from openmdao.core.explicitcomponent import ExplicitComponent


class SparsityComp(ExplicitComponent):
    def __init__(self, sparsity, **kwargs):
        super(SparsityComp, self).__init__(**kwargs)
        if isinstance(sparsity, np.ndarray):
            self.use_sparse = False
            self.sparsity = sparsity
            self.nzrows, self.nzcols = np.nonzero(self.sparsity)
        else:
            self.use_sparse = True
            self.sparsity = sparsity.tocoo()
            self.nzrows, self.nzcols = self.sparsity.row, self.sparsity.col

    def setup(self):
        self.add_input('x', shape=self.sparsity.shape[1])
        self.add_output('y', shape=self.sparsity.shape[0])

    def setup_partials(self):
        if self.use_sparse:
            self.declare_partials('y', 'x', rows=self.nzrows, cols=self.nzcols)
        else:
            self.declare_partials('y', 'x')

    def compute(self, inputs, outputs):
        outputs['y'] = self.sparsity @ inputs['x']

    def compute_partials(self, inputs, partials):
        if self.use_sparse:
            partials['y', 'x'] = self.sparsity.data
        else:
            partials['y', 'x'] = self.sparsity  # [self.nzrows, self.nzcols]


class JaxSparsityComp(JaxExplicitComponent):
    """
    A simple component that multiplies a sparse matrix by an input vector.

    The sparsity structure is defined by the 'sparsity' argument, and the data values are
    just the (index + 1) of the nonzeros in the sparsity structure.

    This component is used to test coloring. The jacobian can be computed with and
    without coloring and the two results should be the same.

    If `declare_partials` is False, then sparsity must be computed in order to determine
    rows/cols for the subjacobian declarations.

    Parameters
    ----------
    sparsity : ndarray or coo_matrix
        Sparsity structure to be tested.

    Attributes
    ----------
    sparsity : coo_matrix
        Dense or sparse version of the sparsity structure.
    """
    def __init__(self, sparsity, declare_partials=False, **kwargs):
        super(JaxSparsityComp, self).__init__(**kwargs)
        self.declare = declare_partials
        if isinstance(sparsity, (np.ndarray, jnp.ndarray)):
            self.sparsity = jnp.array(sparsity)
            self.nzrows, self.nzcols = jnp.nonzero(self.sparsity)
        elif issparse(sparsity):
            sparsity = sparsity.tocoo()
            self.nzrows, self.nzcols = jnp.array(sparsity.row), jnp.array(sparsity.col)
            indices = jnp.array(list(zip(self.nzrows, self.nzcols)), dtype=jnp.int32)
            self.sparsity = jsparse.BCOO((jnp.array(sparsity.data), indices), shape=sparsity.shape)
        else:
            raise ValueError(f"{self.msginfo}: no support for sparse type of {type(sparsity)}")

    def setup(self):
        self.add_input('x', shape=self.sparsity.shape[1])
        self.add_output('y', shape=self.sparsity.shape[0])

    def setup_partials(self):
        if self.declare:
            self.declare_partials('y', 'x', rows=self.nzrows, cols=self.nzcols)

    def compute_primal(self, x):
        return self.sparsity @ x


if __name__ == '__main__':
    from openmdao.test_suite.comp_tester import ComponentTester

    sparsity = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    sparsity_coo = coo_matrix(sparsity)
    jax_sparsity = jnp.array(sparsity)

    ComponentTester(SparsityComp, (sparsity,)).run()
    ComponentTester(SparsityComp, (sparsity_coo,)).run()
    ComponentTester(JaxSparsityComp, (jax_sparsity,)).run()
    ComponentTester(JaxSparsityComp, (jax_sparsity,), {'declare_partials': True}).run()

