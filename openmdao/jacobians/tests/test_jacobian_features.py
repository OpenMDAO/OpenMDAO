import unittest
import numpy as np
import itertools
from scipy.sparse import coo_matrix, csr_matrix, issparse

from openmdao.api import IndepVarComp, Group, Problem, ExplicitComponent, DenseMatrix, \
     GlobalJacobian, CsrMatrix, CooMatrix
from openmdao.jacobians.default_jacobian import DefaultJacobian
from openmdao.devtools.testutil import assert_rel_error


class SimpleComp(ExplicitComponent):
    def initialize_variables(self):
        self.add_input('x', shape=1)
        self.add_input('y1', shape=2)
        self.add_input('y2', shape=2)
        self.add_input('z', shape=(2, 2))

        self.add_output('f', shape=1)
        self.add_output('g', shape=(2, 2))

        self.declare_partials('f', ['y1', 'y2'], dependent=False)
        self.declare_partials('g', 'z', dependent=False)

    def compute(self, inputs, outputs):
        outputs['f'] = np.sum(inputs['z']) + inputs['x']
        outputs['g'] = np.outer(inputs['y1'], inputs['y2']) + inputs['x']*np.eye(2)


class TestJacobianFeatures(unittest.TestCase):

    def setUp(self):
        self.model = model = Group()
        model.add_subsystem('simple', SimpleComp(), promotes=['x', 'y1', 'y2', 'z', 'f', 'g'])
        self.problem = problem = Problem(model=model)
        model.jacobian = GlobalJacobian(matrix_class=CooMatrix)
        problem.setup(check=False)


    def test_something(self):
        problem = self.problem
        problem['x'] = 2
        problem['y1'] = [1, 2]
        problem['y2'] = [1, 2]
        problem['z'] = [[1, 2], [3, 4]]

        problem.run_model()
        print(problem['f'])
        print(problem['g'])

if __name__ == '__main__':
    unittest.main()
