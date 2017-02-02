import unittest
import numpy as np
import scipy as sp
import itertools
from scipy.sparse import coo_matrix, csr_matrix, issparse

from openmdao.api import IndepVarComp, Group, Problem, ExplicitComponent, DenseMatrix, \
     GlobalJacobian, CsrMatrix, CooMatrix, ScipyIterativeSolver
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

        self.initialize_partials()

    def initialize_partials(self):
        pass

    def compute(self, inputs, outputs):
        outputs['f'] = np.sum(inputs['z']) + inputs['x']
        outputs['g'] = np.outer(inputs['y1'], inputs['y2']) + inputs['x']*np.eye(2)

    def compute_jacobian(self, inputs, outputs, jacobian):
        jacobian['f', 'x'] = 1.
        jacobian['f', 'z'] = np.ones((2, 2)).flat[:]

        jacobian['g', 'y1'] = np.hstack((np.array([[1, 1], [0, 0]]).flat,
                                         np.array([[0, 0], [1, 1]]).flat))

        jacobian['g', 'y2'] = np.hstack((np.array([[1, 0], [1, 0]]).flat,
                                         np.array([[0, 1], [0, 1]]).flat))

        jacobian['g', 'x'] = np.eye(2)


class SimpleCompDependence(SimpleComp):
    def initialize_partials(self):
        self.declare_partials('f', ['y1', 'y2'], dependent=False)
        self.declare_partials('g', 'z', dependent=False)

class SimpleCompConst(SimpleComp):
    def initialize_partials(self):
        self.declare_partials('f', ['y1', 'y2'], dependent=False)
        self.declare_partials('g', 'z', dependent=False)

        self.declare_partials('f', 'x', val=1.)
        self.declare_partials('f', 'z', val=np.ones((2, 2)))
        self.declare_partials('g', 'y1', val=[[1, 0], [1, 0], [0, 1], [0, 1]])
        self.declare_partials('g', 'y2', val=1., cols=[0, 0, 1, 1], rows=[0, 3, 0, 3])
        self.declare_partials('g', 'x', val=sp.sparse.coo_matrix(((1., 1.), ((0, 3), (0, 0)))))

    def compute_jacobian(self, inputs, outputs, jacobian):
        pass

class TestJacobianFeatures(unittest.TestCase):

    def setUp(self):
        self.model = model = Group()
        model.add_subsystem('input_comp', IndepVarComp(
            (('x', 1.),
             ('y1', np.ones(2)),
             ('y2', np.ones(2)),
             ('z', np.ones((2, 2))))
            ), promotes=['x', 'y1', 'y2', 'z'])

        self.problem = Problem(model=model)
        model.suppress_solver_output = True
        model.ln_solver = ScipyIterativeSolver()
        # model.jacobian = GlobalJacobian(matrix_class=CsrMatrix)


    def test_something(self):
        problem = self.problem
        model = problem.model
        model.add_subsystem('simple', SimpleComp(), promotes=['x', 'y1', 'y2', 'z', 'f', 'g'])
        problem.setup(check=False)
        problem['x'] = 2.
        problem['y1'] = [1, 2]
        problem['y2'] = [1, 2]
        problem['z'] = [[1, 2], [3, 4]]

        problem.run_model()
        assert_rel_error(self, problem['f'], 12)
        assert_rel_error(self, problem['g'], np.array([[3, 2], [2, 6]]))

    def test_dependence(self):
        problem = self.problem
        model = problem.model
        model.add_subsystem('simple', SimpleCompConst(), promotes=['x', 'y1', 'y2', 'z', 'f', 'g'])
        problem.setup(check=False)
        problem.run_model()
        totals = problem.compute_total_derivs(
            of=['f', 'g'],
            wrt=['x', 'y1', 'y2', 'z']
        )

        raise ValueError(totals)


if __name__ == '__main__':
    unittest.main()
