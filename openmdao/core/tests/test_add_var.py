"""Acceptance and developer tests for add_input and add_output."""
from __future__ import division

import numpy
import unittest

from openmdao.api import Problem, ExplicitComponent
from openmdao.devtools.testutil import assert_rel_error


class TestAddVarCompVal(ExplicitComponent):
    """Component for tests for declaring only default value."""

    def initialize_variables(self):
        self.add_input('x_a')
        self.add_input('x_b', 3.)
        self.add_input('x_c', (3., 3.))
        self.add_input('x_d', [3., 3.])
        self.add_input('x_e', 3. * numpy.ones((2, 2)))
        self.add_output('y_a')
        self.add_output('y_b', 6.)
        self.add_output('y_c', (6., 6., 6.))
        self.add_output('y_d', [6., 6., 6.])
        self.add_output('y_e', 6. * numpy.ones((3, 2)))


class TestAddVarCompShape(ExplicitComponent):
    """Component for tests for declaring only shape."""

    def initialize_variables(self):
        self.add_input('x_a', shape=2)
        self.add_input('x_b', shape=(2, 2))
        self.add_input('x_c', shape=[2, 2])
        self.add_output('y_a', shape=3)
        self.add_output('y_b', shape=(3, 3))
        self.add_output('y_c', shape=[3, 3])


class TestAddVarCompIndices(ExplicitComponent):
    """Component for tests for declaring only indices."""

    def initialize_variables(self):
        self.add_input('x_a', src_indices=0)
        self.add_input('x_b', src_indices=(0, 1))
        self.add_input('x_c', src_indices=[0, 1])
        self.add_input('x_d', src_indices=numpy.arange(6))
        self.add_input('x_e', src_indices=numpy.arange(6).reshape((3, 2)))
        self.add_output('y')


class TestAddVarCompScalarArray(ExplicitComponent):
    """Component for tests for declaring a scalar val with an array variable."""

    def initialize_variables(self):
        self.add_input('x_a', 2.0, shape=(6))
        self.add_input('x_b', 2.0, shape=(3, 2))
        self.add_input('x_c', 2.0, src_indices=numpy.arange(6))
        self.add_input('x_d', 2.0, src_indices=numpy.arange(6).reshape((3,2)))
        self.add_output('y_a', 3.0, shape=(6))
        self.add_output('y_b', 3.0, shape=(3, 2))


class TestAddVarCompArrayIndices(ExplicitComponent):
    """Component for tests for declaring with array val and array indices."""

    def initialize_variables(self):
        self.add_input('x_a', 2.0 * numpy.ones(6), src_indices=numpy.arange(6))
        self.add_input('x_b', 2.0 * numpy.ones((3, 2)), src_indices=numpy.arange(6).reshape((3, 2)))
        self.add_output('y')


class TestAddVarCompBounds(ExplicitComponent):
    """Component for tests for declaring bounds."""

    def initialize_variables(self):
        self.add_input('x')
        self.add_output('y_a', 2.0, lower=0.)
        self.add_output('y_b', 2.0, lower=0., upper=10.)
        self.add_output('y_c', 2.0 * numpy.ones(6),  lower=numpy.zeros(6), upper=10.)
        self.add_output('y_d', 2.0 * numpy.ones(6), lower=0., upper=[12, 10, 10, 10, 10, 12])
        self.add_output('y_e', 2.0 * numpy.ones((3, 2)), lower=numpy.zeros((3, 2)))


class TestAddVar(unittest.TestCase):

    def test_val(self):
        """Test declaring only default value."""
        p = Problem(model=TestAddVarCompVal())
        p.setup()

        assert_rel_error(self, p['x_a'], 1.)
        assert_rel_error(self, p['x_b'], 3.)
        assert_rel_error(self, p['x_c'], 3. * numpy.ones(2))
        assert_rel_error(self, p['x_d'], 3. * numpy.ones(2))
        assert_rel_error(self, p['x_e'], 3. * numpy.ones((2, 2)))
        assert_rel_error(self, p['y_a'], 1.)
        assert_rel_error(self, p['y_b'], 6.)
        assert_rel_error(self, p['y_c'], 6. * numpy.ones(3))
        assert_rel_error(self, p['y_d'], 6. * numpy.ones(3))
        assert_rel_error(self, p['y_e'], 6. * numpy.ones((3, 2)))

    def test_shape(self):
        """Test declaring only shape."""
        p = Problem(model=TestAddVarCompShape())
        p.setup()

        assert_rel_error(self, p['x_a'], numpy.ones(2))
        assert_rel_error(self, p['x_b'], numpy.ones((2, 2)))
        assert_rel_error(self, p['x_c'], numpy.ones((2, 2)))
        assert_rel_error(self, p['y_a'], numpy.ones(3))
        assert_rel_error(self, p['y_b'], numpy.ones((3, 3)))
        assert_rel_error(self, p['y_c'], numpy.ones((3, 3)))

    def test_indices(self):
        """Test declaring only indices."""
        p = Problem(model=TestAddVarCompIndices())
        p.setup()

        assert_rel_error(self, p['x_a'], 1.)
        assert_rel_error(self, p['x_b'], numpy.ones(2))
        assert_rel_error(self, p['x_c'], numpy.ones(2))
        assert_rel_error(self, p['x_d'], numpy.ones(6))
        assert_rel_error(self, p['x_e'], numpy.ones((3,2)))

    def test_scalar_array(self):
        """Test declaring a scalar val with an array variable."""
        p = Problem(model=TestAddVarCompScalarArray())
        p.setup()

        assert_rel_error(self, p['x_a'], 2. * numpy.ones(6))
        assert_rel_error(self, p['x_b'], 2. * numpy.ones((3, 2)))
        assert_rel_error(self, p['x_c'], 2. * numpy.ones(6))
        assert_rel_error(self, p['x_d'], 2. * numpy.ones((3, 2)))
        assert_rel_error(self, p['y_a'], 3. * numpy.ones(6))
        assert_rel_error(self, p['y_b'], 3. * numpy.ones((3, 2)))

    def test_array_indices(self):
        """Test declaring with array val and array indices."""
        p = Problem(model=TestAddVarCompArrayIndices())
        p.setup()

        assert_rel_error(self, p['x_a'], 2. * numpy.ones(6))
        assert_rel_error(self, p['x_b'], 2. * numpy.ones((3, 2)))

    def test_bounds(self):
        """Test declaring bounds."""
        p = Problem(model=TestAddVarCompBounds())
        p.setup()

        assert_rel_error(self, p['y_a'], 2.)
        assert_rel_error(self, p['y_b'], 2.)
        assert_rel_error(self, p['y_c'], 2. * numpy.ones(6))
        assert_rel_error(self, p['y_d'], 2. * numpy.ones(6))
        assert_rel_error(self, p['y_e'], 2. * numpy.ones((3, 2)))


if __name__ == '__main__':
    unittest.main()
