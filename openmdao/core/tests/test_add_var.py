"""Acceptance and developer tests for add_input and add_output."""
import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal, assert_warning
from openmdao.utils.om_warnings import OMDeprecationWarning


class CompAddWithDefault(om.ExplicitComponent):
    """Component for tests for declaring only default value."""

    def setup(self):
        self.add_input('x_a')
        self.add_input('x_b', val=3.)
        self.add_input('x_c', val=(3., 3.))
        self.add_input('x_d', val=[3., 3.])
        self.add_input('x_e', val=3. * np.ones((2, 2)))

        self.add_output('y_a')
        self.add_output('y_b', val=6.)
        self.add_output('y_c', val=(6., 6., 6.))
        self.add_output('y_d', val=[6., 6., 6.])
        self.add_output('y_e', val=6. * np.ones((3, 2)))


class CompAddWithShape(om.ExplicitComponent):
    """Component for tests for declaring only shape."""

    def setup(self):
        self.add_input('x_a', shape=2)
        self.add_input('x_b', shape=(2, 2))
        self.add_input('x_c', shape=[2, 2])

        self.add_output('y_a', shape=3)
        self.add_output('y_b', shape=(3, 3))
        self.add_output('y_c', shape=[3, 3])


class CompAddArrayWithScalar(om.ExplicitComponent):
    """Component for tests for declaring a scalar val with an array variable."""

    def setup(self):
        self.add_input('x_a', val=2.0, shape=(6))
        self.add_input('x_b', val=2.0, shape=(3, 2))

        self.add_output('y_a', val=3.0, shape=(6))
        self.add_output('y_b', val=3.0, shape=(3, 2))


class CompAddWithBounds(om.ExplicitComponent):
    """Component for tests for declaring bounds."""

    def setup(self):
        self.add_input('x')

        self.add_output('y_a', val=2.0, lower=0.)
        self.add_output('y_b', val=2.0, lower=0., upper=10.)
        self.add_output('y_c', val=2.0 * np.ones(6),  lower=np.zeros(6), upper=10.)
        self.add_output('y_d', val=2.0 * np.ones(6), lower=0., upper=[12, 10, 10, 10, 10, 12])
        self.add_output('y_e', val=2.0 * np.ones((3, 2)), lower=np.zeros((3, 2)))


class TestAddVar(unittest.TestCase):

    def test_val(self):
        """Test declaring only default value."""

        p = om.Problem()
        p.model.add_subsystem('comp', CompAddWithDefault(), promotes=['*'])
        p.setup()

        assert_near_equal(p.get_val('x_a'), 1.)
        assert_near_equal(p.get_val('x_b'), 3.)
        assert_near_equal(p.get_val('x_c'), 3. * np.ones(2))
        assert_near_equal(p.get_val('x_d'), 3. * np.ones(2))
        assert_near_equal(p.get_val('x_e'), 3. * np.ones((2, 2)))
        assert_near_equal(p.get_val('y_a'), 1.)
        assert_near_equal(p.get_val('y_b'), 6.)
        assert_near_equal(p.get_val('y_c'), 6. * np.ones(3))
        assert_near_equal(p.get_val('y_d'), 6. * np.ones(3))
        assert_near_equal(p.get_val('y_e'), 6. * np.ones((3, 2)))

    def test_shape(self):
        """Test declaring only shape."""

        p = om.Problem()
        p.model.add_subsystem('comp', CompAddWithShape(), promotes=['*'])
        p.setup()

        assert_near_equal(p.get_val('x_a'), np.ones(2))
        assert_near_equal(p.get_val('x_b'), np.ones((2, 2)))
        assert_near_equal(p.get_val('x_c'), np.ones((2, 2)))
        assert_near_equal(p.get_val('y_a'), np.ones(3))
        assert_near_equal(p.get_val('y_b'), np.ones((3, 3)))
        assert_near_equal(p.get_val('y_c'), np.ones((3, 3)))

    def test_scalar_array(self):
        """Test declaring a scalar val with an array variable."""

        p = om.Problem()
        p.model.add_subsystem('comp', CompAddArrayWithScalar(), promotes=['*'])
        p.setup()

        assert_near_equal(p.get_val('x_a'), 2. * np.ones(6))
        assert_near_equal(p.get_val('x_b'), 2. * np.ones((3, 2)))
        assert_near_equal(p.get_val('y_a'), 3. * np.ones(6))
        assert_near_equal(p.get_val('y_b'), 3. * np.ones((3, 2)))

    def test_bounds(self):
        """Test declaring bounds."""

        p = om.Problem()
        p.model.add_subsystem('comp', CompAddWithBounds(), promotes=['*'])
        p.setup()

        assert_near_equal(p.get_val('y_a'), 2.)
        assert_near_equal(p.get_val('y_b'), 2.)
        assert_near_equal(p.get_val('y_c'), 2. * np.ones(6))
        assert_near_equal(p.get_val('y_d'), 2. * np.ones(6))
        assert_near_equal(p.get_val('y_e'), 2. * np.ones((3, 2)))


if __name__ == '__main__':
    unittest.main()
