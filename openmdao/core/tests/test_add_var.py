"""Acceptance and developer tests for add_input and add_output."""
import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal


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


class CompAddWithIndices(om.ExplicitComponent):
    """Component for tests for declaring only indices."""

    def setup(self):
        self.add_input('x_a', src_indices=0)
        self.add_input('x_b', src_indices=(0, 1))
        self.add_input('x_c', src_indices=[0, 1])
        self.add_input('x_d', src_indices=np.arange(6))
        self.add_input('x_e', src_indices=np.arange(6).reshape((3, 2)), shape=(3,2))

        self.add_output('y')


class CompAddWithShapeAndIndices(om.ExplicitComponent):
    """Component for tests for declaring shape and array indices."""

    def setup(self):
        self.add_input('x_a', shape=2, src_indices=(0,1))
        self.add_input('x_b', shape=(2,), src_indices=(0,1))
        self.add_input('x_c', shape=(2, 2), src_indices=np.arange(4).reshape((2, 2)))
        self.add_input('x_d', shape=[2, 2], src_indices=np.arange(4).reshape((2, 2)))


class CompAddArrayWithScalar(om.ExplicitComponent):
    """Component for tests for declaring a scalar val with an array variable."""

    def setup(self):
        self.add_input('x_a', val=2.0, shape=(6))
        self.add_input('x_b', val=2.0, shape=(3, 2))
        self.add_input('x_c', val=2.0, src_indices=np.arange(6))
        self.add_input('x_d', val=2.0, src_indices=np.arange(6).reshape((3,2)), shape=(3,2))

        self.add_output('y_a', val=3.0, shape=(6))
        self.add_output('y_b', val=3.0, shape=(3, 2))


class CompAddWithArrayIndices(om.ExplicitComponent):
    """Component for tests for declaring with array val and array indices."""

    def setup(self):
        self.add_input('x_a', val=2.0 * np.ones(6), src_indices=np.arange(6))
        self.add_input('x_b', val=2.0 * np.ones((3, 2)), src_indices=np.arange(6).reshape((3, 2)))

        self.add_output('y')


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
        import openmdao.api as om
        from openmdao.core.tests.test_add_var import CompAddWithDefault

        p = om.Problem(model=CompAddWithDefault())
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
        import openmdao.api as om
        from openmdao.core.tests.test_add_var import CompAddWithShape

        p = om.Problem(model=CompAddWithShape())
        p.setup()

        assert_near_equal(p.get_val('x_a'), np.ones(2))
        assert_near_equal(p.get_val('x_b'), np.ones((2, 2)))
        assert_near_equal(p.get_val('x_c'), np.ones((2, 2)))
        assert_near_equal(p.get_val('y_a'), np.ones(3))
        assert_near_equal(p.get_val('y_b'), np.ones((3, 3)))
        assert_near_equal(p.get_val('y_c'), np.ones((3, 3)))

    def test_indices(self):
        """Test declaring only indices."""
        import openmdao.api as om
        from openmdao.core.tests.test_add_var import CompAddWithIndices

        p = om.Problem(model=CompAddWithIndices())
        p.setup()

        assert_near_equal(p.get_val('x_a'), 1.)
        assert_near_equal(p.get_val('x_b'), np.ones(2))
        assert_near_equal(p.get_val('x_c'), np.ones(2))
        assert_near_equal(p.get_val('x_d'), np.ones(6))
        assert_near_equal(p.get_val('x_e'), np.ones((3,2)))

    def test_shape_and_indices(self):
        """Test declaring shape and indices."""
        p = om.Problem(model=CompAddWithShapeAndIndices())
        p.setup()

        assert_near_equal(p.get_val('x_a'), np.ones(2))
        assert_near_equal(p.get_val('x_b'), np.ones(2))
        assert_near_equal(p.get_val('x_c'), np.ones((2,2)))
        assert_near_equal(p.get_val('x_d'), np.ones((2,2)))

    def test_scalar_array(self):
        """Test declaring a scalar val with an array variable."""
        import openmdao.api as om
        from openmdao.core.tests.test_add_var import CompAddArrayWithScalar

        p = om.Problem(model=CompAddArrayWithScalar())
        p.setup()

        assert_near_equal(p.get_val('x_a'), 2. * np.ones(6))
        assert_near_equal(p.get_val('x_b'), 2. * np.ones((3, 2)))
        assert_near_equal(p.get_val('x_c'), 2. * np.ones(6))
        assert_near_equal(p.get_val('x_d'), 2. * np.ones((3, 2)))
        assert_near_equal(p.get_val('y_a'), 3. * np.ones(6))
        assert_near_equal(p.get_val('y_b'), 3. * np.ones((3, 2)))

    def test_array_indices(self):
        """Test declaring with array val and array indices."""
        import openmdao.api as om
        from openmdao.core.tests.test_add_var import CompAddWithArrayIndices

        p = om.Problem(model=CompAddWithArrayIndices())
        p.setup()

        assert_near_equal(p.get_val('x_a'), 2. * np.ones(6))
        assert_near_equal(p.get_val('x_b'), 2. * np.ones((3, 2)))

    def test_bounds(self):
        """Test declaring bounds."""
        import openmdao.api as om
        from openmdao.core.tests.test_add_var import CompAddWithBounds

        p = om.Problem(model=CompAddWithBounds())
        p.setup()

        assert_near_equal(p.get_val('y_a'), 2.)
        assert_near_equal(p.get_val('y_b'), 2.)
        assert_near_equal(p.get_val('y_c'), 2. * np.ones(6))
        assert_near_equal(p.get_val('y_d'), 2. * np.ones(6))
        assert_near_equal(p.get_val('y_e'), 2. * np.ones((3, 2)))


if __name__ == '__main__':
    unittest.main()
