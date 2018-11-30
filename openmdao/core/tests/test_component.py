"""Component unittests."""
from __future__ import division

import numpy as np
import unittest

from six.moves import range
from six import assertRaisesRegex

from openmdao.api import Problem, ExplicitComponent, Group, IndepVarComp
from openmdao.core.component import Component
from openmdao.test_suite.components.expl_comp_simple import TestExplCompSimple
from openmdao.test_suite.components.expl_comp_array import TestExplCompArray
from openmdao.test_suite.components.impl_comp_simple import TestImplCompSimple
from openmdao.test_suite.components.impl_comp_array import TestImplCompArray
from openmdao.utils.assert_utils import assert_rel_error


class TestExplicitComponent(unittest.TestCase):

    def test___init___simple(self):
        """Test a simple explicit component."""
        comp = TestExplCompSimple()
        prob = Problem(comp).setup(check=False)

        # check optional metadata (desc)
        self.assertEqual(
            comp._var_abs2meta['length']['desc'],
            'length of rectangle')
        self.assertEqual(
            comp._var_abs2meta['width']['desc'],
            'width of rectangle')
        self.assertEqual(
            comp._var_abs2meta['area']['desc'],
            'area of rectangle')

        prob['length'] = 3.
        prob['width'] = 2.
        prob.run_model()
        assert_rel_error(self, prob['area'], 6.)

    def test___init___array(self):
        """Test an explicit component with array inputs/outputs."""
        comp = TestExplCompArray(thickness=1.)
        prob = Problem(comp).setup(check=False)

        prob['lengths'] = 3.
        prob['widths'] = 2.
        prob.run_model()
        assert_rel_error(self, prob['total_volume'], 24.)

    def test_error_handling(self):
        """Test error handling when adding inputs/outputs."""
        comp = ExplicitComponent()

        msg = "Incompatible shape for '.*': Expected (.*) but got (.*)"

        with assertRaisesRegex(self, ValueError, msg):
            comp.add_output('arr', val=np.ones((2, 2)), shape=([2]))

        with assertRaisesRegex(self, ValueError, msg):
            comp.add_input('arr', val=np.ones((2, 2)), shape=([2]))

        msg = "Shape of indices does not match shape for '.*': Expected (.*) but got (.*)"

        with assertRaisesRegex(self, ValueError, msg):
            comp.add_input('arr', val=np.ones((2, 2)), src_indices=[0, 1])

        msg = ("The shape argument should be an int, tuple, or list "
               "but a '<(.*) 'numpy.ndarray'>' was given")
        with assertRaisesRegex(self, TypeError, msg):
            comp.add_output('arr', shape=np.array([2.]))

        with assertRaisesRegex(self, TypeError, msg):
            comp.add_input('arr', shape=np.array([2.]))

        msg = ("The shape argument should be an int, tuple, or list "
               "but a '<(.*) 'float'>' was given")
        with assertRaisesRegex(self, TypeError, msg):
            comp.add_output('arr', shape=2.)

        with assertRaisesRegex(self, TypeError, msg):
            comp.add_input('arr', shape=2.)

        # check that a numpy integer type is accepted for shape
        shapes = np.array([3], dtype=np.uint32)
        comp.add_output('aro', shape=shapes[0])
        comp.add_input('ari', shape=shapes[0])

        msg = "The name argument should be a string"
        name = 3

        with assertRaisesRegex(self, TypeError, msg):
            comp.add_input(name, val=np.ones((2, 2)))

        with assertRaisesRegex(self, TypeError, msg):
            comp.add_output(name, val=np.ones((2, 2)))

        msg = 'The val argument should be a float, list, tuple, ndarray or Iterable'
        val = Component

        with assertRaisesRegex(self, TypeError, msg):
            comp.add_input('x', val=val)

        with assertRaisesRegex(self, TypeError, msg):
            comp.add_output('x', val=val)

        msg = 'The src_indices argument should be an int, list, tuple, ndarray or Iterable'
        src = Component

        with assertRaisesRegex(self, TypeError, msg):
            comp.add_input('x', val=np.ones((2, 2)), src_indices=src)

        msg = 'The units argument should be a str or None'
        units = Component

        with assertRaisesRegex(self, TypeError, msg):
            comp.add_input('x', val=np.ones((2, 2)), units=units)

        with assertRaisesRegex(self, TypeError, msg):
            comp.add_output('x', val=np.ones((2, 2)), units=units)

        msg = 'The ref argument should be a float, list, tuple, ndarray or Iterable'
        val = Component

        with assertRaisesRegex(self, TypeError, msg):
            comp.add_output('x', val=5.0, ref=val)

        msg = 'The ref0 argument should be a float, list, tuple, ndarray or Iterable'
        val = Component

        with assertRaisesRegex(self, TypeError, msg):
            comp.add_output('x', val=5.0, ref0=val)

        msg = 'The res_ref argument should be a float, list, tuple, ndarray or Iterable'
        val = Component

        with assertRaisesRegex(self, TypeError, msg):
            comp.add_output('x', val=5.0, res_ref=val)

        msg = 'The res_units argument should be a str or None'
        units = Component

        with assertRaisesRegex(self, TypeError, msg):
            comp.add_output('x', val=5.0, res_units=val)

        # Test some forbidden names.

        msg = "'x.y' is not a valid input name."
        with assertRaisesRegex(self, NameError, msg):
            comp.add_input('x.y', val=5.0)

        msg = "'*' is not a valid input name."
        with assertRaisesRegex(self, NameError, msg):
            comp.add_input('*', val=5.0)

        msg = "'?' is not a valid input name."
        with assertRaisesRegex(self, NameError, msg):
            comp.add_input('?', val=5.0)

        msg = r"'\[' is not a valid input name."
        with assertRaisesRegex(self, NameError, msg):
            comp.add_input('[', val=5.0)

        msg = r"'\]' is not a valid input name."
        with assertRaisesRegex(self, NameError, msg):
            comp.add_input(']', val=5.0)

        msg = "'x.y' is not a valid output name."
        with assertRaisesRegex(self, NameError, msg):
            comp.add_output('x.y', val=5.0)

        msg = "'*' is not a valid output name."
        with assertRaisesRegex(self, NameError, msg):
            comp.add_output('*', val=5.0)

        msg = "'?' is not a valid output name."
        with assertRaisesRegex(self, NameError, msg):
            comp.add_output('?', val=5.0)

        msg = r"'\[' is not a valid output name."
        with assertRaisesRegex(self, NameError, msg):
            comp.add_output('[', val=5.0)

        msg = r"'\]' is not a valid output name."
        with assertRaisesRegex(self, NameError, msg):
            comp.add_output(']', val=5.0)

        # Stuff we allow.
        comp.add_input('a:b', val=5.0)
        comp.add_output('b:c', val=5.0)
        comp.add_input('x-y', val=5.0)
        comp.add_output('---', val=5.0)
        comp.add_output('-+=&$(;"<>@;^', val=5.0)

    def test_setup_bug1(self):
        # This tests a bug where, if you run setup more than once on a derived component class,
        # the list of var names continually gets prepended with the component global path.

        class NewBase(Component):
            def __init__(self, **kwargs):
                super(NewBase, self).__init__(**kwargs)

        class MyComp(NewBase):
            def __init__(self, **kwargs):
                super(MyComp, self).__init__(**kwargs)

            def setup(self):
                self.add_input('x', val=0.0)
                self.add_output('y', val=0.0)

        prob = Problem()
        model = prob.model = Group()
        comp = model.add_subsystem('comp', MyComp())

        prob.setup(check=False)
        self.assertEqual(comp._var_abs_names['input'], ['comp.x'])
        self.assertEqual(comp._var_abs_names['output'], ['comp.y'])

        prob.run_model()
        prob.setup(check=False)
        self.assertEqual(comp._var_abs_names['input'], ['comp.x'])
        self.assertEqual(comp._var_abs_names['output'], ['comp.y'])

    def test_add_input_output_dupes(self):

        class Comp(ExplicitComponent):
            def setup(self):
                self.add_input('x', val=3.0)
                self.add_input('x', val=3.0)
                self.add_output('y', val=3.0)

        prob = Problem()
        model = prob.model = Group()
        model.add_subsystem('px', IndepVarComp('x', val=3.0))
        model.add_subsystem('comp', Comp())

        model.connect('px.x', 'comp.x')

        msg = "Variable name 'x' already exists."
        with assertRaisesRegex(self, ValueError, msg):
            prob.setup(check=False)

        class Comp(ExplicitComponent):
            def setup(self):
                self.add_input('x', val=3.0)
                self.add_output('y', val=3.0)
                self.add_output('y', val=3.0)

        prob = Problem()
        model = prob.model = Group()
        model.add_subsystem('px', IndepVarComp('x', val=3.0))
        model.add_subsystem('comp', Comp())

        model.connect('px.x', 'comp.x')

        msg = "Variable name 'y' already exists."
        with assertRaisesRegex(self, ValueError, msg):
            prob.setup(check=False)

        class Comp(ExplicitComponent):
            def setup(self):
                self.add_input('x', val=3.0)
                self.add_output('x', val=3.0)
                self.add_output('y', val=3.0)

        prob = Problem()
        model = prob.model = Group()
        model.add_subsystem('px', IndepVarComp('x', val=3.0))
        model.add_subsystem('comp', Comp())

        model.connect('px.x', 'comp.x')

        msg = "Variable name 'x' already exists."
        with assertRaisesRegex(self, ValueError, msg):
            prob.setup(check=False)

        # Make sure we can reconfigure.

        class Comp(ExplicitComponent):
            def setup(self):
                self.add_input('x', val=3.0)
                self.add_output('y', val=3.0)

        prob = Problem()
        model = prob.model = Group()
        model.add_subsystem('px', IndepVarComp('x', val=3.0))
        model.add_subsystem('comp', Comp())

        model.connect('px.x', 'comp.x')

        prob.setup(check=False)

        # pretend we reconfigured
        prob.setup(check=False)


class TestImplicitComponent(unittest.TestCase):

    def test___init___simple(self):
        """Test a simple implicit component."""
        x = -0.5
        a = np.abs(np.exp(0.5 * x) / x)

        comp = TestImplCompSimple()
        prob = Problem(comp).setup(check=False)

        prob['a'] = a
        prob.run_model()
        assert_rel_error(self, prob['x'], x)

    def test___init___array(self):
        """Test an implicit component with array inputs/outputs."""
        comp = TestImplCompArray()
        prob = Problem(comp).setup(check=False)

        prob['rhs'] = np.ones(2)
        prob.run_model()
        assert_rel_error(self, prob['x'], np.ones(2))


class TestRangePartials(unittest.TestCase):

    def test_range_partials(self):
        class RangePartialsComp(ExplicitComponent):
            def __init__(self, size=4):
                super(RangePartialsComp, self).__init__()
                self.size = size

            def setup(self):
                # verify that both iterable and array types are valid
                # for val and src_indices arguments to add_input
                self.add_input('v1', val=range(self.size),
                                     src_indices=range(self.size))

                self.add_input('v2', val=2*np.ones(self.size),
                                     src_indices=np.array(range(self.size)))

                # verify that both iterable and array types are valid
                # for val, upper and lower arguments to add_output
                self.add_output('vSum', val=range(self.size),
                                        lower=np.zeros(self.size),
                                        upper=range(self.size))

                self.add_output('vProd', val=np.zeros(self.size),
                                         lower=range(self.size),
                                         upper=np.ones(self.size))

                # verify that both iterable and list types are valid
                # for rows and cols arguments to declare_partials
                rows = range(self.size)
                cols = list(range(self.size))
                self.declare_partials(of='vProd', wrt='v1',
                                      val=np.ones(self.size),
                                      rows=rows, cols=cols)

            def compute(self, inputs, outputs):
                outputs['vSum'] = inputs['v1'] + inputs['v2']
                outputs['vProd'] = inputs['v1'] * inputs['v2']

        comp = RangePartialsComp()

        prob = Problem(model=comp)
        prob.setup(check=False)
        prob.run_model()

        assert_rel_error(self, prob['vSum'], np.array([2., 3., 4., 5.]), 0.00001)
        assert_rel_error(self, prob['vProd'], np.array([0., 2., 4., 6.]), 0.00001)


if __name__ == '__main__':
    unittest.main()
