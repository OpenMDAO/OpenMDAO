"""Component unittests."""
import numpy as np
import unittest

from openmdao.api import Problem, ExplicitComponent, IndepVarComp
from openmdao.core.component import Component
from openmdao.test_suite.components.expl_comp_simple import TestExplCompSimple
from openmdao.test_suite.components.expl_comp_array import TestExplCompArray
from openmdao.test_suite.components.impl_comp_simple import TestImplCompSimple
from openmdao.test_suite.components.impl_comp_array import TestImplCompArray
from openmdao.utils.assert_utils import assert_near_equal, assert_warning
from openmdao.utils.om_warnings import OMDeprecationWarning


class TestExplicitComponent(unittest.TestCase):

    def test___init___simple(self):
        """Test a simple explicit component."""
        comp = TestExplCompSimple()

        prob = Problem()
        prob.model.add_subsystem('comp', comp, promotes=['*'])
        prob.setup()
        # check optional metadata (desc)
        self.assertEqual(comp._var_abs2meta['input']['comp.length']['desc'],
                         'length of rectangle')
        self.assertEqual(comp._var_abs2meta['input']['comp.width']['desc'],
                         'width of rectangle')
        self.assertEqual(comp._var_abs2meta['output']['comp.area']['desc'],
                         'area of rectangle')

        prob['length'] = 3.
        prob['width'] = 2.
        prob.run_model()
        assert_near_equal(prob['area'], 6.)

    def test___init___array(self):
        """Test an explicit component with array inputs/outputs."""
        comp = TestExplCompArray(thickness=1.)

        prob = Problem()
        prob.model.add_subsystem('comp', comp, promotes=['*'])
        prob.setup()

        prob['lengths'] = 3.
        prob['widths'] = 2.
        prob.run_model()
        assert_near_equal(prob['total_volume'], 24.)

    def test_error_handling(self):
        """Test error handling when adding inputs/outputs."""
        comp = ExplicitComponent()

        msg = "Incompatible shape for '.*': Expected (.*) but got (.*)"

        with self.assertRaisesRegex(ValueError, msg):
            comp.add_output('arr', val=np.ones((2, 2)), shape=([2]))

        with self.assertRaisesRegex(ValueError, msg):
            comp.add_input('arr', val=np.ones((2, 2)), shape=([2]))

        msg = ("The shape argument should be an int, tuple, or list "
               "but a '<(.*) 'numpy.ndarray'>' was given")
        with self.assertRaisesRegex(TypeError, msg):
            comp.add_output('arr', shape=np.array([2.]))

        with self.assertRaisesRegex(TypeError, msg):
            comp.add_input('arr', shape=np.array([2.]))

        msg = ("The shape argument should be an int, tuple, or list "
               "but a '<(.*) 'float'>' was given")
        with self.assertRaisesRegex(TypeError, msg):
            comp.add_output('arr', shape=2.)

        with self.assertRaisesRegex(TypeError, msg):
            comp.add_input('arr', shape=2.)

        # check that a numpy integer type is accepted for shape
        shapes = np.array([3], dtype=np.uint32)
        comp.add_output('aro', shape=shapes[0])
        comp.add_input('ari', shape=shapes[0])

        msg = 'The val argument should be a float, list, tuple, ndarray or Iterable'
        val = Component

        with self.assertRaisesRegex(TypeError, msg):
            comp.add_input('x', val=val)

        with self.assertRaisesRegex(TypeError, msg):
            comp.add_output('x', val=val)

        msg = 'The units argument should be a str or None'
        units = Component

        with self.assertRaisesRegex(TypeError, msg):
            comp.add_input('x', val=np.ones((2, 2)), units=units)

        with self.assertRaisesRegex(TypeError, msg):
            comp.add_output('x', val=np.ones((2, 2)), units=units)

        msg = 'The ref argument should be a float, list, tuple, ndarray or Iterable'
        val = Component

        with self.assertRaisesRegex(TypeError, msg):
            comp.add_output('x', val=5.0, ref=val)

        msg = 'The ref0 argument should be a float, list, tuple, ndarray or Iterable'
        val = Component

        with self.assertRaisesRegex(TypeError, msg):
            comp.add_output('x', val=5.0, ref0=val)

        msg = 'The res_ref argument should be a float, list, tuple, ndarray or Iterable'
        val = Component

        with self.assertRaisesRegex(TypeError, msg):
            comp.add_output('x', val=5.0, res_ref=val)

        msg = 'The res_units argument should be a str or None'
        units = Component

        with self.assertRaisesRegex(TypeError, msg):
            comp.add_output('x', val=5.0, res_units=val)

    def test_unit_simplify(self):
        comp = Component()
        comp.add_input('y', units='ft*ft/ft')
        comp.add_output('z', units='ft*ft/ft')

        self.assertEqual(comp._static_var_rel2meta['y']['units'], 'ft')
        self.assertEqual(comp._static_var_rel2meta['z']['units'], 'ft')

    def test_invalid_name(self):
        comp = ExplicitComponent()

        add_input_methods = [comp.add_input, comp.add_discrete_input]
        add_output_methods = [comp.add_output, comp.add_discrete_output]

        # Test some forbidden names.
        invalid_names = ['a.b', 'a*b', 'a?b', 'a!', '[a', 'b]']
        invalid_error = "<class ExplicitComponent>: '%s' is not a valid %s name."

        nostr_names = [3, None, object, object()]
        nostr_error = "<class ExplicitComponent>: The name argument should be a string."

        empty_in_error = "<class ExplicitComponent>: '' is not a valid input name."
        empty_out_error = "<class ExplicitComponent>: '' is not a valid output name."

        for func in add_input_methods:
            for name in invalid_names:
                with self.assertRaises(NameError) as cm:
                    func(name, val=5.0)
                self.assertEqual(str(cm.exception), invalid_error % (name, 'input'))

            for name in nostr_names:
                with self.assertRaises(TypeError) as cm:
                    func(name, val=5.0)
                self.assertEqual(str(cm.exception), nostr_error)

            with self.assertRaises(NameError) as cm:
                func('', val=5.0)
            self.assertEqual(str(cm.exception), empty_in_error)

        for func in add_output_methods:
            for name in invalid_names:
                with self.assertRaises(NameError) as cm:
                    func(name, val=5.0)
                self.assertEqual(str(cm.exception), invalid_error % (name, 'output'))

            for name in nostr_names:
                with self.assertRaises(TypeError) as cm:
                    func(name, val=5.0)
                self.assertEqual(str(cm.exception), nostr_error)

            with self.assertRaises(NameError) as cm:
                func('', val=5.0)
            self.assertEqual(str(cm.exception), empty_out_error)


        # Stuff we allow.
        comp.add_input('a:b', val=5.0)
        comp.add_output('b:c', val=5.0)
        comp.add_input('x-y', val=5.0)
        comp.add_output('---', val=5.0)
        comp.add_output('-+=&$(;"<>@;^', val=5.0)

    def test_setup_bug1(self):
        # This tests a bug where, if you run setup more than once on a derived component class,
        # the list of var names continually gets prepended with the component global path.

        class NewBase(ExplicitComponent):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)

        class MyComp(NewBase):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)

            def setup(self):
                self.add_input('x', val=0.0)
                self.add_output('y', val=0.0)

        prob = Problem()
        model = prob.model
        comp = model.add_subsystem('comp', MyComp())

        prob.setup()
        self.assertEqual(list(comp._var_abs2meta['input']), ['comp.x'])
        self.assertEqual(list(comp._var_abs2meta['output']), ['comp.y'])

        prob.run_model()
        prob.setup()
        self.assertEqual(list(comp._var_abs2meta['input']), ['comp.x'])
        self.assertEqual(list(comp._var_abs2meta['output']), ['comp.y'])

    def test_add_input_output_dupes(self):

        class Comp(ExplicitComponent):
            def setup(self):
                self.add_input('x', val=3.0)
                self.add_input('x', val=3.0)
                self.add_output('y', val=3.0)

        prob = Problem()
        model = prob.model
        model.add_subsystem('px', IndepVarComp('x', val=3.0))
        model.add_subsystem('comp', Comp())

        model.connect('px.x', 'comp.x')

        msg = "Variable name 'x' already exists."
        with self.assertRaisesRegex(ValueError, msg):
            prob.setup()

        class Comp(ExplicitComponent):
            def setup(self):
                self.add_input('x', val=3.0)
                self.add_output('y', val=3.0)
                self.add_output('y', val=3.0)

        prob = Problem()
        model = prob.model
        model.add_subsystem('px', IndepVarComp('x', val=3.0))
        model.add_subsystem('comp', Comp())

        model.connect('px.x', 'comp.x')

        msg = "Variable name 'y' already exists."
        with self.assertRaisesRegex(ValueError, msg):
            prob.setup()

        class Comp(ExplicitComponent):
            def setup(self):
                self.add_input('x', val=3.0)
                self.add_output('x', val=3.0)
                self.add_output('y', val=3.0)

        prob = Problem()
        model = prob.model
        model.add_subsystem('px', IndepVarComp('x', val=3.0))
        model.add_subsystem('comp', Comp())

        model.connect('px.x', 'comp.x')

        msg = "Variable name 'x' already exists."
        with self.assertRaisesRegex(ValueError, msg):
            prob.setup()

        # Make sure we can reconfigure.

        class Comp(ExplicitComponent):
            def setup(self):
                self.add_input('x', val=3.0)
                self.add_output('y', val=3.0)

        prob = Problem()
        model = prob.model
        model.add_subsystem('px', IndepVarComp('x', val=3.0))
        model.add_subsystem('comp', Comp())

        model.connect('px.x', 'comp.x')

        prob.setup()

        # pretend we reconfigured
        prob.setup()

    def test_zero_partial(self):
        class Comp(ExplicitComponent):
            def setup(self):
                self.add_input('x', val=3.0)
                self.add_output('y', val=3.0)

                self.declare_partials(of='y', wrt='x', val=0.0)

        prob = Problem()
        prob.model.add_subsystem('comp', Comp())

        msg = "'comp' <class Comp>: d(y)/d(x): Partial was declared to be exactly zero. " \
              "This is inefficient and the declaration should be removed. In a future " \
              "version of OpenMDAO this behavior will raise an error."

        with assert_warning(OMDeprecationWarning, msg):
            prob.setup()
    
    def test_setup_residuals_error(self):
        class Comp(ExplicitComponent):
            def setup(self):
                self.add_input('x', val=3.0)
                self.add_output('y', val=3.0)

            def setup_residuals(self):
                # Overriding setup_residuals should raise an error.
                pass

        prob = Problem()
        prob.model.add_subsystem('comp', Comp())

        msg = ("'comp' <class Comp>: Class overrides setup_residuals "
               "but is an ExplicitComponent. setup_residuals may only "
               "be overridden by ImplicitComponents.")

        prob.setup()

        with self.assertRaises(RuntimeError) as e:
            prob.final_setup()
        
        self.assertEqual(str(e.exception), msg)

    def test_add_residual_error(self):
        class Comp(ExplicitComponent):
            def setup(self):
                self.add_input('x', val=3.0)
                self.add_output('y', val=3.0)
                # Invalid to add a residual to an explicit component
                self.add_residual('resid_x', shape=(1,))

        prob = Problem()
        prob.model.add_subsystem('comp', Comp())

        msg = ("'Comp' object has no attribute 'add_residual'")

        with self.assertRaises(AttributeError) as e:
            prob.setup()
        
        self.assertEqual(str(e.exception), msg)

class TestImplicitComponent(unittest.TestCase):

    def test___init___simple(self):
        """Test a simple implicit component."""
        x = -0.5
        a = np.abs(np.exp(0.5 * x) / x)

        comp = TestImplCompSimple()

        prob = Problem()
        prob.model.add_subsystem('comp', comp, promotes=['*'])
        prob.setup()

        prob['a'] = a
        prob.run_model()
        assert_near_equal(prob['x'], x)

    def test___init___array(self):
        """Test an implicit component with array inputs/outputs."""
        comp = TestImplCompArray()

        prob = Problem()
        prob.model.add_subsystem('comp', comp, promotes=['*'])
        prob.setup()

        prob['rhs'] = np.ones(2)
        prob.run_model()
        assert_near_equal(prob['x'], np.ones(2))


class TestRangePartials(unittest.TestCase):

    def test_range_partials(self):
        class RangePartialsComp(ExplicitComponent):
            def __init__(self, size=4):
                super().__init__()
                self.size = size

            def setup(self):
                self.add_input('v1', shape=self.size)
                self.add_input('v2', shape=self.size)

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


        size = 4

        prob = Problem()

        indep = prob.model.add_subsystem('indep', IndepVarComp())
        indep.add_output('v1', val=range(size))
        indep.add_output('v2', val=2*np.ones(size))

        prob.model.add_subsystem('comp', RangePartialsComp())

        # verify that both iterable and array types are valid
        # for val and src_indices arguments to connect
        prob.model.connect('indep.v1', 'comp.v1', src_indices=range(size))
        prob.model.connect('indep.v2', 'comp.v2', src_indices=np.array(range(size)))

        prob.setup()
        prob.run_model()

        assert_near_equal(prob['comp.vSum'], np.array([2., 3., 4., 5.]), 0.00001)
        assert_near_equal(prob['comp.vProd'], np.array([0., 2., 4., 6.]), 0.00001)


if __name__ == '__main__':
    unittest.main()
