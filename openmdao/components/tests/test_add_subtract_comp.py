import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials


class TestAddSubtractCompScalars(unittest.TestCase):

    def setUp(self):
        self.nn = 1
        self.p = om.Problem()
        ivc = om.IndepVarComp()
        ivc.add_output(name='a', shape=(self.nn,))
        ivc.add_output(name='b', shape=(self.nn,))

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['a', 'b'])

        adder=self.p.model.add_subsystem(name='add_subtract_comp',
                                   subsys=om.AddSubtractComp())
        adder.add_equation('adder_output',['input_a','input_b'])

        self.p.model.connect('a', 'add_subtract_comp.input_a')
        self.p.model.connect('b', 'add_subtract_comp.input_b')

        self.p.setup()

        self.p['a'] = np.random.rand(self.nn,)
        self.p['b'] = np.random.rand(self.nn,)

        self.p.run_model()

    def test_results(self):
        a = self.p['a']
        b = self.p['b']
        out = self.p['add_subtract_comp.adder_output']
        expected = a + b
        assert_near_equal(out, expected,1e-16)

    def test_partials(self):
        partials = self.p.check_partials(method='fd', out_stream=None)
        assert_check_partials(partials)


class TestAddSubtractCompNx1(unittest.TestCase):

    def setUp(self):
        self.nn = 5

        self.p = om.Problem()

        ivc = om.IndepVarComp()
        ivc.add_output(name='a', shape=(self.nn,))
        ivc.add_output(name='b', shape=(self.nn,))

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['a', 'b'])

        adder=self.p.model.add_subsystem(name='add_subtract_comp',
                                   subsys=om.AddSubtractComp())
        adder.add_equation('adder_output',['input_a','input_b'],vec_size=self.nn)

        self.p.model.connect('a', 'add_subtract_comp.input_a')
        self.p.model.connect('b', 'add_subtract_comp.input_b')

        self.p.setup()

        self.p['a'] = np.random.rand(self.nn,)
        self.p['b'] = np.random.rand(self.nn,)

        self.p.run_model()

    def test_results(self):
        a = self.p['a']
        b = self.p['b']
        out = self.p['add_subtract_comp.adder_output']
        expected = a + b
        assert_near_equal(out, expected,1e-16)

    def test_partials(self):
        partials = self.p.check_partials(method='fd', out_stream=None)
        assert_check_partials(partials)


class TestAddSubtractCompNx3(unittest.TestCase):

    def setUp(self):
        self.nn = 5

        self.p = om.Problem()

        ivc = om.IndepVarComp()
        ivc.add_output(name='a', shape=(self.nn, 3))
        ivc.add_output(name='b', shape=(self.nn, 3))

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['a', 'b'])

        adder=self.p.model.add_subsystem(name='add_subtract_comp',
                                   subsys=om.AddSubtractComp())
        adder.add_equation('adder_output',['input_a','input_b'],vec_size=self.nn,length=3)

        self.p.model.connect('a', 'add_subtract_comp.input_a')
        self.p.model.connect('b', 'add_subtract_comp.input_b')

        self.p.setup()

        self.p['a'] = np.random.rand(self.nn, 3)
        self.p['b'] = np.random.rand(self.nn, 3)

        self.p.run_model()

    def test_results(self):
        a = self.p['a']
        b = self.p['b']
        out = self.p['add_subtract_comp.adder_output']
        expected = a + b
        assert_near_equal(out, expected,1e-16)

    def test_partials(self):
        partials = self.p.check_partials(method='fd', out_stream=None)
        assert_check_partials(partials)


class TestAddSubtractMultipleInputs(unittest.TestCase):

    def setUp(self):
        self.nn = 5

        self.p = om.Problem()

        ivc = om.IndepVarComp()
        ivc.add_output(name='a', shape=(self.nn, 3))
        ivc.add_output(name='b', shape=(self.nn, 3))
        ivc.add_output(name='c', shape=(self.nn, 3))

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['a', 'b','c'])

        adder=self.p.model.add_subsystem(name='add_subtract_comp',
                                   subsys=om.AddSubtractComp())
        adder.add_equation('adder_output',['input_a','input_b','input_c'],vec_size=self.nn,length=3)

        self.p.model.connect('a', 'add_subtract_comp.input_a')
        self.p.model.connect('b', 'add_subtract_comp.input_b')
        self.p.model.connect('c', 'add_subtract_comp.input_c')

        self.p.setup()

        self.p['a'] = np.random.rand(self.nn, 3)
        self.p['b'] = np.random.rand(self.nn, 3)
        self.p['c'] = np.random.rand(self.nn, 3)

        self.p.run_model()

    def test_results(self):
        a = self.p['a']
        b = self.p['b']
        c = self.p['c']
        out = self.p['add_subtract_comp.adder_output']
        expected = a + b + c
        assert_near_equal(out, expected,1e-16)

    def test_partials(self):
        partials = self.p.check_partials(method='fd', out_stream=None)
        assert_check_partials(partials)


class TestAddSubtractScalingFactors(unittest.TestCase):

    def setUp(self):
        self.nn = 5

        self.p = om.Problem()

        ivc = om.IndepVarComp()
        ivc.add_output(name='a', shape=(self.nn, 3))
        ivc.add_output(name='b', shape=(self.nn, 3))
        ivc.add_output(name='c', shape=(self.nn, 3))

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['a', 'b','c'])

        adder=self.p.model.add_subsystem(name='add_subtract_comp',
                                   subsys=om.AddSubtractComp())
        adder.add_equation('adder_output',['input_a','input_b','input_c'],vec_size=self.nn,length=3,scaling_factors=[2.,1.,-1])

        self.p.model.connect('a', 'add_subtract_comp.input_a')
        self.p.model.connect('b', 'add_subtract_comp.input_b')
        self.p.model.connect('c', 'add_subtract_comp.input_c')

        self.p.setup()

        self.p['a'] = np.random.rand(self.nn, 3)
        self.p['b'] = np.random.rand(self.nn, 3)
        self.p['c'] = np.random.rand(self.nn, 3)

        self.p.run_model()

    def test_results(self):
        a = self.p['a']
        b = self.p['b']
        c = self.p['c']
        out = self.p['add_subtract_comp.adder_output']
        expected = 2*a + b - c
        assert_near_equal(out, expected,1e-16)

    def test_partials(self):
        partials = self.p.check_partials(method='fd', out_stream=None)
        assert_check_partials(partials)


class TestAddSubtractUnits(unittest.TestCase):

    def setUp(self):
        self.nn = 5

        self.p = om.Problem()

        ivc = om.IndepVarComp()
        ivc.add_output(name='a', shape=(self.nn, 3),units='ft')
        ivc.add_output(name='b', shape=(self.nn, 3),units='m')
        ivc.add_output(name='c', shape=(self.nn, 3),units='m')

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['a', 'b','c'])

        adder=self.p.model.add_subsystem(name='add_subtract_comp',
                                   subsys=om.AddSubtractComp())
        adder.add_equation('adder_output',['input_a','input_b','input_c'],vec_size=self.nn,length=3,units='ft')

        self.p.model.connect('a', 'add_subtract_comp.input_a')
        self.p.model.connect('b', 'add_subtract_comp.input_b')
        self.p.model.connect('c', 'add_subtract_comp.input_c')

        self.p.setup()

        self.p['a'] = np.random.rand(self.nn, 3)
        self.p['b'] = np.random.rand(self.nn, 3)
        self.p['c'] = np.random.rand(self.nn, 3)

        self.p.run_model()

    def test_results(self):
        a = self.p['a']
        b = self.p['b']
        c = self.p['c']
        out = self.p['add_subtract_comp.adder_output']
        m_to_ft = 3.280839895
        expected = a + b*m_to_ft + c*m_to_ft
        assert_near_equal(out, expected,1e-8)

    def test_partials(self):
        partials = self.p.check_partials(method='fd', out_stream=None)
        assert_check_partials(partials)


class TestAddSubtractInit(unittest.TestCase):

    def setUp(self):
        self.nn = 5

        self.p = om.Problem()

        ivc = om.IndepVarComp()
        ivc.add_output(name='a', shape=(self.nn, 3), units='ft')
        ivc.add_output(name='b', shape=(self.nn, 3), units='m')
        ivc.add_output(name='c', shape=(self.nn, 3), units='m')

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['a', 'b','c'])

        # verify proper handling of constructor args
        adder = om.AddSubtractComp(output_name='adder_output',
                                   input_names=['input_a', 'input_b', 'input_c'],
                                   vec_size=self.nn, length=3,
                                   scaling_factors=[2., 1., -1],
                                   units='ft')

        self.p.model.add_subsystem(name='add_subtract_comp', subsys=adder)

        self.p.model.connect('a', 'add_subtract_comp.input_a')
        self.p.model.connect('b', 'add_subtract_comp.input_b')
        self.p.model.connect('c', 'add_subtract_comp.input_c')

        self.p.setup()

        self.p['a'] = np.random.rand(self.nn, 3)
        self.p['b'] = np.random.rand(self.nn, 3)
        self.p['c'] = np.random.rand(self.nn, 3)

        self.p.run_model()

    def test_results(self):
        a = self.p['a']
        b = self.p['b']
        c = self.p['c']
        out = self.p['add_subtract_comp.adder_output']
        m_to_ft = 3.280839895
        expected = 2*a + b*m_to_ft - c*m_to_ft
        assert_near_equal(out, expected, 1e-8)

    def test_partials(self):
        partials = self.p.check_partials(method='fd', out_stream=None)
        assert_check_partials(partials)


class TestForExceptions(unittest.TestCase):

    def test_for_bad_scale_factors(self):
        self.nn = 5
        self.p = om.Problem()

        ivc = om.IndepVarComp()
        ivc.add_output(name='a', shape=(self.nn, 3))
        ivc.add_output(name='b', shape=(self.nn, 3))
        ivc.add_output(name='c', shape=(self.nn, 3))

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['a', 'b','c'])

        adder=self.p.model.add_subsystem(name='add_subtract_comp',
                                   subsys=om.AddSubtractComp())

        with self.assertRaises(ValueError) as err:
            adder.add_equation('adder_output', ['input_a', 'input_b', 'input_c'], vec_size=self.nn,
                               length=3, scaling_factors=[1, -1])

        expected_msg = "'add_subtract_comp' <class AddSubtractComp>: Scaling factors list needs to be " \
                       "same length as input names"

        self.assertEqual(str(err.exception), expected_msg)


    def test_for_bad_input_set(self):
        self.nn = 5
        self.p = om.Problem()

        ivc = om.IndepVarComp()
        ivc.add_output(name='a', shape=(self.nn, 3))
        ivc.add_output(name='b', shape=(self.nn, 3))
        ivc.add_output(name='c', shape=(self.nn, 3))

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['a', 'b','c'])

        adder=self.p.model.add_subsystem(name='add_subtract_comp',
                                   subsys=om.AddSubtractComp())

        with self.assertRaises(ValueError) as err:
            adder.add_equation('adder_output', ['input_a',], vec_size=self.nn,
                               length=3, scaling_factors=[1, -1])

        expected_msg = "'add_subtract_comp' <class AddSubtractComp>: must specify more than one input " \
                       "name for an equation, but only one given"

        self.assertEqual(str(err.exception), expected_msg)

        with self.assertRaises(ValueError) as err:
            adder.add_equation('adder_output', 'input_a', vec_size=self.nn,
                               length=3, scaling_factors=[1, -1])

        expected_msg = "'add_subtract_comp' <class AddSubtractComp>: must specify more than one input " \
                       "name for an equation, but only one given"

        self.assertEqual(str(err.exception), expected_msg)


class TestAddSubtractCompTags(unittest.TestCase):

    def setUp(self):
        self.nn = 1
        self.p = om.Problem()
        ivc = om.IndepVarComp()
        ivc.add_output(name='a', shape=(self.nn,))
        ivc.add_output(name='b', shape=(self.nn,))

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['a', 'b'])

        adder=self.p.model.add_subsystem(name='add_subtract_comp', subsys=om.AddSubtractComp())
        adder.add_equation('adder_output', ['input_a','input_b'], tags={'foo'})
        adder.add_equation('adder_output2', ['input_a','input_a'], tags={'bar'})

        self.p.model.connect('a', 'add_subtract_comp.input_a')
        self.p.model.connect('b', 'add_subtract_comp.input_b')

        self.p.setup()

        self.p['a'] = np.random.rand(self.nn,)
        self.p['b'] = np.random.rand(self.nn,)

        self.p.run_model()

    def test_results(self):
        a = self.p['a']
        b = self.p['b']

        foo_outputs = self.p.model.list_outputs(tags={'foo'}, out_stream=None)
        bar_outputs = self.p.model.list_outputs(tags={'bar'}, out_stream=None)

        self.assertEqual(len(foo_outputs), 1)
        self.assertEqual(len(bar_outputs), 1)

        assert_near_equal(foo_outputs[0][1]['value'], a + b)
        assert_near_equal(bar_outputs[0][1]['value'], a + a)


class TestFeature(unittest.TestCase):

    def test(self):
        """
        A simple example to compute the resultant force on an aircraft and demonstrate the AddSubtract component
        """
        import numpy as np

        import openmdao.api as om

        n = 3

        p = om.Problem()
        model = p.model

        # Construct an adder/subtracter here. create a relationship through the add_equation method
        adder = om.AddSubtractComp()
        adder.add_equation('total_force', input_names=['thrust', 'drag', 'lift', 'weight'],
                           vec_size=n, length=2, scaling_factors=[1, -1, 1, -1], units='kN')
        # Note the scaling factors. we assume all forces are positive sign upstream

        # The vector represents forces at 3 time points (rows) in 2 dimensional plane (cols)
        p.model.add_subsystem(name='totalforcecomp', subsys=adder,
                              promotes_inputs=['thrust', 'drag', 'lift', 'weight'])

        p.setup()

        # Set thrust to exceed drag, weight to equal lift for this scenario
        p['thrust'][:, 0] = [500, 600, 700]
        p['drag'][:, 0] = [400, 400, 400]
        p['weight'][:, 1] = [1000, 1001, 1002]
        p['lift'][:, 1] = [1000, 1000, 1000]

        p.run_model()

        # Verify the results
        expected_i = np.array([[100, 200, 300], [0, -1, -2]]).T
        assert_near_equal(p.get_val('totalforcecomp.total_force', units='kN'), expected_i)


if __name__ == '__main__':
    unittest.main()
