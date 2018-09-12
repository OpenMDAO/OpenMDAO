"""
Unit tests for the unstructured metamodel component.
"""
import numpy as np
import unittest

from openmdao.api import Group, Problem, MetaModelUnStructuredComp, IndepVarComp, ResponseSurface, \
    FloatKrigingSurrogate, KrigingSurrogate, ScipyOptimizeDriver
from openmdao.utils.assert_utils import assert_rel_error
from openmdao.utils.logger_utils import TestLogger

class MetaModelTestCase(unittest.TestCase):

    def test_sin_metamodel(self):
        # create a MetaModelUnStructuredComp for sine and add it to a Problem
        sin_mm = MetaModelUnStructuredComp()
        sin_mm.add_input('x', 0.)
        sin_mm.add_output('f_x', 0.)

        prob = Problem()
        prob.model.add_subsystem('sin_mm', sin_mm)

        # check that missing surrogate is detected in check_config
        testlogger = TestLogger()
        prob.setup(check=True, logger=testlogger)

        # Conclude setup but don't run model.
        prob.final_setup()

        msg = ("No default surrogate model is defined and the "
               "following outputs do not have a surrogate model:\n"
               "['f_x']\n"
               "Either specify a default_surrogate, or specify a "
               "surrogate model for all outputs.")
        self.assertEqual(len(testlogger.get('error')), 1)
        self.assertTrue(msg in testlogger.get('error')[0])

        # check that output with no specified surrogate gets the default
        sin_mm.options['default_surrogate'] = FloatKrigingSurrogate()
        prob.setup(check=False)
        surrogate = sin_mm._metadata('f_x').get('surrogate')
        self.assertTrue(isinstance(surrogate, FloatKrigingSurrogate),
                        'sin_mm.f_x should get the default surrogate')

        # check error message when no training data is provided
        with self.assertRaises(RuntimeError) as cm:
            prob.run_model()

        msg = ("MetaModelUnStructuredComp: The following training data sets must be "
               "provided as options for sin_mm: ['train:x', 'train:f_x']")
        self.assertEqual(str(cm.exception), msg)

        # train the surrogate and check predicted value
        sin_mm.options['train:x'] = np.linspace(0,10,20)
        sin_mm.options['train:f_x'] = .5*np.sin(sin_mm.options['train:x'])

        prob['sin_mm.x'] = 2.1

        prob.run_model()

        assert_rel_error(self, prob['sin_mm.f_x'], .5*np.sin(prob['sin_mm.x']), 1e-4)

    def test_error_no_surrogate(self):
        # Seems like the error message from above should also be present and readable even if the
        # user chooses to skip checking the model.
        sin_mm = MetaModelUnStructuredComp()
        sin_mm.add_input('x', 0.)
        sin_mm.add_output('f_x', 0.)

        prob = Problem()
        prob.model.add_subsystem('sin_mm', sin_mm)

        prob.setup(check=False)

        sin_mm.options['train:x'] = np.linspace(0,10,20)
        sin_mm.options['train:f_x'] = .5*np.sin(sin_mm.options['train:x'])

        with self.assertRaises(RuntimeError) as cm:
            prob.run_model()

        msg = ("Metamodel 'sin_mm': No surrogate specified for output 'f_x'")
        self.assertEqual(str(cm.exception), msg)

    def test_sin_metamodel_preset_data(self):
        # preset training data
        x = np.linspace(0,10,200)
        f_x = .5*np.sin(x)

        # create a MetaModelUnStructuredComp for Sin and add it to a Problem
        sin_mm = MetaModelUnStructuredComp()
        sin_mm.add_input('x', 0., training_data=np.linspace(0,10,200))
        sin_mm.add_output('f_x', 0., training_data=f_x)

        prob = Problem()
        prob.model.add_subsystem('sin_mm', sin_mm)

        # check that missing surrogate is detected in check_setup
        testlogger = TestLogger()
        prob.setup(check=True, logger=testlogger)

        # Conclude setup but don't run model.
        prob.final_setup()

        msg = ("No default surrogate model is defined and the "
               "following outputs do not have a surrogate model:\n"
               "['f_x']\n"
               "Either specify a default_surrogate, or specify a "
               "surrogate model for all outputs.")
        self.assertEqual(len(testlogger.get('error')), 1)
        self.assertTrue(msg in testlogger.get('error')[0])

        # check that output with no specified surrogate gets the default
        sin_mm.options['default_surrogate'] = FloatKrigingSurrogate()
        prob.setup(check=False)

        surrogate = sin_mm._metadata('f_x').get('surrogate')
        self.assertTrue(isinstance(surrogate, FloatKrigingSurrogate),
                        'sin_mm.f_x should get the default surrogate')

        prob['sin_mm.x'] = 2.22

        prob.run_model()

        assert_rel_error(self, prob['sin_mm.f_x'], .5*np.sin(prob['sin_mm.x']), 1e-4)

    def test_sin_metamodel_rmse(self):
        # create MetaModelUnStructuredComp with Kriging, using the rmse option
        sin_mm = MetaModelUnStructuredComp()
        sin_mm.add_input('x', 0.)
        sin_mm.add_output('f_x', 0.)

        sin_mm.options['default_surrogate'] = KrigingSurrogate(eval_rmse=True)

        # add it to a Problem
        prob = Problem()
        prob.model.add_subsystem('sin_mm', sin_mm)
        prob.setup(check=False)

        # train the surrogate and check predicted value
        sin_mm.options['train:x'] = np.linspace(0,10,20)
        sin_mm.options['train:f_x'] = np.sin(sin_mm.options['train:x'])

        prob['sin_mm.x'] = 2.1

        prob.run_model()

        assert_rel_error(self, prob['sin_mm.f_x'], np.sin(2.1), 1e-4) # mean
        self.assertTrue(self, sin_mm._metadata('f_x')['rmse'] < 1e-5) # std deviation

    def test_basics(self):
        # create a metamodel component
        mm = MetaModelUnStructuredComp()

        mm.add_input('x1', 0.)
        mm.add_input('x2', 0.)

        mm.add_output('y1', 0.)
        mm.add_output('y2', 0., surrogate=FloatKrigingSurrogate())

        mm.options['default_surrogate'] = ResponseSurface()

        # add metamodel to a problem
        prob = Problem(model=Group())
        prob.model.add_subsystem('mm', mm)
        prob.setup(check=False)

        # check that surrogates were properly assigned
        surrogate = mm._metadata('y1').get('surrogate')
        self.assertTrue(isinstance(surrogate, ResponseSurface))

        surrogate = mm._metadata('y2').get('surrogate')
        self.assertTrue(isinstance(surrogate, FloatKrigingSurrogate))

        # populate training data
        mm.options['train:x1'] = [1.0, 2.0, 3.0]
        mm.options['train:x2'] = [1.0, 3.0, 4.0]
        mm.options['train:y1'] = [3.0, 2.0, 1.0]
        mm.options['train:y2'] = [1.0, 4.0, 7.0]

        # run problem for provided data point and check prediction
        prob['mm.x1'] = 2.0
        prob['mm.x2'] = 3.0

        self.assertTrue(mm.train)   # training will occur before 1st run
        prob.run_model()

        assert_rel_error(self, prob['mm.y1'], 2.0, .00001)
        assert_rel_error(self, prob['mm.y2'], 4.0, .00001)

        # run problem for interpolated data point and check prediction
        prob['mm.x1'] = 2.5
        prob['mm.x2'] = 3.5

        self.assertFalse(mm.train)  # training will not occur before 2nd run
        prob.run_model()

        assert_rel_error(self, prob['mm.y1'], 1.5934, .001)

        # change default surrogate, re-setup and check that metamodel re-trains
        mm.options['default_surrogate'] = FloatKrigingSurrogate()
        prob.setup(check=False)

        surrogate = mm._metadata('y1').get('surrogate')
        self.assertTrue(isinstance(surrogate, FloatKrigingSurrogate))

        self.assertTrue(mm.train)  # training will occur after re-setup

    def test_vector_inputs(self):
        mm = MetaModelUnStructuredComp()
        mm.add_input('x', np.zeros(4))
        mm.add_output('y1', 0.)
        mm.add_output('y2', 0.)

        mm.options['default_surrogate'] = FloatKrigingSurrogate()

        prob = Problem()
        prob.model.add_subsystem('mm', mm)
        prob.setup(check=False)

        mm.options['train:x'] = [
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 1.0, 1.0, 1.0],
            [1.0, 2.0, 1.0, 1.0],
            [1.0, 1.0, 2.0, 1.0],
            [1.0, 1.0, 1.0, 2.0]
        ]
        mm.options['train:y1'] = [3.0, 2.0, 1.0, 6.0, -2.0]
        mm.options['train:y2'] = [1.0, 4.0, 7.0, -3.0, 3.0]

        prob['mm.x'] = [1.0, 2.0, 1.0, 1.0]
        prob.run_model()

        assert_rel_error(self, prob['mm.y1'], 1.0, .00001)
        assert_rel_error(self, prob['mm.y2'], 7.0, .00001)

    def test_array_inputs(self):
        mm = MetaModelUnStructuredComp()
        mm.add_input('x', np.zeros((2,2)))
        mm.add_output('y1', 0.)
        mm.add_output('y2', 0.)

        mm.options['default_surrogate'] = FloatKrigingSurrogate()

        prob = Problem()
        prob.model.add_subsystem('mm', mm)
        prob.setup(check=False)

        mm.options['train:x'] = [
            [[1.0, 1.0], [1.0, 1.0]],
            [[2.0, 1.0], [1.0, 1.0]],
            [[1.0, 2.0], [1.0, 1.0]],
            [[1.0, 1.0], [2.0, 1.0]],
            [[1.0, 1.0], [1.0, 2.0]]
        ]
        mm.options['train:y1'] = [3.0, 2.0, 1.0, 6.0, -2.0]
        mm.options['train:y2'] = [1.0, 4.0, 7.0, -3.0, 3.0]

        prob['mm.x'] = [[1.0, 2.0], [1.0, 1.0]]
        prob.run_model()

        assert_rel_error(self, prob['mm.y1'], 1.0, .00001)
        assert_rel_error(self, prob['mm.y2'], 7.0, .00001)

    def test_array_outputs(self):
        mm = MetaModelUnStructuredComp()
        mm.add_input('x', np.zeros((2, 2)))
        mm.add_output('y', np.zeros(2,))

        mm.options['default_surrogate'] = FloatKrigingSurrogate()

        prob = Problem()
        prob.model.add_subsystem('mm', mm)
        prob.setup(check=False)

        mm.options['train:x'] = [
            [[1.0, 1.0], [1.0, 1.0]],
            [[2.0, 1.0], [1.0, 1.0]],
            [[1.0, 2.0], [1.0, 1.0]],
            [[1.0, 1.0], [2.0, 1.0]],
            [[1.0, 1.0], [1.0, 2.0]]
        ]

        mm.options['train:y'] = [
            [3.0, 1.0],
            [2.0, 4.0],
            [1.0, 7.0],
            [6.0, -3.0],
            [-2.0, 3.0]
        ]

        prob['mm.x'] = [[1.0, 2.0], [1.0, 1.0]]
        prob.run_model()

        assert_rel_error(self, prob['mm.y'], np.array([1.0, 7.0]), .00001)

    def test_2darray_outputs(self):
        mm = MetaModelUnStructuredComp()
        mm.add_input('x', np.zeros((2, 2)))
        mm.add_output('y', np.zeros((2, 2)))

        mm.options['default_surrogate'] = FloatKrigingSurrogate()

        prob = Problem()
        prob.model.add_subsystem('mm', mm)
        prob.setup(check=False)

        mm.options['train:x'] = [
            [[1.0, 1.0], [1.0, 1.0]],
            [[2.0, 1.0], [1.0, 1.0]],
            [[1.0, 2.0], [1.0, 1.0]],
            [[1.0, 1.0], [2.0, 1.0]],
            [[1.0, 1.0], [1.0, 2.0]]
        ]

        mm.options['train:y'] = [
            [[3.0, 1.0],[3.0, 1.0]],
            [[2.0, 4.0],[2.0, 4.0]],
            [[1.0, 7.0],[1.0, 7.0]],
            [[6.0, -3.0],[6.0, -3.0]],
            [[-2.0, 3.0],[-2.0, 3.0]]
        ]

        prob['mm.x'] = [[1.0, 2.0], [1.0, 1.0]]
        prob.run_model()

        assert_rel_error(self, prob['mm.y'], np.array([[1.0, 7.0], [1.0, 7.0]]), .00001)

    def test_unequal_training_inputs(self):
        mm = MetaModelUnStructuredComp()
        mm.add_input('x', 0.)
        mm.add_input('y', 0.)
        mm.add_output('f', 0.)

        mm.options['default_surrogate'] = FloatKrigingSurrogate()

        prob = Problem()
        prob.model.add_subsystem('mm', mm)
        prob.setup(check=False)

        mm.options['train:x'] = [1.0, 1.0, 1.0, 1.0]
        mm.options['train:y'] = [1.0, 2.0]
        mm.options['train:f'] = [1.0, 1.0, 1.0, 1.0]

        prob['mm.x'] = 1.0
        prob['mm.y'] = 1.0

        with self.assertRaises(RuntimeError) as cm:
            prob.run_model()

        expected = ("MetaModelUnStructuredComp: Each variable must have the same number"
                    " of training points. Expected 4 but found"
                    " 2 points for 'y'.")

        self.assertEqual(str(cm.exception), expected)

    def test_unequal_training_outputs(self):
        mm = MetaModelUnStructuredComp()
        mm.add_input('x', 0.)
        mm.add_input('y', 0.)
        mm.add_output('f', 0.)

        mm.options['default_surrogate'] = FloatKrigingSurrogate()

        prob = Problem()
        prob.model.add_subsystem('mm', mm)
        prob.setup(check=False)

        mm.options['train:x'] = [1.0, 1.0, 1.0, 1.0]
        mm.options['train:y'] = [1.0, 2.0, 3.0, 4.0]
        mm.options['train:f'] = [1.0, 1.0]

        prob['mm.x'] = 1.0
        prob['mm.y'] = 1.0

        with self.assertRaises(RuntimeError) as cm:
            prob.run_model()

        expected = ("MetaModelUnStructuredComp: Each variable must have the same number"
                    " of training points. Expected 4 but found"
                    " 2 points for 'f'.")
        self.assertEqual(str(cm.exception), expected)

    def test_derivatives(self):
        mm = MetaModelUnStructuredComp()
        mm.add_input('x', 0.)
        mm.add_output('f', 0.)

        mm.options['default_surrogate'] = FloatKrigingSurrogate()

        prob = Problem()
        prob.model.add_subsystem('p', IndepVarComp('x', 0.),
                                 promotes_outputs=['x'])
        prob.model.add_subsystem('mm', mm,
                                 promotes_inputs=['x'])
        prob.setup()

        mm.options['train:x'] = [0., .25, .5, .75, 1.]
        mm.options['train:f'] = [1., .75, .5, .25, 0.]

        prob['x'] = 0.125
        prob.run_model()

        data = prob.check_partials(out_stream=None)

        Jf = data['mm'][('f', 'x')]['J_fwd']
        Jr = data['mm'][('f', 'x')]['J_rev']

        assert_rel_error(self, Jf[0][0], -1., 1.e-3)
        assert_rel_error(self, Jr[0][0], -1., 1.e-3)

        abs_errors = data['mm'][('f', 'x')]['abs error']
        self.assertTrue(len(abs_errors) > 0)
        for match in abs_errors:
            abs_error = float(match)
            self.assertTrue(abs_error < 1.e-6)

        # Complex step
        prob.setup(force_alloc_complex=True)
        prob.model.mm.set_check_partial_options(wrt='*', method='cs')
        data = prob.check_partials(out_stream=None)

        abs_errors = data['mm'][('f', 'x')]['abs error']
        self.assertTrue(len(abs_errors) > 0)
        for match in abs_errors:
            abs_error = float(match)
            self.assertTrue(abs_error < 1.e-6)

    def test_metamodel_feature(self):
        # create a MetaModelUnStructuredComp, specifying surrogates for the outputs
        import numpy as np

        from openmdao.api import Problem, MetaModelUnStructuredComp, FloatKrigingSurrogate

        trig = MetaModelUnStructuredComp()

        x_train = np.linspace(0,10,20)

        trig.add_input('x', 0., training_data=x_train)

        trig.add_output('sin_x', 0.,
                        training_data=.5*np.sin(x_train),
                        surrogate=FloatKrigingSurrogate())
        trig.add_output('cos_x', 0.,
                        training_data=.5*np.cos(x_train))

        trig.options['default_surrogate'] = FloatKrigingSurrogate()

        # add it to a Problem, run and check the predicted values
        prob = Problem()
        prob.model.add_subsystem('trig', trig)
        prob.setup(check=False)

        prob['trig.x'] = 2.1
        prob.run_model()

        assert_rel_error(self, prob['trig.sin_x'], .5*np.sin(prob['trig.x']), 1e-4)
        assert_rel_error(self, prob['trig.cos_x'], .5*np.cos(prob['trig.x']), 1e-4)

    def test_metamodel_feature2d(self):
        # similar to previous example, but output is 2d
        import numpy as np

        from openmdao.api import Problem, MetaModelUnStructuredComp, FloatKrigingSurrogate

        # create a MetaModelUnStructuredComp that predicts sine and cosine as an array
        trig = MetaModelUnStructuredComp(default_surrogate=FloatKrigingSurrogate())
        trig.add_input('x', 0)
        trig.add_output('y', np.zeros(2))

        # add it to a Problem
        prob = Problem()
        prob.model.add_subsystem('trig', trig)
        prob.setup(check=False)

        # provide training data
        trig.options['train:x'] = np.linspace(0, 10, 20)
        trig.options['train:y'] = np.column_stack((
            .5*np.sin(trig.options['train:x']),
            .5*np.cos(trig.options['train:x'])
        ))

        # train the surrogate and check predicted value
        prob['trig.x'] = 2.1
        prob.run_model()
        assert_rel_error(self, prob['trig.y'],
                         np.append(
                             .5*np.sin(prob['trig.x']),
                             .5*np.cos(prob['trig.x'])
                         ),
                         1e-4)

    def test_vectorized(self):
        size = 3

        # create a vectorized MetaModelUnStructuredComp for sine
        trig = MetaModelUnStructuredComp(vec_size=size, default_surrogate=FloatKrigingSurrogate())
        trig.add_input('x', np.zeros(size))
        trig.add_output('y', np.zeros(size))

        # add it to a Problem
        prob = Problem()
        prob.model.add_subsystem('trig', trig)
        prob.setup(check=False)

        # provide training data
        trig.options['train:x'] = np.linspace(0, 10, 20)
        trig.options['train:y'] = .5*np.sin(trig.options['train:x'])

        # train the surrogate and check predicted value
        prob['trig.x'] = np.array([2.1, 3.2, 4.3])
        prob.run_model()
        assert_rel_error(self, prob['trig.y'],
                         np.array(.5*np.sin(prob['trig.x'])),
                         1e-4)

        data = prob.check_partials(out_stream=None)

        abs_errors = data['trig'][('y', 'x')]['abs error']
        self.assertTrue(len(abs_errors) > 0)
        for match in abs_errors:
            abs_error = float(match)
            self.assertTrue(abs_error < 1.e-6)

    def test_vectorized_kriging(self):
        # Test for coverage (handling the rmse)
        size = 3

        # create a vectorized MetaModelUnStructuredComp for sine
        trig = MetaModelUnStructuredComp(vec_size=size,
                                         default_surrogate=KrigingSurrogate(eval_rmse=True))
        trig.add_input('x', np.zeros(size))
        trig.add_output('y', np.zeros(size))

        # add it to a Problem
        prob = Problem()
        prob.model.add_subsystem('trig', trig)
        prob.setup(check=False)

        # provide training data
        trig.options['train:x'] = np.linspace(0, 10, 20)
        trig.options['train:y'] = .5*np.sin(trig.options['train:x'])

        # train the surrogate and check predicted value
        prob['trig.x'] = np.array([2.1, 3.2, 4.3])
        prob.run_model()
        assert_rel_error(self, prob['trig.y'],
                         np.array(.5*np.sin(prob['trig.x'])),
                         1e-4)
        self.assertEqual(len(prob.model.trig._metadata('y')['rmse']), 3)

    def test_derivatives_vectorized_multiD(self):
        vec_size = 5

        mm = MetaModelUnStructuredComp(vec_size=vec_size)
        mm.add_input('x', np.zeros((vec_size, 2, 3)))
        mm.add_input('xx', np.zeros((vec_size, 1)))
        mm.add_output('y', np.zeros((vec_size, 4, 2)))

        mm.options['default_surrogate'] = FloatKrigingSurrogate()

        prob = Problem()
        prob.model.add_subsystem('mm', mm)
        prob.setup(check=False)

        mm.options['train:x'] = [
            [[1.0, 2.0, 1.0], [1.0, 2.0, 1.0]],
            [[2.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            [[1.0, 1.0, 2.0], [1.0, 2.0, 1.0]],
            [[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]],
            [[1.0, 2.0, 1.0], [1.0, 2.0, 2.0]]
        ]

        mm.options['train:xx'] = [1.0, 2.0, 1.0, 1.0, 2.0]


        mm.options['train:y'] = [
            [[30.0, 10.0], [30.0, 25.0], [50.0, 10.7], [15.0, 25.7]],
            [[20.0, 40.0], [20.0, 40.0], [80.0, 30.3], [12.0, 20.7]],
            [[10.0, 70.0], [10.0, 70.0], [20.0, 10.9], [13.0, 15.7]],
            [[60.0, -30.0], [60.0, -30.0], [50.0, 50.5], [14.0, 10.7]],
            [[-20.0, 30.0], [-20.0, 30.0], [20.2, 10.0], [15.0, 60.7]]
        ]

        prob['mm.x'] = [[[1.3, 1.3, 1.3], [1.5, 1.5, 1.5]],
                        [[1.4, 1.4, 1.4], [1.5, 1.5, 1.5]],
                        [[1.5, 1.5, 1.5], [1.5, 1.5, 1.5]],
                        [[1.5, 1.5, 1.5], [1.4, 1.4, 1.4]],
                        [[1.5, 1.5, 1.5], [1.3, 1.3, 1.3]]]

        prob['mm.xx'] = [[1.4], [1.5], [1.6], [1.5], [1.4]]

        prob.run_model()

        data = prob.check_partials(out_stream=None)

        abs_errors = data['mm'][('y', 'x')]['abs error']
        self.assertTrue(len(abs_errors) > 0)
        for match in abs_errors:
            abs_error = float(match)
            self.assertTrue(abs_error < 1.e-5)

        abs_errors = data['mm'][('y', 'xx')]['abs error']
        self.assertTrue(len(abs_errors) > 0)
        for match in abs_errors:
            abs_error = float(match)
            self.assertTrue(abs_error < 1.e-5)

        # Complex step
        prob.setup(force_alloc_complex=True)
        prob.model.mm.set_check_partial_options(wrt='*', method='cs')
        data = prob.check_partials(out_stream=None)

        abs_errors = data['mm'][('y', 'x')]['abs error']
        self.assertTrue(len(abs_errors) > 0)
        for match in abs_errors:
            abs_error = float(match)
            self.assertTrue(abs_error < 1.e-5)

        abs_errors = data['mm'][('y', 'xx')]['abs error']
        self.assertTrue(len(abs_errors) > 0)
        for match in abs_errors:
            abs_error = float(match)
            self.assertTrue(abs_error < 1.e-5)

    def test_metamodel_feature_vector(self):
        # Like simple sine example, but with input of length n instead of scalar
        # The expected behavior is that the output is also of length n, with
        # each one being an independent prediction.
        # Its as if you stamped out n copies of metamodel, ran n scalars
        # through its input, then muxed all those outputs into one contiguous
        # array but you skip all the n-copies thing and do it all as an array
        import numpy as np

        from openmdao.api import Problem, MetaModelUnStructuredComp, FloatKrigingSurrogate

        size = 3

        # create a vectorized MetaModelUnStructuredComp for sine
        trig = MetaModelUnStructuredComp(vec_size=size, default_surrogate=FloatKrigingSurrogate())
        trig.add_input('x', np.zeros(size))
        trig.add_output('y', np.zeros(size))

        # add it to a Problem
        prob = Problem()
        prob.model.add_subsystem('trig', trig)
        prob.setup(check=False)

        # provide training data
        trig.options['train:x'] = np.linspace(0, 10, 20)
        trig.options['train:y'] = .5*np.sin(trig.options['train:x'])

        # train the surrogate and check predicted value
        prob['trig.x'] = np.array([2.1, 3.2, 4.3])
        prob.run_model()
        assert_rel_error(self, prob['trig.y'],
                         np.array(.5*np.sin(prob['trig.x'])),
                         1e-4)

    def test_metamodel_feature_vector2d(self):
        # similar to previous example, but processes 3 inputs/outputs at a time
        import numpy as np

        from openmdao.api import Problem, MetaModelUnStructuredComp, FloatKrigingSurrogate

        size = 3

        # create a vectorized MetaModelUnStructuredComp for sine and cosine
        trig = MetaModelUnStructuredComp(vec_size=size, default_surrogate=FloatKrigingSurrogate())
        trig.add_input('x', np.zeros(size))
        trig.add_output('y', np.zeros((size, 2)))

        # add it to a Problem
        prob = Problem()
        prob.model.add_subsystem('trig', trig)
        prob.setup(check=False)

        # provide training data
        trig.options['train:x'] = np.linspace(0, 10, 20)
        trig.options['train:y'] = np.column_stack((
            .5*np.sin(trig.options['train:x']),
            .5*np.cos(trig.options['train:x'])
        ))

        # train the surrogate and check predicted value
        prob['trig.x'] = np.array([2.1, 3.2, 4.3])
        prob.run_model()
        assert_rel_error(self, prob['trig.y'],
                         np.column_stack((
                             .5*np.sin(prob['trig.x']),
                             .5*np.cos(prob['trig.x'])
                         )),
                         1e-4)

    def test_metamodel_vector_errors(self):
        # first dimension of all inputs/outputs must be 3
        mm = MetaModelUnStructuredComp(vec_size=3)

        with self.assertRaises(RuntimeError) as cm:
            mm.add_input('x', np.zeros(2))
        self.assertEqual(str(cm.exception),
                         "Metamodel: First dimension of input 'x' must be 3")

        with self.assertRaises(RuntimeError) as cm:
            mm.add_output('y', np.zeros(4))
        self.assertEqual(str(cm.exception),
                         "Metamodel: First dimension of output 'y' must be 3")

    def test_metamodel_subclass_optimize(self):
        class Trig(MetaModelUnStructuredComp):
            def setup(self):
                self.add_input('x', 0.,
                               training_data=np.linspace(0,10,20))
                self.add_output('sin_x', 0.,
                                surrogate=FloatKrigingSurrogate(),
                                training_data=.5*np.sin(np.linspace(0,10,20)))

                self.declare_partials(of='sin_x', wrt='x', method='fd')

        prob = Problem()

        indep = IndepVarComp()
        indep.add_output('x', 5.)

        prob.model.add_subsystem('indep', indep)
        prob.model.add_subsystem('trig', Trig())

        prob.model.connect('indep.x', 'trig.x')

        prob.driver = ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'COBYLA'

        prob.model.add_design_var('indep.x', lower=4, upper=7)
        prob.model.add_objective('trig.sin_x')

        prob.setup(check=True)

        self.assertEqual(prob['trig.x'], [5.])
        assert_rel_error(self, prob['trig.sin_x'], [.0], 1e-6)

    def test_meta_model_unstructured_deprecated(self):
        # run same test as above, only with the deprecated component,
        # to ensure we get the warning and the correct answer.
        # self-contained, to be removed when class name goes away.
        from openmdao.components.meta_model_unstructured_comp import MetaModelUnStructured  # deprecated
        import warnings

        with warnings.catch_warnings(record=True) as w:
            mm = MetaModelUnStructured()

        self.assertEqual(len(w), 1)
        self.assertTrue(issubclass(w[0].category, DeprecationWarning))
        self.assertEqual(str(w[0].message), "'MetaModelUnStructured' has been deprecated. Use "
                                            "'MetaModelUnStructuredComp' instead.")

        mm.add_input('x1', 0.)
        mm.add_input('x2', 0.)

        mm.add_output('y1', 0.)
        mm.add_output('y2', 0., surrogate=FloatKrigingSurrogate())

        with warnings.catch_warnings(record=True) as w:
            mm.default_surrogate = ResponseSurface()
        self.assertEqual(len(w), 1)
        self.assertTrue(issubclass(w[0].category, DeprecationWarning))
        self.assertEqual(str(w[0].message),
                         "The 'default_surrogate' attribute provides backwards compatibility "
                         "with earlier version of OpenMDAO; use options['default_surrogate'] "
                         "instead.")

        # add metamodel to a problem
        prob = Problem(model=Group())
        prob.model.add_subsystem('mm', mm)
        prob.setup(check=False)

        # check that surrogates were properly assigned
        surrogate = mm._metadata('y1').get('surrogate')
        self.assertTrue(isinstance(surrogate, ResponseSurface))

        surrogate = mm._metadata('y2').get('surrogate')
        self.assertTrue(isinstance(surrogate, FloatKrigingSurrogate))

        # populate training data
        with warnings.catch_warnings(record=True) as w:
            mm.metadata['train:x1'] = [1.0, 2.0, 3.0]
            mm.metadata['train:x2'] = [1.0, 3.0, 4.0]
            mm.metadata['train:y1'] = [3.0, 2.0, 1.0]
            mm.metadata['train:y2'] = [1.0, 4.0, 7.0]
        self.assertEqual(len(w), 4)
        self.assertTrue(issubclass(w[0].category, DeprecationWarning))
        self.assertEqual(str(w[0].message),
                         "The 'metadata' attribute provides backwards compatibility "
                         "with earlier version of OpenMDAO; use 'options' instead.")

        # run problem for provided data point and check prediction
        prob['mm.x1'] = 2.0
        prob['mm.x2'] = 3.0

        self.assertTrue(mm.train)   # training will occur before 1st run
        prob.run_model()

        assert_rel_error(self, prob['mm.y1'], 2.0, .00001)
        assert_rel_error(self, prob['mm.y2'], 4.0, .00001)

        # run problem for interpolated data point and check prediction
        prob['mm.x1'] = 2.5
        prob['mm.x2'] = 3.5

        self.assertFalse(mm.train)  # training will not occur before 2nd run
        prob.run_model()

        assert_rel_error(self, prob['mm.y1'], 1.5934, .001)

        # change default surrogate, re-setup and check that metamodel re-trains
        with warnings.catch_warnings(record=True) as w:
            mm.default_surrogate = FloatKrigingSurrogate()
        self.assertEqual(len(w), 1)
        self.assertTrue(issubclass(w[0].category, DeprecationWarning))
        self.assertEqual(str(w[0].message),
                         "The 'default_surrogate' attribute provides backwards compatibility "
                         "with earlier version of OpenMDAO; use options['default_surrogate'] "
                         "instead.")
        prob.setup(check=False)

        surrogate = mm._metadata('y1').get('surrogate')
        self.assertTrue(isinstance(surrogate, FloatKrigingSurrogate))

        self.assertTrue(mm.train)  # training will occur after re-setup

        prob['mm.x1'] = 2.5
        prob['mm.x2'] = 3.5

        prob.run_model()
        assert_rel_error(self, prob['mm.y1'], 1.5, 1e-2)

    def test_metamodel_deprecated(self):
        # run same test as above, only with the deprecated component,
        # to ensure we get the warning and the correct answer.
        # self-contained, to be removed when class name goes away.
        from openmdao.components.meta_model_unstructured_comp import MetaModel  # deprecated
        import warnings

        with warnings.catch_warnings(record=True) as w:
            mm = MetaModel()

        self.assertEqual(len(w), 1)
        self.assertTrue(issubclass(w[0].category, DeprecationWarning))
        self.assertEqual(str(w[0].message), "'MetaModel' has been deprecated. Use "
                         "'MetaModelUnStructuredComp' instead.")

        mm.add_input('x1', 0.)
        mm.add_input('x2', 0.)

        mm.add_output('y1', 0.)
        mm.add_output('y2', 0., surrogate=FloatKrigingSurrogate())

        with warnings.catch_warnings(record=True) as w:
            mm.default_surrogate = ResponseSurface()
        self.assertEqual(len(w), 1)
        self.assertTrue(issubclass(w[0].category, DeprecationWarning))
        self.assertEqual(str(w[0].message),
                         "The 'default_surrogate' attribute provides backwards compatibility "
                         "with earlier version of OpenMDAO; use options['default_surrogate'] "
                         "instead.")

        # add metamodel to a problem
        prob = Problem(model=Group())
        prob.model.add_subsystem('mm', mm)
        prob.setup(check=False)

        # check that surrogates were properly assigned
        surrogate = mm._metadata('y1').get('surrogate')
        self.assertTrue(isinstance(surrogate, ResponseSurface))

        surrogate = mm._metadata('y2').get('surrogate')
        self.assertTrue(isinstance(surrogate, FloatKrigingSurrogate))

        # populate training data
        with warnings.catch_warnings(record=True) as w:
            mm.metadata['train:x1'] = [1.0, 2.0, 3.0]
            mm.metadata['train:x2'] = [1.0, 3.0, 4.0]
            mm.metadata['train:y1'] = [3.0, 2.0, 1.0]
            mm.metadata['train:y2'] = [1.0, 4.0, 7.0]
        self.assertEqual(len(w), 4)
        self.assertTrue(issubclass(w[0].category, DeprecationWarning))
        self.assertEqual(str(w[0].message),
                         "The 'metadata' attribute provides backwards compatibility "
                         "with earlier version of OpenMDAO; use 'options' instead.")

        # run problem for provided data point and check prediction
        prob['mm.x1'] = 2.0
        prob['mm.x2'] = 3.0

        self.assertTrue(mm.train)   # training will occur before 1st run
        prob.run_model()

        assert_rel_error(self, prob['mm.y1'], 2.0, .00001)
        assert_rel_error(self, prob['mm.y2'], 4.0, .00001)

        # run problem for interpolated data point and check prediction
        prob['mm.x1'] = 2.5
        prob['mm.x2'] = 3.5

        self.assertFalse(mm.train)  # training will not occur before 2nd run
        prob.run_model()

        assert_rel_error(self, prob['mm.y1'], 1.5934, .001)

        # change default surrogate, re-setup and check that metamodel re-trains
        with warnings.catch_warnings(record=True) as w:
            mm.default_surrogate = FloatKrigingSurrogate()
        self.assertEqual(len(w), 1)
        self.assertTrue(issubclass(w[0].category, DeprecationWarning))
        self.assertEqual(str(w[0].message),
                         "The 'default_surrogate' attribute provides backwards compatibility "
                         "with earlier version of OpenMDAO; use options['default_surrogate'] "
                         "instead.")
        prob.setup(check=False)

        surrogate = mm._metadata('y1').get('surrogate')
        self.assertTrue(isinstance(surrogate, FloatKrigingSurrogate))

        self.assertTrue(mm.train)  # training will occur after re-setup

        prob['mm.x1'] = 2.5
        prob['mm.x2'] = 3.5

        prob.run_model()
        assert_rel_error(self, prob['mm.y1'], 1.5, 1e-2)

    # def test_metamodel_use_fd_if_no_surrogate_linearize(self):
    #     class Trig(MetaModelUnStructuredComp):
    #         def setup(self):
    #
    #             surrogate = FloatKrigingSurrogate()
    #             self.add_input('x', 0.,
    #                            training_data=np.linspace(0,10,20))
    #             self.add_output('sin_x', 0.,
    #                             surrogate=surrogate,
    #                             training_data=.5*np.sin(np.linspace(0,10,20)))
    #
    #     prob = Problem()
    #     indep = IndepVarComp()
    #     indep.add_output('x', 5.)
    #     prob.model.add_subsystem('indep', indep)
    #     prob.model.add_subsystem('trig', Trig())
    #     prob.model.connect('indep.x', 'trig.x')
    #
    #     prob.model.add_design_var('indep.x', lower=4, upper=7)
    #     prob.model.add_objective('trig.sin_x')
    #
    #     prob.setup(check=True)
    #     prob['indep.x'] = 5.0
    #     prob.run_model()
    #     J = prob.compute_totals()
    #
    #     deriv_using_surrogate_linearize = J[('trig.sin_x', 'indep.x')]
    #
    #     prob['indep.x'] = 5.0
    #
    #     # Check to see that if the surrogate does not have a linearize,
    #     #    fd is used to compute the derivatives correctly
    #     linearize = KrigingSurrogate.linearize #  save for later
    #     del KrigingSurrogate.linearize
    #
    #     prob.run_model()
    #     J = prob.compute_totals()
    #
    #     # restore this so that other tests are not affected
    #     KrigingSurrogate.linearize = linearize
    #
    #     deriv_using_fd = J[('trig.sin_x', 'indep.x')]
    #     assert_rel_error(self, deriv_using_fd, deriv_using_surrogate_linearize, 1e-4)
    #     # Just to be extra sure
    #     assert_rel_error(self, deriv_using_fd[0], .5 * np.cos(prob['indep.x']), 1e-4)

    # def test_metamodel_use_fd_if_no_surrogate_linearize_original(self):
    #     class Trig(MetaModelUnStructuredComp):
    #         def setup(self):
    #
    #             surrogate = FloatKrigingSurrogate()
    #             self.add_input('x', 0.,
    #                            training_data=np.linspace(0,10,20))
    #             self.add_output('sin_x', 0.,
    #                             surrogate=surrogate,
    #                             training_data=.5*np.sin(np.linspace(0,10,20)))
    #
    #     # prob = Problem()
    #     # indep = IndepVarComp()
    #     # indep.add_output('x', 5.)
    #     # prob.model.add_subsystem('indep', indep)
    #     # prob.model.add_subsystem('trig', Trig())
    #     # prob.model.connect('indep.x', 'trig.x')
    #     #
    #     # prob.model.add_design_var('indep.x', lower=4, upper=7)
    #     # prob.model.add_objective('trig.sin_x')
    #     #
    #     # prob.setup(check=True)
    #     # prob['indep.x'] = 5.0
    #     # prob.run_model()
    #     # J = prob.compute_totals()
    #     #
    #     # deriv_using_surrogate_linearize = J[('trig.sin_x', 'indep.x')]
    #     #
    #
    #
    #
    #     prob = Problem()
    #     indep = IndepVarComp()
    #     indep.add_output('x', 5.)
    #     prob.model.add_subsystem('indep', indep)
    #     prob.model.add_subsystem('trig', Trig())
    #     prob.model.connect('indep.x', 'trig.x')
    #
    #     prob.model.add_design_var('indep.x', lower=4, upper=7)
    #     prob.model.add_objective('trig.sin_x')
    #
    #     prob.setup(check=True)
    #     prob['indep.x'] = 5.0
    #     prob.run_model()
    #     J = prob.compute_totals()
    #
    #     deriv_using_surrogate_linearize = J[('trig.sin_x', 'indep.x')]
    #
    #
    #     linearize = KrigingSurrogate.linearize #  save for later
    #     del KrigingSurrogate.linearize
    #
    #
    #     prob = Problem()
    #     indep = IndepVarComp()
    #     indep.add_output('x', 5.)
    #     prob.model.add_subsystem('indep', indep)
    #     prob.model.add_subsystem('trig', Trig())
    #     prob.model.connect('indep.x', 'trig.x')
    #
    #     prob.model.add_design_var('indep.x', lower=4, upper=7)
    #     prob.model.add_objective('trig.sin_x')
    #
    #     prob.setup(check=True)
    #     prob['indep.x'] = 5.0
    #     prob.run_model()
    #     J = prob.compute_totals()
    #
    #     deriv_using_fd = J[('trig.sin_x', 'indep.x')]
    #
    #     # restore this so that other tests are not affected
    #     KrigingSurrogate.linearize = linearize
    #
    #     deriv_using_fd = J[('trig.sin_x', 'indep.x')]
    #     assert_rel_error(self, deriv_using_fd, deriv_using_surrogate_linearize, 1e-4)
    #     # Just to be extra sure
    #     assert_rel_error(self, deriv_using_fd[0], .5 * np.cos(prob['indep.x']), 1e-4)
    #

    # def test_metamodel_use_fd_if_no_surrogate_linearize_v5(self):
    #     class Trig(MetaModelUnStructuredComp):
    #         def setup(self):
    #
    #             surrogate = FloatKrigingSurrogate()
    #             self.add_input('x', 0.,
    #                            training_data=np.linspace(0,10,20))
    #             self.add_output('sin_x', 0.,
    #                             surrogate=surrogate,
    #                             training_data=.5*np.sin(np.linspace(0,10,20)))
    #
    #     prob = Problem()
    #     indep = IndepVarComp()
    #     indep.add_output('x', 5.)
    #     prob.model.add_subsystem('indep', indep)
    #     prob.model.add_subsystem('trig', Trig())
    #     prob.model.connect('indep.x', 'trig.x')
    #
    #     prob.model.add_design_var('indep.x', lower=4, upper=7)
    #     prob.model.add_objective('trig.sin_x')
    #
    #     prob.setup(check=True)
    #     prob['indep.x'] = 5.0
    #     prob.run_model()
    #     J = prob.compute_totals()
    #
    #     deriv_using_surrogate_linearize = J[('trig.sin_x', 'indep.x')]

    def test_metamodel_use_fd_if_no_surrogate_linearize(self):
        from openmdao.api import SurrogateModel
        class SinSurrogate(SurrogateModel):

            def train(self, x, y):
                pass

            def predict(self, x):
                from math import sin
                y = sin(x)
                return y

        class SinTwoInputsSurrogate(SurrogateModel):

            def train(self, x, y):
                pass

            def predict(self, x):
                from math import sin
                y = sin(x[0] + x[1])
                return y

        class Trig(MetaModelUnStructuredComp):
            def setup(self):
                surrogate = SinSurrogate()
                self.add_input('x', 0.,
                               training_data=np.linspace(0, 10, 20))
                self.add_output('sin_x', 0.,
                                surrogate=surrogate,
                                training_data=np.sin(np.linspace(0, 10, 20)))

        class TrigWithFdInSetup(MetaModelUnStructuredComp):
            def setup(self):
                surrogate = SinSurrogate()
                self.add_input('x', 0.,
                               training_data=np.linspace(0, 10, 20))
                self.add_output('sin_x', 0.,
                                surrogate=surrogate,
                                training_data=np.sin(np.linspace(0, 10, 20)))
                self.declare_partials('sin_x', 'x', method='fd',
                                      form='backward', step=1e-7,
                                      step_calc='rel')

        class TrigWithCsInSetup(MetaModelUnStructuredComp):
            def setup(self):
                surrogate = SinSurrogate()
                self.add_input('x', 0.,
                               training_data=np.linspace(0, 10, 20))
                self.add_output('sin_x', 0.,
                                surrogate=surrogate,
                                training_data=np.sin(np.linspace(0, 10, 20)))
                self.declare_partials('sin_x', 'x', method='cs')

        class TrigGroup(Group):
            def configure(self):
                trig = self._get_subsystem('trig')
                trig.declare_partials('sin_x', 'x', method='fd',
                                      form='backward', step=1e-7,
                                      step_calc='rel')

        class TrigWithFdInConfigure(MetaModelUnStructuredComp):
            def setup(self):
                surrogate = SinSurrogate()
                self.add_input('x', 0.,
                               training_data=np.linspace(0, 10, 20))
                self.add_output('sin_x', 0.,
                                surrogate=surrogate,
                                training_data=np.sin(np.linspace(0, 10, 20)))

        class TrigTwoInputsWithFdInSetup(MetaModelUnStructuredComp):
            def setup(self):
                surrogate = SinTwoInputsSurrogate()
                self.add_input('x1', 0.,
                               training_data=np.linspace(0, 10, 20))
                self.add_input('x2', 0.,
                               training_data=np.linspace(0, 10, 20))
                self.add_output('sin_x', 0.,
                                surrogate=surrogate,
                                training_data=np.sin(np.linspace(0, 10, 20)))
                self.declare_partials('sin_x', 'x1', method='fd',
                                      form='backward', step=1e-7,
                                      step_calc='rel')

        def no_surrogate_test_setup(trig, group=None):
            prob = Problem()
            if group:
                prob.model = group
            indep = IndepVarComp()
            indep.add_output('x', 5.)
            prob.model.add_subsystem('indep', indep)
            prob.model.add_subsystem('trig', trig)
            prob.model.connect('indep.x', 'trig.x')
            # prob.model.add_design_var('indep.x', lower=4, upper=7)
            prob.model.add_design_var('indep.x')
            prob.model.add_objective('trig.sin_x')
            prob.setup(check=True)
            prob['indep.x'] = 5.0
            prob.run_model()
            return prob

        # Test with user not explicitly setting fd
        trig = Trig()
        prob = no_surrogate_test_setup(trig)
        J = prob.compute_totals()
        deriv_using_surrogate_linearize = J[('trig.sin_x', 'indep.x')]
        assert_rel_error(self, deriv_using_surrogate_linearize[0], np.cos(prob['indep.x']), 1e-4)

        # Test with user not explicitly setting fd
        # prob = Problem()
        # indep = IndepVarComp()
        # indep.add_output('x', 5.)
        # prob.model.add_subsystem('indep', indep)
        # trig = Trig()
        # prob.model.add_subsystem('trig', trig)
        # prob.model.connect('indep.x', 'trig.x')
        # prob.model.add_design_var('indep.x', lower=4, upper=7)
        # prob.model.add_objective('trig.sin_x')
        # prob.setup(check=True)
        # prob['indep.x'] = 5.0
        # prob.run_model()
        #
        # J = prob.compute_totals()
        # deriv_using_surrogate_linearize = J[('trig.sin_x', 'indep.x')]
        # assert_rel_error(self, deriv_using_surrogate_linearize[0], np.cos(prob['indep.x']), 1e-4)

        # Test with user explicitly setting fd inside of setup
        trig = TrigWithFdInSetup()
        prob = no_surrogate_test_setup(trig)
        of, wrt, method, fd_options = trig._approximated_partials[0]
        expected_fd_options = {
            'step': 1e-7,
            'form': 'backward',
            'step_calc': 'rel',
        }
        self.assertEqual(expected_fd_options, fd_options)
        J = prob.compute_totals()
        deriv_using_surrogate_linearize = J[('trig.sin_x', 'indep.x')]
        assert_rel_error(self, deriv_using_surrogate_linearize[0], np.cos(prob['indep.x']), 1e-4)

        # Test with user explicitly setting fd inside of setup
        # prob = Problem()
        # indep = IndepVarComp()
        # indep.add_output('x', 5.)
        # prob.model.add_subsystem('indep', indep)
        # trig = TrigWithFdInSetup()
        # prob.model.add_subsystem('trig', trig)
        # prob.model.connect('indep.x', 'trig.x')
        # prob.model.add_design_var('indep.x', lower=4, upper=7)
        # prob.model.add_objective('trig.sin_x')
        # prob.setup(check=True)
        # of, wrt, method, fd_options = trig._approximated_partials[0]
        # expected_fd_options = {
        #     'step': 1e-7,
        #     'form': 'backward',
        #     'step_calc': 'rel',
        # }
        # self.assertEqual(expected_fd_options, fd_options)
        # prob['indep.x'] = 5.0
        # prob.run_model()
        # J = prob.compute_totals()

        # Test with user explicitly setting fd inside of configure for a group
        trig = TrigWithFdInConfigure()
        prob = no_surrogate_test_setup(trig, group = TrigGroup())
        of, wrt, method, fd_options = trig._approximated_partials[0]
        expected_fd_options = {
            'step': 1e-7,
            'form': 'backward',
            'step_calc': 'rel',
        }
        self.assertEqual(expected_fd_options, fd_options)
        J = prob.compute_totals()
        deriv_using_surrogate_linearize = J[('trig.sin_x', 'indep.x')]
        assert_rel_error(self, deriv_using_surrogate_linearize[0], np.cos(prob['indep.x']), 1e-4)


        # Test with user explicitly setting fd inside of configure for a group
        # prob = Problem()
        # prob.model = TrigGroup()
        # indep = IndepVarComp()
        # indep.add_output('x', 5.)
        # prob.model.add_subsystem('indep', indep)
        # trig = TrigWithFdInConfigure()
        # prob.model.add_subsystem('trig', trig)
        # prob.model.connect('indep.x', 'trig.x')
        # prob.model.add_design_var('indep.x', lower=4, upper=7)
        # prob.model.add_objective('trig.sin_x')
        # prob.setup(check=True)
        # prob['indep.x'] = 5.0
        # prob.run_model()
        #
        # of, wrt, method, fd_options = trig._approximated_partials[0]
        # expected_fd_options = {
        #     'step': 1e-7,
        #     'form': 'backward',
        #     'step_calc': 'rel',
        # }
        # self.assertEqual(expected_fd_options, fd_options)
        #
        # J = prob.compute_totals()
        #
        # deriv_using_surrogate_linearize = J[('trig.sin_x', 'indep.x')]
        # assert_rel_error(self, deriv_using_surrogate_linearize[0], np.cos(prob['indep.x']), 1e-4)

        # Test with user explicitly setting cs inside of setup. Should throw an error
        prob = Problem()
        indep = IndepVarComp()
        indep.add_output('x', 5.)
        prob.model.add_subsystem('indep', indep)
        trig = TrigWithCsInSetup()
        prob.model.add_subsystem('trig', trig)
        prob.model.connect('indep.x', 'trig.x')
        prob.model.add_design_var('indep.x', lower=4, upper=7)
        prob.model.add_objective('trig.sin_x')

        with self.assertRaises(ValueError) as context:
            prob.setup(check=True)
        expected_msg = 'Complex step has not been tested for MetaModelUnStructuredComp'
        self.assertEqual(expected_msg, str(context.exception))

        # Test with user explicitly setting fd on one of the inputs for a meta model
        #   with two inputs. Check to make sure all inputs are fd
        prob = Problem()
        indep = IndepVarComp()
        indep.add_output('x1', 5.)
        indep.add_output('x2', 5.)
        prob.model.add_subsystem('indep', indep)
        trig = TrigTwoInputsWithFdInSetup()
        prob.model.add_subsystem('trig', trig)
        prob.model.connect('indep.x1', 'trig.x1')
        prob.model.connect('indep.x2', 'trig.x2')
        prob.model.add_design_var('indep.x1', lower=4, upper=7)
        prob.model.add_design_var('indep.x2', lower=4, upper=7)
        prob.model.add_objective('trig.sin_x')
        prob.setup(check=True)
        prob['indep.x1'] = 5.0
        prob['indep.x2'] = 5.0
        prob.run_model()

        self.assertEqual('fd', trig._subjacs_info[('trig.sin_x', 'trig.x1')]['method'])
        self.assertEqual('backward', trig._subjacs_info[('trig.sin_x', 'trig.x1')]['form'])
        self.assertEqual(1e-07, trig._subjacs_info[('trig.sin_x', 'trig.x1')]['step'])
        self.assertEqual('rel', trig._subjacs_info[('trig.sin_x', 'trig.x1')]['step_calc'])

        self.assertEqual('fd', trig._subjacs_info[('trig.sin_x', 'trig.x2')]['method'])
        self.assertTrue('form' not in trig._subjacs_info[('trig.sin_x', 'trig.x2')])
        self.assertTrue('step' not in trig._subjacs_info[('trig.sin_x', 'trig.x2')])
        self.assertTrue('step_calc' not in trig._subjacs_info[('trig.sin_x', 'trig.x2')])

    # def test_metamodel_use_fd_if_no_surrogate_linearize_using_sin_junk(self):
    #     class SinTwoInputsSurrogate(SurrogateModel):
    #
    #         def train(self, x, y):
    #             pass
    #
    #         def predict(self, x):
    #             from math import sin
    #             y = sin(x[0] + x[1])
    #             return y
    #
    #     class TrigWithFdInSetup(MetaModelUnStructuredComp):
    #         def setup(self):
    #             surrogate = SinTwoInputsSurrogate()
    #             self.add_input('x1', 0.,
    #                            training_data=np.linspace(0, 10, 20))
    #             self.add_input('x2', 0.,
    #                            training_data=np.linspace(0, 10, 20))
    #             self.add_output('sin_x', 0.,
    #                             surrogate=surrogate,
    #                             training_data=np.sin(np.linspace(0, 10, 20)))
    #             # self.declare_partials('sin_x', 'x', method='fd', step=None, form=None,
    #             #                       step_calc=None)
    #             # self.declare_partials('sin_x', '*', method='fd',
    #             #                       form='forward', step=1e-8,
    #             #                       step_calc='rel')
    #             self.declare_partials('sin_x', 'x1', method='fd',
    #                                   form='backward', step=1e-7,
    #                                   step_calc='rel')
    #
    #     # Test with user explicitly setting fd inside of setup
    #     prob = Problem()
    #     indep = IndepVarComp()
    #     indep.add_output('x1', 5.)
    #     indep.add_output('x2', 5.)
    #     prob.model.add_subsystem('indep', indep)
    #     trig = TrigWithFdInSetup()
    #     prob.model.add_subsystem('trig', trig)
    #     prob.model.connect('indep.x1', 'trig.x1')
    #     prob.model.connect('indep.x2', 'trig.x2')
    #     prob.model.add_design_var('indep.x1', lower=4, upper=7)
    #     prob.model.add_design_var('indep.x2', lower=4, upper=7)
    #     prob.model.add_objective('trig.sin_x')
    #     prob.setup(check=True)
    #     of, wrt, method, fd_options = trig._approximated_partials[0]
    #     expected_fd_options = {
    #         'step': 1e-7,
    #         'form': 'backward',
    #         'step_calc': 'rel',
    #     }
    #     # self.assertEqual(expected_fd_options, fd_options)
    #     prob['indep.x1'] = 5.0
    #     prob['indep.x2'] = 5.0
    #     prob.run_model()
    #
    #     self.assertEqual('fd', trig._subjacs_info[('trig.sin_x', 'trig.x1')]['method'])
    #     self.assertEqual('backward', trig._subjacs_info[('trig.sin_x', 'trig.x1')]['form'])
    #     self.assertEqual(1e-07, trig._subjacs_info[('trig.sin_x', 'trig.x1')]['step'])
    #     self.assertEqual('rel', trig._subjacs_info[('trig.sin_x', 'trig.x1')]['step_calc'])
    #
    #     self.assertEqual('fd', trig._subjacs_info[('trig.sin_x', 'trig.x2')]['method'])
    #     self.assertTrue('form' not in trig._subjacs_info[('trig.sin_x', 'trig.x2')])
    #     self.assertTrue('step' not in trig._subjacs_info[('trig.sin_x', 'trig.x2')])
    #     self.assertTrue('step_calc' not in trig._subjacs_info[('trig.sin_x', 'trig.x2')])


if __name__ == "__main__":
    unittest.main()
