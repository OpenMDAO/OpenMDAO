"""
Unit tests for the unstructured metamodel component.
"""
import sys
import unittest
from math import sin
from io import StringIO

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal, assert_warning, assert_check_partials
from openmdao.utils.logger_utils import TestLogger


class MetaModelTestCase(unittest.TestCase):

    def test_sin_metamodel(self):
        # create a MetaModelUnStructuredComp for sine and add it to a om.Problem
        sin_mm = om.MetaModelUnStructuredComp()
        sin_mm.add_input('x', 0.)
        sin_mm.add_output('f_x', 0.)

        prob = om.Problem()
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
        sin_mm.options['default_surrogate'] = om.KrigingSurrogate()
        prob.setup()
        surrogate = sin_mm._metadata('f_x').get('surrogate')
        self.assertTrue(isinstance(surrogate, om.KrigingSurrogate),
                        'sin_mm.f_x should get the default surrogate')

        # check error message when no training data is provided
        with self.assertRaises(RuntimeError) as cm:
            prob.run_model()

        msg = ("MetaModelUnStructuredComp (sin_mm): The following training data sets must be "
               "provided as options: ['train:x', 'train:f_x']")
        self.assertEqual(str(cm.exception), msg)

        # train the surrogate and check predicted value
        sin_mm.options['train:x'] = np.linspace(0,10,20)
        sin_mm.options['train:f_x'] = .5*np.sin(sin_mm.options['train:x'])

        prob['sin_mm.x'] = 2.1

        prob.run_model()

        assert_near_equal(prob['sin_mm.f_x'], .5*np.sin(prob['sin_mm.x']), 1e-4)

    def test_error_no_surrogate(self):
        # Seems like the error message from above should also be present and readable even if the
        # user chooses to skip checking the model.
        sin_mm = om.MetaModelUnStructuredComp()
        sin_mm.add_input('x', 0.)
        sin_mm.add_output('f_x', 0.)

        prob = om.Problem()
        prob.model.add_subsystem('sin_mm', sin_mm)

        prob.setup()

        sin_mm.options['train:x'] = np.linspace(0,10,20)
        sin_mm.options['train:f_x'] = .5*np.sin(sin_mm.options['train:x'])

        with self.assertRaises(RuntimeError) as cm:
            prob.run_model()

        msg = ("MetaModelUnStructuredComp (sin_mm): No surrogate specified for output 'f_x'")
        self.assertEqual(str(cm.exception), msg)

    def test_sin_metamodel_preset_data(self):
        # preset training data
        x = np.linspace(0,10,200)
        f_x = .5*np.sin(x)

        # create a MetaModelUnStructuredComp for Sin and add it to a Problem
        sin_mm = om.MetaModelUnStructuredComp()
        sin_mm.add_input('x', 0., training_data=x)
        sin_mm.add_output('f_x', 0., training_data=f_x)

        prob = om.Problem()
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
        sin_mm.options['default_surrogate'] = om.KrigingSurrogate()
        prob.setup()

        surrogate = sin_mm._metadata('f_x').get('surrogate')
        self.assertTrue(isinstance(surrogate, om.KrigingSurrogate),
                        'sin_mm.f_x should get the default surrogate')

        prob['sin_mm.x'] = 2.22

        prob.run_model()

        assert_near_equal(prob['sin_mm.f_x'], .5*np.sin(prob['sin_mm.x']), 1e-4)

    def test_sin_metamodel_rmse(self):
        # create MetaModelUnStructuredComp with Kriging, using the rmse option
        sin_mm = om.MetaModelUnStructuredComp()
        sin_mm.add_input('x', 0.)
        sin_mm.add_output('f_x', 0.)

        sin_mm.options['default_surrogate'] = om.KrigingSurrogate(eval_rmse=True)

        # add it to a Problem
        prob = om.Problem()
        prob.model.add_subsystem('sin_mm', sin_mm)
        prob.setup()

        # train the surrogate and check predicted value
        sin_mm.options['train:x'] = np.linspace(0,10,20)
        sin_mm.options['train:f_x'] = np.sin(sin_mm.options['train:x'])

        prob['sin_mm.x'] = 2.1

        prob.run_model()

        assert_near_equal(prob['sin_mm.f_x'], np.sin(2.1), 1e-4) # mean
        self.assertTrue(self, sin_mm._metadata('f_x')['rmse'] < 1e-5) # std deviation

    def test_basics(self):
        # create a metamodel component
        mm = om.MetaModelUnStructuredComp()

        mm.add_input('x1', 0.)
        mm.add_input('x2', 0.)

        mm.add_output('y1', 0.)
        mm.add_output('y2', 0., surrogate=om.KrigingSurrogate())

        mm.options['default_surrogate'] = om.ResponseSurface()

        # add metamodel to a problem
        prob = om.Problem()
        prob.model.add_subsystem('mm', mm)
        prob.setup()

        # check that surrogates were properly assigned
        surrogate = mm._metadata('y1').get('surrogate')
        self.assertTrue(isinstance(surrogate, om.ResponseSurface))

        surrogate = mm._metadata('y2').get('surrogate')
        self.assertTrue(isinstance(surrogate, om.KrigingSurrogate))

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

        assert_near_equal(prob['mm.y1'], 2.0, .00001)
        assert_near_equal(prob['mm.y2'], 4.0, .00001)

        # run problem for interpolated data point and check prediction
        prob['mm.x1'] = 2.5
        prob['mm.x2'] = 3.5

        self.assertFalse(mm.train)  # training will not occur before 2nd run
        prob.run_model()

        assert_near_equal(prob['mm.y1'], 1.5934, .001)

        # change default surrogate, re-setup and check that metamodel re-trains
        mm.options['default_surrogate'] = om.KrigingSurrogate()
        prob.setup()

        surrogate = mm._metadata('y1').get('surrogate')
        self.assertTrue(isinstance(surrogate, om.KrigingSurrogate))

        self.assertTrue(mm.train)  # training will occur after re-setup

    def test_vector_inputs(self):
        mm = om.MetaModelUnStructuredComp()
        mm.add_input('x', np.zeros(4))
        mm.add_output('y1', 0.)
        mm.add_output('y2', 0.)

        mm.options['default_surrogate'] = om.KrigingSurrogate()

        prob = om.Problem()
        prob.model.add_subsystem('mm', mm)
        prob.setup()

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

        assert_near_equal(prob['mm.y1'], 1.0, .00001)
        assert_near_equal(prob['mm.y2'], 7.0, .00001)

    def test_array_inputs(self):
        mm = om.MetaModelUnStructuredComp()
        mm.add_input('x', np.zeros((2,2)))
        mm.add_output('y1', 0.)
        mm.add_output('y2', 0.)

        mm.options['default_surrogate'] = om.KrigingSurrogate()

        prob = om.Problem()
        prob.model.add_subsystem('mm', mm)
        prob.setup()

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

        assert_near_equal(prob['mm.y1'], 1.0, .00001)
        assert_near_equal(prob['mm.y2'], 7.0, .00001)

    def test_array_outputs(self):
        mm = om.MetaModelUnStructuredComp()
        mm.add_input('x', np.zeros((2, 2)))
        mm.add_output('y', np.zeros(2,))

        mm.options['default_surrogate'] = om.KrigingSurrogate()

        prob = om.Problem()
        prob.model.add_subsystem('mm', mm)
        prob.setup()

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

        assert_near_equal(prob['mm.y'], np.array([1.0, 7.0]), .00001)

    def test_2darray_outputs(self):
        mm = om.MetaModelUnStructuredComp()
        mm.add_input('x', np.zeros((2, 2)))
        mm.add_output('y', np.zeros((2, 2)))

        mm.options['default_surrogate'] = om.KrigingSurrogate()

        prob = om.Problem()
        prob.model.add_subsystem('mm', mm)
        prob.setup()

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

        assert_near_equal(prob['mm.y'], np.array([[1.0, 7.0], [1.0, 7.0]]), .00001)

    def test_unequal_training_inputs(self):
        mm = om.MetaModelUnStructuredComp()
        mm.add_input('x', 0.)
        mm.add_input('y', 0.)
        mm.add_output('f', 0.)

        mm.options['default_surrogate'] = om.KrigingSurrogate()

        prob = om.Problem()
        prob.model.add_subsystem('mm', mm)
        prob.setup()

        mm.options['train:x'] = [1.0, 1.0, 1.0, 1.0]
        mm.options['train:y'] = [1.0, 2.0]
        mm.options['train:f'] = [1.0, 1.0, 1.0, 1.0]

        prob['mm.x'] = 1.0
        prob['mm.y'] = 1.0

        with self.assertRaises(RuntimeError) as cm:
            prob.run_model()

        expected = ("MetaModelUnStructuredComp (mm): Each variable must have the same number"
                    " of training points. Expected 4 but found"
                    " 2 points for 'y'.")

        self.assertEqual(str(cm.exception), expected)

    def test_unequal_training_outputs(self):
        mm = om.MetaModelUnStructuredComp()
        mm.add_input('x', 0.)
        mm.add_input('y', 0.)
        mm.add_output('f', 0.)

        mm.options['default_surrogate'] = om.KrigingSurrogate()

        prob = om.Problem()
        prob.model.add_subsystem('mm', mm)
        prob.setup()

        mm.options['train:x'] = [1.0, 1.0, 1.0, 1.0]
        mm.options['train:y'] = [1.0, 2.0, 3.0, 4.0]
        mm.options['train:f'] = [1.0, 1.0]

        prob['mm.x'] = 1.0
        prob['mm.y'] = 1.0

        with self.assertRaises(RuntimeError) as cm:
            prob.run_model()

        expected = ("MetaModelUnStructuredComp (mm): Each variable must have the same number"
                    " of training points. Expected 4 but found"
                    " 2 points for 'f'.")
        self.assertEqual(str(cm.exception), expected)

    def test_derivatives(self):
        mm = om.MetaModelUnStructuredComp()
        mm.add_input('x', 0.)
        mm.add_output('f', 0.)

        mm.options['default_surrogate'] = om.KrigingSurrogate()

        prob = om.Problem()
        prob.model.add_subsystem('p', om.IndepVarComp('x', 0.),
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

        assert_near_equal(Jf[0][0], -1., 1.e-3)

        assert_check_partials(data, atol=1e-6, rtol=1e-6)

        # Complex step
        prob.setup(force_alloc_complex=True)
        prob.model.mm.set_check_partial_options(wrt='*', method='cs')
        data = prob.check_partials(out_stream=None)

        assert_check_partials(data, atol=1e-11, rtol=1e-11)

    def test_metamodel_feature(self):
        # create a MetaModelUnStructuredComp, specifying surrogates for the outputs
        import numpy as np

        import openmdao.api as om

        trig = om.MetaModelUnStructuredComp()

        x_train = np.linspace(0,10,20)

        trig.add_input('x', 0., training_data=x_train)

        trig.add_output('sin_x', 0.,
                        training_data=.5*np.sin(x_train),
                        surrogate=om.KrigingSurrogate())
        trig.add_output('cos_x', 0.,
                        training_data=.5*np.cos(x_train))

        trig.options['default_surrogate'] = om.KrigingSurrogate()

        # add it to a Problem, run and check the predicted values
        prob = om.Problem()
        prob.model.add_subsystem('trig', trig)
        prob.setup()

        prob.set_val('trig.x', 2.1)
        prob.run_model()

        assert_near_equal(prob.get_val('trig.sin_x'), .5*np.sin(prob.get_val('trig.x')), 1e-4)
        assert_near_equal(prob.get_val('trig.cos_x'), .5*np.cos(prob.get_val('trig.x')), 1e-4)

    def test_metamodel_feature2d(self):
        # similar to previous example, but output is 2d
        import numpy as np

        import openmdao.api as om

        # create a MetaModelUnStructuredComp that predicts sine and cosine as an array
        trig = om.MetaModelUnStructuredComp(default_surrogate=om.KrigingSurrogate())
        trig.add_input('x', 0)
        trig.add_output('y', np.zeros(2))

        # add it to a Problem
        prob = om.Problem()
        prob.model.add_subsystem('trig', trig)
        prob.setup()

        # provide training data
        trig.options['train:x'] = np.linspace(0, 10, 20)
        trig.options['train:y'] = np.column_stack((
            .5*np.sin(trig.options['train:x']),
            .5*np.cos(trig.options['train:x'])
        ))

        # train the surrogate and check predicted value
        prob.set_val('trig.x', 2.1)
        prob.run_model()
        assert_near_equal(prob.get_val('trig.y'),
                         np.append(
                             .5*np.sin(prob.get_val('trig.x')),
                             .5*np.cos(prob.get_val('trig.x'))
                         ),
                         1e-4)

    def test_vectorized(self):
        size = 3

        # create a vectorized MetaModelUnStructuredComp for sine
        trig = om.MetaModelUnStructuredComp(vec_size=size, default_surrogate=om.KrigingSurrogate())
        trig.add_input('x', np.zeros(size))
        trig.add_output('y', np.zeros(size))

        # add it to a Problem
        prob = om.Problem()
        prob.model.add_subsystem('trig', trig)
        prob.setup()

        # provide training data
        trig.options['train:x'] = np.linspace(0, 10, 20)
        trig.options['train:y'] = .5*np.sin(trig.options['train:x'])

        # train the surrogate and check predicted value
        prob['trig.x'] = np.array([2.1, 3.2, 4.3])
        prob.run_model()
        assert_near_equal(prob['trig.y'],
                         np.array(.5*np.sin(prob['trig.x'])),
                         1e-4)

        data = prob.check_partials(out_stream=None)

        assert_check_partials(data, atol=1e-6, rtol=1e-6)

    def test_vectorized_kriging(self):
        # Test for coverage (handling the rmse)
        size = 3

        # create a vectorized MetaModelUnStructuredComp for sine
        trig = om.MetaModelUnStructuredComp(vec_size=size,
                                         default_surrogate=om.KrigingSurrogate(eval_rmse=True))
        trig.add_input('x', np.zeros(size))
        trig.add_output('y', np.zeros(size))

        # add it to a Problem
        prob = om.Problem()
        prob.model.add_subsystem('trig', trig)
        prob.setup()

        # provide training data
        trig.options['train:x'] = np.linspace(0, 10, 20)
        trig.options['train:y'] = .5*np.sin(trig.options['train:x'])

        # train the surrogate and check predicted value
        prob['trig.x'] = np.array([2.1, 3.2, 4.3])
        prob.run_model()
        assert_near_equal(prob['trig.y'],
                         np.array(.5*np.sin(prob['trig.x'])),
                         1e-4)
        self.assertEqual(len(prob.model.trig._metadata('y')['rmse']), 3)

    def test_derivatives_vectorized_multiD(self):
        vec_size = 5

        mm = om.MetaModelUnStructuredComp(vec_size=vec_size)
        mm.add_input('x', np.zeros((vec_size, 2, 3)))
        mm.add_input('xx', np.zeros((vec_size, 1)))
        mm.add_output('y', np.zeros((vec_size, 4, 2)))

        mm.options['default_surrogate'] = om.KrigingSurrogate()

        prob = om.Problem()
        prob.model.add_subsystem('mm', mm)
        prob.setup()

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

        assert_check_partials(data, atol=1e-5, rtol=1e-5)

        # Complex step
        prob.setup(force_alloc_complex=True)
        prob.model.mm.set_check_partial_options(wrt='*', method='cs')
        data = prob.check_partials(out_stream=None)

        assert_check_partials(data, atol=1e-11, rtol=1e-11)

    def test_metamodel_feature_vector(self):
        # Like simple sine example, but with input of length n instead of scalar
        # The expected behavior is that the output is also of length n, with
        # each one being an independent prediction.
        # Its as if you stamped out n copies of metamodel, ran n scalars
        # through its input, then muxed all those outputs into one contiguous
        # array but you skip all the n-copies thing and do it all as an array
        import numpy as np

        import openmdao.api as om

        size = 3

        # create a vectorized MetaModelUnStructuredComp for sine
        trig = om.MetaModelUnStructuredComp(vec_size=size, default_surrogate=om.KrigingSurrogate())
        trig.add_input('x', np.zeros(size))
        trig.add_output('y', np.zeros(size))

        # add it to a Problem
        prob = om.Problem()
        prob.model.add_subsystem('trig', trig)
        prob.setup()

        # provide training data
        trig.options['train:x'] = np.linspace(0, 10, 20)
        trig.options['train:y'] = .5*np.sin(trig.options['train:x'])

        # train the surrogate and check predicted value
        prob['trig.x'] = np.array([2.1, 3.2, 4.3])
        prob.run_model()
        assert_near_equal(prob['trig.y'],
                         np.array(.5*np.sin(prob['trig.x'])),
                         1e-4)

    def test_metamodel_feature_vector2d(self):
        # similar to previous example, but processes 3 inputs/outputs at a time
        import numpy as np

        import openmdao.api as om

        size = 3

        # create a vectorized MetaModelUnStructuredComp for sine and cosine
        trig = om.MetaModelUnStructuredComp(vec_size=size, default_surrogate=om.KrigingSurrogate())
        trig.add_input('x', np.zeros(size))
        trig.add_output('y', np.zeros((size, 2)))

        # add it to a Problem
        prob = om.Problem()
        prob.model.add_subsystem('trig', trig)
        prob.setup()

        # provide training data
        trig.options['train:x'] = np.linspace(0, 10, 20)
        trig.options['train:y'] = np.column_stack((
            .5*np.sin(trig.options['train:x']),
            .5*np.cos(trig.options['train:x'])
        ))

        # train the surrogate and check predicted value
        prob.set_val('trig.x', np.array([2.1, 3.2, 4.3]))
        prob.run_model()
        assert_near_equal(prob.get_val('trig.y'),
                         np.column_stack((
                             .5*np.sin(prob.get_val('trig.x')),
                             .5*np.cos(prob.get_val('trig.x'))
                         )),
                         1e-4)

    def test_metamodel_vector_errors(self):
        # first dimension of all inputs/outputs must be 3
        mm = om.MetaModelUnStructuredComp(vec_size=3)

        with self.assertRaises(RuntimeError) as cm:
            mm.add_input('x', np.zeros(2))
        self.assertEqual(str(cm.exception),
                         "MetaModelUnStructuredComp: First dimension of input 'x' must be 3")

        with self.assertRaises(RuntimeError) as cm:
            mm.add_output('y', np.zeros(4))
        self.assertEqual(str(cm.exception),
                         "MetaModelUnStructuredComp: First dimension of output 'y' must be 3")

    def test_metamodel_subclass_optimize(self):
        class Trig(om.MetaModelUnStructuredComp):
            def setup(self):
                self.add_input('x', 0.,
                               training_data=np.linspace(0,10,20))
                self.add_output('sin_x', 0.,
                                surrogate=om.KrigingSurrogate(),
                                training_data=.5*np.sin(np.linspace(0,10,20)))

                self.declare_partials(of='sin_x', wrt='x', method='fd')

        prob = om.Problem()

        indep = om.IndepVarComp()
        indep.add_output('x', 5.)

        prob.model.add_subsystem('indep', indep)
        prob.model.add_subsystem('trig', Trig())

        prob.model.connect('indep.x', 'trig.x')

        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'COBYLA'

        prob.model.add_design_var('indep.x', lower=4, upper=7)
        prob.model.add_objective('trig.sin_x')

        prob.setup(check=True)

        self.assertEqual(prob['trig.x'], [5.])
        assert_near_equal(prob['trig.sin_x'], [.0], 1e-6)

    def test_metamodel_use_fd_if_no_surrogate_linearize(self):
        class SinSurrogate(om.SurrogateModel):
            def train(self, x, y):
                pass

            def predict(self, x):
                return sin(x)

        class SinTwoInputsSurrogate(om.SurrogateModel):
            def train(self, x, y):
                pass

            def predict(self, x):
                return sin(x[0] + x[1])

        class Trig(om.MetaModelUnStructuredComp):
            def setup(self):
                surrogate = SinSurrogate()
                self.add_input('x', 0.)
                self.add_output('sin_x', 0., surrogate=surrogate)

        class TrigWithFdInSetup(om.MetaModelUnStructuredComp):
            def setup(self):
                surrogate = SinSurrogate()
                self.add_input('x', 0.)
                self.add_output('sin_x', 0., surrogate=surrogate)
                self.declare_partials('sin_x', 'x', method='fd',
                                      form='backward', step=1e-7, step_calc='rel')

        class TrigWithCsInSetup(om.MetaModelUnStructuredComp):
            def setup(self):
                surrogate = SinSurrogate()
                self.add_input('x', 0.)
                self.add_output('sin_x', 0., surrogate=surrogate)
                self.declare_partials('sin_x', 'x', method='cs')

        class TrigGroup(om.Group):
            def configure(self):
                trig = self._get_subsystem('trig')
                trig.declare_partials('sin_x', 'x', method='fd',
                                      form='backward', step=1e-7, step_calc='rel')

        class TrigWithFdInConfigure(om.MetaModelUnStructuredComp):
            def setup(self):
                surrogate = SinSurrogate()
                self.add_input('x', 0.)
                self.add_output('sin_x', 0., surrogate=surrogate)

        class TrigTwoInputsWithFdInSetup(om.MetaModelUnStructuredComp):
            def setup(self):
                surrogate = SinTwoInputsSurrogate()
                self.add_input('x1', 0.)
                self.add_input('x2', 0.)
                self.add_output('sin_x', 0., surrogate=surrogate)
                self.declare_partials('sin_x', 'x1', method='fd',
                                      form='backward', step=1e-7, step_calc='rel')

        def no_surrogate_test_setup(trig, group=None):
            prob = om.Problem()
            if group:
                prob.model = group
            indep = om.IndepVarComp()
            indep.add_output('x', 5.)
            prob.model.add_subsystem('indep', indep)
            prob.model.add_subsystem('trig', trig)
            prob.model.connect('indep.x', 'trig.x')
            prob.setup()
            prob['indep.x'] = 5.0
            trig.train = False
            prob.run_model()
            return prob

        # Test with user not explicitly setting fd
        trig = Trig()

        msg = "Because the MetaModelUnStructuredComp 'trig' uses a surrogate which does not define a linearize method,\n" \
              "OpenMDAO will use finite differences to compute derivatives. Some of the derivatives will be computed\n" \
              "using default finite difference options because they were not explicitly declared.\n" \
              "The derivatives computed using the defaults are:\n" \
              "    trig.sin_x, trig.x\n"

        with assert_warning(RuntimeWarning, msg):
            prob = no_surrogate_test_setup(trig)

        J = prob.compute_totals(of=['trig.sin_x'], wrt=['indep.x'])
        deriv_using_fd = J[('trig.sin_x', 'indep.x')]
        assert_near_equal(deriv_using_fd[0], np.cos(prob['indep.x']), 1e-4)

        # Test with user explicitly setting fd inside of setup
        trig = TrigWithFdInSetup()
        prob = no_surrogate_test_setup(trig)
        opts = trig._subjacs_info['trig.sin_x', 'trig.x']
        expected_fd_options = {
            'step': 1e-7,
            'form': 'backward',
            'step_calc': 'rel',
        }
        for name in expected_fd_options:
            self.assertEqual(expected_fd_options[name], opts[name])
        J = prob.compute_totals(of=['trig.sin_x'], wrt=['indep.x'])
        deriv_using_fd = J[('trig.sin_x', 'indep.x')]
        assert_near_equal(deriv_using_fd[0], np.cos(prob['indep.x']), 1e-4)

        # Test with user explicitly setting fd inside of configure for a group
        trig = TrigWithFdInConfigure()
        prob = no_surrogate_test_setup(trig, group = TrigGroup())
        opts = trig._subjacs_info['trig.sin_x', 'trig.x']
        expected_fd_options = {
            'step': 1e-7,
            'form': 'backward',
            'step_calc': 'rel',
        }
        for name in expected_fd_options:
            self.assertEqual(expected_fd_options[name], opts[name])
        J = prob.compute_totals(of=['trig.sin_x'], wrt=['indep.x'])
        deriv_using_fd = J[('trig.sin_x', 'indep.x')]
        assert_near_equal(deriv_using_fd[0], np.cos(prob['indep.x']), 1e-4)

        # Test with user explicitly setting cs inside of setup. Should throw an error
        prob = om.Problem()
        indep = om.IndepVarComp()
        indep.add_output('x', 5.)
        prob.model.add_subsystem('indep', indep)
        trig = TrigWithCsInSetup()
        prob.model.add_subsystem('trig', trig)
        prob.model.connect('indep.x', 'trig.x')
        with self.assertRaises(ValueError) as context:
            prob.setup()
        expected_msg = 'Complex step has not been tested for MetaModelUnStructuredComp'
        self.assertEqual(expected_msg, str(context.exception))

        # Test with user explicitly setting fd on one of the inputs for a meta model
        #   with two inputs. Check to make sure all inputs are fd and with the correct
        #   options
        prob = om.Problem()
        indep = om.IndepVarComp()
        indep.add_output('x1', 5.)
        indep.add_output('x2', 5.)
        prob.model.add_subsystem('indep', indep)
        trig = TrigTwoInputsWithFdInSetup()
        prob.model.add_subsystem('trig', trig)
        prob.model.connect('indep.x1', 'trig.x1')
        prob.model.connect('indep.x2', 'trig.x2')
        prob.setup()
        prob['indep.x1'] = 5.0
        prob['indep.x2'] = 5.0
        trig.train = False

        msg = "Because the MetaModelUnStructuredComp 'trig' uses a surrogate which does not define a linearize method,\n" \
              "OpenMDAO will use finite differences to compute derivatives. Some of the derivatives will be computed\n" \
              "using default finite difference options because they were not explicitly declared.\n" \
              "The derivatives computed using the defaults are:\n" \
              "    trig.sin_x, trig.x2\n"

        with assert_warning(RuntimeWarning, msg):
            prob.run_model()

        self.assertEqual('fd', trig._subjacs_info[('trig.sin_x', 'trig.x1')]['method'])
        self.assertEqual('backward', trig._subjacs_info[('trig.sin_x', 'trig.x1')]['form'])
        self.assertEqual(1e-07, trig._subjacs_info[('trig.sin_x', 'trig.x1')]['step'])
        self.assertEqual('rel', trig._subjacs_info[('trig.sin_x', 'trig.x1')]['step_calc'])

        self.assertEqual('fd', trig._subjacs_info[('trig.sin_x', 'trig.x2')]['method'])
        self.assertTrue('form' not in trig._subjacs_info[('trig.sin_x', 'trig.x2')])
        self.assertTrue('step' not in trig._subjacs_info[('trig.sin_x', 'trig.x2')])
        self.assertTrue('step_calc' not in trig._subjacs_info[('trig.sin_x', 'trig.x2')])

    def test_feature_metamodel_use_fd_if_no_surrogate_linearize(self):
        from math import sin
        import openmdao.api as om

        class SinSurrogate(om.SurrogateModel):
            def train(self, x, y):
                pass

            def predict(self, x):
                return sin(x)

        class TrigWithFdInSetup(om.MetaModelUnStructuredComp):
            def setup(self):
                surrogate = SinSurrogate()
                self.add_input('x', 0.)
                self.add_output('sin_x', 0., surrogate=surrogate)
                self.declare_partials('sin_x', 'x', method='fd',
                                      form='backward', step=1e-7, step_calc='rel')

        # Testing explicitly setting fd inside of setup
        prob = om.Problem()
        trig = TrigWithFdInSetup()
        prob.model.add_subsystem('trig', trig, promotes_inputs=['x'])
        prob.setup(check=True)
        prob.set_val('x', 5.)
        trig.train = False
        prob.run_model()
        J = prob.compute_totals(of=['trig.sin_x'], wrt=['x'])
        deriv_using_fd = J[('trig.sin_x', 'x')]
        assert_near_equal(deriv_using_fd[0], np.cos(prob['x']), 1e-4)

    def test_metamodel_setup_called_twice_bug(self):
        class Trig(om.MetaModelUnStructuredComp):
            def setup(self):
                surrogate = om.NearestNeighbor()
                self.add_input('x', 0.,
                               training_data=np.linspace(0, 10, 20))
                self.add_output('sin_x', 0.,
                                surrogate=surrogate,
                                training_data=.5 * np.sin(np.linspace(0, 10, 20)))

        # Check to make sure bug reported in story 160200719 is fixed
        prob = om.Problem()

        indep = om.IndepVarComp()
        indep.add_output('x', 5.)

        prob.model.add_subsystem('indep', indep)
        prob.model.add_subsystem('trig', Trig())

        prob.model.connect('indep.x', 'trig.x')

        prob.model.add_design_var('indep.x', lower=4, upper=7)
        prob.model.add_objective('trig.sin_x')

        prob.setup()
        prob['indep.x'] = 5.0
        prob.run_model()
        J = prob.compute_totals()
        # First value.
        deriv_first_time = J[('trig.sin_x', 'indep.x')]

        # Setup and run a second time
        prob.setup()
        prob['indep.x'] = 5.0
        prob.run_model()
        J = prob.compute_totals()
        # Second time.
        deriv_second_time = J[('trig.sin_x', 'indep.x')]

        assert_near_equal(deriv_first_time, deriv_second_time, 1e-4)

    def test_metamodel_setup_called_twice_bug_called_outside_setup(self):
        class Trig(om.MetaModelUnStructuredComp):
            def __init__(self):
                super(Trig, self).__init__()
                self.add_input('x', 0.,
                               training_data=np.linspace(0, 10, 20))

            def setup(self):
                surrogate = om.NearestNeighbor()
                self.add_output('sin_x', 0.,
                                surrogate=surrogate,
                                training_data=.5 * np.sin(np.linspace(0, 10, 20)))

        prob = om.Problem()

        indep = om.IndepVarComp()
        indep.add_output('x', 5.)

        prob.model.add_subsystem('indep', indep)
        trig = Trig()
        prob.model.add_subsystem('trig', trig)

        prob.model.connect('indep.x', 'trig.x')

        prob.model.add_design_var('indep.x', lower=4, upper=7)
        prob.model.add_objective('trig.sin_x')

        # Check to make sure bug reported in story 160200719 is fixed
        prob.setup()
        prob['indep.x'] = 5.0
        prob.run_model()
        J = prob.compute_totals()
        # First value.
        deriv_first_time = J[('trig.sin_x', 'indep.x')]

        # Setup and run a second time
        prob.setup()
        prob['indep.x'] = 5.0
        prob.run_model()
        J = prob.compute_totals()
        # Second time.
        deriv_second_time = J[('trig.sin_x', 'indep.x')]

        assert_near_equal(deriv_first_time, deriv_second_time, 1e-4)

    def test_warning_bug(self):
        # Make sure we don't warn that we are doing FD when the surrogate has analytic derivs.

        x_train = np.arange(0., 10.)
        y_train = np.arange(10., 20.)
        z_train = x_train**2 + y_train**2

        p = om.Problem()
        p.model = m = om.Group()

        params = om.IndepVarComp()
        params.add_output('x', val=0.)
        params.add_output('y', val=0.)

        m.add_subsystem('params', params, promotes=['*'])

        sm = om.MetaModelUnStructuredComp(default_surrogate=om.ResponseSurface())
        sm.add_input('x', val=0.)
        sm.add_input('y', val=0.)
        sm.add_output('z', val=0.)

        sm.options['train:x'] = x_train
        sm.options['train:y'] = y_train
        sm.options['train:z'] = z_train

        # With or without the line below does not matter
        # Only when method is set to fd, then RuntimeWarning disappears
        sm.declare_partials('*', '*', method='exact')

        m.add_subsystem('sm', sm, promotes=['*'])

        m.add_design_var('x', lower=0., upper=10.)
        m.add_design_var('y', lower=0., upper=10.)
        m.add_objective('z')

        p.setup(check=True)

        stderr = sys.stderr
        str_err = StringIO()
        sys.stderr = str_err
        try:
            p.final_setup()
        finally:
            sys.stderr = stderr

        output = str_err.getvalue()
        self.assertTrue('finite difference' not in output)

    def test_surrogate_message_format(self):
        prob = om.Problem()

        prob.model.add_subsystem('p', om.IndepVarComp('x', 2.1))

        sin_mm = om.MetaModelUnStructuredComp()
        sin_mm.add_input('x', 0.)
        sin_mm.add_output('f_x', 0., surrogate=om.KrigingSurrogate())

        prob.model.add_subsystem('sin_mm', sin_mm)

        prob.model.connect('p.x', 'sin_mm.x')

        prob.setup(check=True)

        # train the surrogate and check predicted value
        sin_mm.options['train:x'] = np.linspace(0,10,1)
        sin_mm.options['train:f_x'] = .5*np.sin(sin_mm.options['train:x'])

        prob['sin_mm.x'] = 2.1

        with self.assertRaises(ValueError) as cm:
            prob.run_model()

        self.assertEqual(str(cm.exception), 'sin_mm: KrigingSurrogate requires at least'
                                            ' 2 training points.')


class MetaModelUnstructuredSurrogatesFeatureTestCase(unittest.TestCase):

    def test_kriging(self):
        import numpy as np

        import openmdao.api as om

        prob = om.Problem()

        sin_mm = om.MetaModelUnStructuredComp()
        sin_mm.add_input('x', 2.1)
        sin_mm.add_output('f_x', 0., surrogate=om.KrigingSurrogate())

        prob.model.add_subsystem('sin_mm', sin_mm)

        prob.setup(check=True)

        # train the surrogate and check predicted value
        sin_mm.options['train:x'] = np.linspace(0,10,20)
        sin_mm.options['train:f_x'] = .5*np.sin(sin_mm.options['train:x'])

        prob.set_val('sin_mm.x', 2.1)

        prob.run_model()

        assert_near_equal(prob.get_val('sin_mm.f_x'), .5*np.sin(prob.get_val('sin_mm.x')), 1e-4)

    def test_nearest_neighbor(self):
        import numpy as np

        import openmdao.api as om

        prob = om.Problem()

        sin_mm = om.MetaModelUnStructuredComp()
        sin_mm.add_input('x', 2.1)
        sin_mm.add_output('f_x', 0., surrogate=om.NearestNeighbor(interpolant_type='linear'))

        prob.model.add_subsystem('sin_mm', sin_mm)

        prob.setup(check=True)

        # train the surrogate and check predicted value
        sin_mm.options['train:x'] = np.linspace(0,10,20)
        sin_mm.options['train:f_x'] = .5*np.sin(sin_mm.options['train:x'])

        prob.set_val('sin_mm.x', 2.1)

        prob.run_model()

        assert_near_equal(prob.get_val('sin_mm.f_x'), .5*np.sin(prob.get_val('sin_mm.x')), 2e-3)

    def test_response_surface(self):
        import numpy as np

        import openmdao.api as om

        prob = om.Problem()

        sin_mm = om.MetaModelUnStructuredComp()
        sin_mm.add_input('x', 2.1)
        sin_mm.add_output('f_x', 0., surrogate=om.ResponseSurface())

        prob.model.add_subsystem('sin_mm', sin_mm)

        prob.setup(check=True)

        # train the surrogate and check predicted value
        sin_mm.options['train:x'] = np.linspace(0, 3.14, 20)
        sin_mm.options['train:f_x'] = .5*np.sin(sin_mm.options['train:x'])

        prob.set_val('sin_mm.x', 2.1)

        prob.run_model()

        assert_near_equal(prob.get_val('sin_mm.f_x'), .5*np.sin(prob.get_val('sin_mm.x')), 2e-3)

    def test_kriging_options_eval_rmse(self):
        import numpy as np

        import openmdao.api as om

        prob = om.Problem()

        sin_mm = om.MetaModelUnStructuredComp()
        sin_mm.add_input('x', 2.1)
        sin_mm.add_output('f_x', 0., surrogate=om.KrigingSurrogate(eval_rmse=True))

        prob.model.add_subsystem('sin_mm', sin_mm)

        prob.setup(check=True)

        # train the surrogate and check predicted value
        sin_mm.options['train:x'] = np.linspace(0,10,20)
        sin_mm.options['train:f_x'] = .5*np.sin(sin_mm.options['train:x'])

        prob.set_val('sin_mm.x', 2.1)

        prob.run_model()

        print("mean")
        assert_near_equal(prob.get_val('sin_mm.f_x'), .5*np.sin(prob.get_val('sin_mm.x')), 1e-4)
        print("std")
        assert_near_equal(sin_mm._metadata('f_x')['rmse'][0, 0], 0.0, 1e-4)

    def test_nearest_neighbor_rbf_options(self):
        import numpy as np

        import openmdao.api as om

        prob = om.Problem()

        sin_mm = om.MetaModelUnStructuredComp()
        sin_mm.add_input('x', 2.1)
        sin_mm.add_output('f_x', 0., surrogate=om.NearestNeighbor(interpolant_type='rbf', num_neighbors=3))

        prob.model.add_subsystem('sin_mm', sin_mm)

        prob.setup(check=True)

        # train the surrogate and check predicted value
        sin_mm.options['train:x'] = np.linspace(0,10,20)
        sin_mm.options['train:f_x'] = .5*np.sin(sin_mm.options['train:x'])

        prob.set_val('sin_mm.x', 2.1)

        prob.run_model()

        assert_near_equal(prob.get_val('sin_mm.f_x'), .5*np.sin(prob.get_val('sin_mm.x')), 5e-3)


if __name__ == "__main__":
    unittest.main()
