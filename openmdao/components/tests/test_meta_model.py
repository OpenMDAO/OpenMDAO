import numpy as np
import unittest

from openmdao.api import Group, Problem, MetaModel, IndepVarComp, ResponseSurface, \
    FloatKrigingSurrogate, KrigingSurrogate
from openmdao.devtools.testutil import assert_rel_error

from six.moves import cStringIO
from re import findall


class TestMetaModel(unittest.TestCase):

    def test_sin_metamodel(self):

        # create a MetaModel for Sin and add it to a Problem
        sin_mm = MetaModel()
        sin_mm.add_input('x', 0.)
        sin_mm.add_output('f_x', 0.)

        prob = Problem(Group())
        prob.model.add_subsystem('sin_mm', sin_mm)

        # check that missing surrogate is detected in check_setup
        # stream = cStringIO()
        # prob.setup(out_stream=stream)
        # msg = ("No default surrogate model is defined and the "
        #        "following outputs do not have a surrogate model:\n"
        #        "['f_x']\n"
        #        "Either specify a default_surrogate, or specify a "
        #        "surrogate model for all outputs.")
        # self.assertTrue(msg in stream.getvalue())

        # check that output with no specified surrogate gets the default
        sin_mm.default_surrogate = FloatKrigingSurrogate()
        prob.setup(check=False)
        surrogate = sin_mm._var_abs2meta['output']['sin_mm.f_x'].get('surrogate')
        print "surrogate:", surrogate
        self.assertTrue(isinstance(surrogate, FloatKrigingSurrogate),
                        'sin_mm.f_x should get the default surrogate')

        # train the surrogate and check predicted value
        prob['sin_mm.train:x'] = np.linspace(0,10,20)
        prob['sin_mm.train:f_x'] = .5*np.sin(prob['sin_mm.train:x'])

        prob['sin_mm.x'] = 2.1

        prob.run()

        self.assertAlmostEqual(prob['sin_mm.f_x'],
                               .5*np.sin(prob['sin_mm.x']),
                               places=4)

    def test_sin_metamodel_preset_data(self):
        # preset training data
        x = np.linspace(0,10,200)
        f_x = .5*np.sin(x)

        # create a MetaModel for Sin and add it to a Problem
        sin_mm = MetaModel()
        sin_mm.add_input('x', 0., training_data = np.linspace(0,10,200))
        sin_mm.add_output('f_x', 0., training_data=f_x)

        prob = Problem(Group())
        prob.model.add_subsystem('sin_mm', sin_mm)

        # check that missing surrogate is detected in check_setup
        stream = cStringIO()
        prob.setup(out_stream=stream)
        msg = ("No default surrogate model is defined and the "
               "following outputs do not have a surrogate model:\n"
               "['f_x']\n"
               "Either specify a default_surrogate, or specify a "
               "surrogate model for all outputs.")
        self.assertTrue(msg in stream.getvalue())

        # check that output with no specified surrogate gets the default
        sin_mm.default_surrogate = FloatKrigingSurrogate()
        prob.setup(check=False)
        surrogate = prob.model.outputs.metadata('sin_mm.f_x').get('surrogate')
        self.assertTrue(isinstance(surrogate, FloatKrigingSurrogate),
                        'sin_mm.f_x should get the default surrogate')

        prob['sin_mm.x'] = 2.22

        prob.run()

        self.assertAlmostEqual(prob['sin_mm.f_x'],
                               .5*np.sin(prob['sin_mm.x']),
                               places=4)

    def test_sin_metamodel_obj_return(self):

        # create a MetaModel for Sin and add it to a Problem
        sin_mm = MetaModel()
        sin_mm.add_input('x', 0.)
        sin_mm.add_output('f_x', (0.,0.))

        prob = Problem(Group())
        prob.model.add_subsystem('sin_mm', sin_mm)

        # check that missing surrogate is detected in check_setup
        stream = cStringIO()
        prob.setup(out_stream=stream)
        msg = ("No default surrogate model is defined and the "
               "following outputs do not have a surrogate model:\n"
               "['f_x']\n"
               "Either specify a default_surrogate, or specify a "
               "surrogate model for all outputs.")
        self.assertTrue(msg in stream.getvalue())

        # check that output with no specified surrogate gets the default
        sin_mm.default_surrogate = KrigingSurrogate(eval_rmse=True)
        prob.setup(check=False)
        surrogate = prob.model.outputs.metadata('sin_mm.f_x').get('surrogate')
        self.assertTrue(isinstance(surrogate, KrigingSurrogate),
                        'sin_mm.f_x should get the default surrogate')

        # train the surrogate and check predicted value
        prob['sin_mm.train:x'] = np.linspace(0,10,20)
        prob['sin_mm.train:f_x'] = np.sin(prob['sin_mm.train:x'])

        prob['sin_mm.x'] = 2.1

        prob.run()
        assert_rel_error(self, prob['sin_mm.f_x'][0], np.sin(2.1), 1e-4) # mean
        self.assertTrue(self, prob['sin_mm.f_x'][1] < 1e-5) #std deviation

    def test_basics(self):
        # create a metamodel component
        mm = MetaModel()

        mm.add_input('x1', 0.)
        mm.add_input('x2', 0.)

        mm.add_output('y1', 0.)
        mm.add_output('y2', 0., surrogate=FloatKrigingSurrogate())

        mm.default_surrogate = ResponseSurface()

        # add metamodel to a problem
        prob = Problem(model=Group())
        prob.model.add_subsystem('mm', mm)
        prob.setup(check=False)

        # check that surrogates were properly assigned
        surrogate = prob.model.outputs.metadata('mm.y1').get('surrogate')
        self.assertTrue(isinstance(surrogate, ResponseSurface))

        surrogate = prob.model.outputs.metadata('mm.y2').get('surrogate')
        self.assertTrue(isinstance(surrogate, FloatKrigingSurrogate))

        # populate training data
        prob['mm.train:x1'] = [1.0, 2.0, 3.0]
        prob['mm.train:x2'] = [1.0, 3.0, 4.0]
        prob['mm.train:y1'] = [3.0, 2.0, 1.0]
        prob['mm.train:y2'] = [1.0, 4.0, 7.0]

        # run problem for provided data point and check prediction
        prob['mm.x1'] = 2.0
        prob['mm.x2'] = 3.0

        self.assertTrue(mm.train)   # training will occur before 1st run
        prob.run()

        assert_rel_error(self, prob['mm.y1'], 2.0, .00001)
        assert_rel_error(self, prob['mm.y2'], 4.0, .00001)

        # run problem for interpolated data point and check prediction
        prob['mm.x1'] = 2.5
        prob['mm.x2'] = 3.5

        self.assertFalse(mm.train)  # training will not occur before 2nd run
        prob.run()

        assert_rel_error(self, prob['mm.y1'], 1.5934, .001)

        # change default surrogate, re-setup and check that metamodel re-trains
        mm.default_surrogate = FloatKrigingSurrogate()
        prob.setup(check=False)

        surrogate = prob.model.outputs.metadata('mm.y1').get('surrogate')
        self.assertTrue(isinstance(surrogate, FloatKrigingSurrogate))

        self.assertTrue(mm.train)  # training will occur after re-setup
        mm.warm_restart = True     # use existing training data

        prob['mm.x1'] = 2.5
        prob['mm.x2'] = 3.5

        prob.run()
        assert_rel_error(self, prob['mm.y1'], 1.5, 1e-2)

    def test_warm_start(self):
        # create metamodel with warm_restart = True
        meta = MetaModel()
        meta.add_input('x1', 0.)
        meta.add_input('x2', 0.)
        meta.add_output('y1', 0.)
        meta.add_output('y2', 0.)
        meta.default_surrogate = ResponseSurface()
        meta.warm_restart = True

        # add to problem
        prob = Problem(Group())
        prob.model.add_subsystem('meta', meta)
        prob.setup(check=False)

        # provide initial training data
        prob['meta.train:x1'] = [1.0, 3.0]
        prob['meta.train:x2'] = [1.0, 4.0]
        prob['meta.train:y1'] = [3.0, 1.0]
        prob['meta.train:y2'] = [1.0, 7.0]

        # run against a data point and check result
        prob['meta.x1'] = 2.0
        prob['meta.x2'] = 3.0
        prob.run()

        assert_rel_error(self, prob['meta.y1'], 1.9085, .001)
        assert_rel_error(self, prob['meta.y2'], 3.9203, .001)

        # Add 3rd training point, moves the estimate for that point
        # back to where it should be.
        prob['meta.train:x1'] = [2.0]
        prob['meta.train:x2'] = [3.0]
        prob['meta.train:y1'] = [2.0]
        prob['meta.train:y2'] = [4.0]

        meta.train = True  # currently need to tell meta to re-train

        prob.run()
        assert_rel_error(self, prob['meta.y1'], 2.0, .00001)
        assert_rel_error(self, prob['meta.y2'], 4.0, .00001)

    def test_vector_inputs(self):

        meta = MetaModel()
        meta.add_input('x', np.zeros(4))
        meta.add_output('y1', 0.)
        meta.add_output('y2', 0.)
        meta.default_surrogate = FloatKrigingSurrogate()

        prob = Problem(Group())
        prob.model.add_subsystem('meta', meta)
        prob.setup(check=False)

        prob['meta.train:x'] = [
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 1.0, 1.0, 1.0],
            [1.0, 2.0, 1.0, 1.0],
            [1.0, 1.0, 2.0, 1.0],
            [1.0, 1.0, 1.0, 2.0]
        ]
        prob['meta.train:y1'] = [3.0, 2.0, 1.0, 6.0, -2.0]
        prob['meta.train:y2'] = [1.0, 4.0, 7.0, -3.0, 3.0]

        prob['meta.x'] = [1.0, 2.0, 1.0, 1.0]
        prob.run()

        assert_rel_error(self, prob['meta.y1'], 1.0, .00001)
        assert_rel_error(self, prob['meta.y2'], 7.0, .00001)

    def test_array_inputs(self):
        meta = MetaModel()
        meta.add_input('x', np.zeros((2,2)))
        meta.add_output('y1', 0.)
        meta.add_output('y2', 0.)
        meta.default_surrogate = FloatKrigingSurrogate()

        prob = Problem(Group())
        prob.model.add_subsystem('meta', meta)
        prob.setup(check=False)

        prob['meta.train:x'] = [
            [[1.0, 1.0], [1.0, 1.0]],
            [[2.0, 1.0], [1.0, 1.0]],
            [[1.0, 2.0], [1.0, 1.0]],
            [[1.0, 1.0], [2.0, 1.0]],
            [[1.0, 1.0], [1.0, 2.0]]
        ]
        prob['meta.train:y1'] = [3.0, 2.0, 1.0, 6.0, -2.0]
        prob['meta.train:y2'] = [1.0, 4.0, 7.0, -3.0, 3.0]

        prob['meta.x'] = [[1.0, 2.0], [1.0, 1.0]]
        prob.run()

        assert_rel_error(self, prob['meta.y1'], 1.0, .00001)
        assert_rel_error(self, prob['meta.y2'], 7.0, .00001)

    def test_array_outputs(self):
        meta = MetaModel()
        meta.add_input('x', np.zeros((2, 2)))
        meta.add_output('y', np.zeros(2,))
        meta.default_surrogate = FloatKrigingSurrogate()

        prob = Problem(Group())
        prob.model.add_subsystem('meta', meta)
        prob.setup(check=False)

        prob['meta.train:x'] = [
            [[1.0, 1.0], [1.0, 1.0]],
            [[2.0, 1.0], [1.0, 1.0]],
            [[1.0, 2.0], [1.0, 1.0]],
            [[1.0, 1.0], [2.0, 1.0]],
            [[1.0, 1.0], [1.0, 2.0]]
        ]

        prob['meta.train:y'] = [[3.0, 1.0],
                                [2.0, 4.0],
                                [1.0, 7.0],
                                [6.0, -3.0],
                                [-2.0, 3.0]]

        prob['meta.x'] = [[1.0, 2.0], [1.0, 1.0]]
        prob.run()

        assert_rel_error(self, prob['meta.y'], np.array([1.0, 7.0]), .00001)

    def test_unequal_training_inputs(self):

        meta = MetaModel()
        meta.add_input('x', 0.)
        meta.add_input('y', 0.)
        meta.add_output('f', 0.)
        meta.default_surrogate = FloatKrigingSurrogate()

        prob = Problem(Group())
        prob.model.add_subsystem('meta', meta)
        prob.setup(check=False)

        prob['meta.train:x'] = [1.0, 1.0, 1.0, 1.0]
        prob['meta.train:y'] = [1.0, 2.0]
        prob['meta.train:f'] = [1.0, 1.0, 1.0, 1.0]

        prob['meta.x'] = 1.0
        prob['meta.y'] = 1.0

        with self.assertRaises(RuntimeError) as cm:
            prob.run()

        expected = "MetaModel: Each variable must have the same number" \
                   " of training points. Expected 4 but found" \
                   " 2 points for 'y'."

        self.assertEqual(str(cm.exception), expected)

    def test_unequal_training_outputs(self):
        meta = MetaModel()
        meta.add_input('x', 0.)
        meta.add_input('y', 0.)
        meta.add_output('f', 0.)
        meta.default_surrogate = FloatKrigingSurrogate()

        prob = Problem(Group())
        prob.model.add_subsystem('meta', meta)
        prob.setup(check=False)

        prob['meta.train:x'] = [1.0, 1.0, 1.0, 1.0]
        prob['meta.train:y'] = [1.0, 2.0, 3.0, 4.0]
        prob['meta.train:f'] = [1.0, 1.0]

        prob['meta.x'] = 1.0
        prob['meta.y'] = 1.0

        with self.assertRaises(RuntimeError) as cm:
            prob.run()

        expected = "MetaModel: Each variable must have the same number" \
                   " of training points. Expected 4 but found" \
                   " 2 points for 'f'."

        self.assertEqual(str(cm.exception), expected)

    def test_derivatives(self):
        meta = MetaModel()
        meta.add_input('x', 0.)
        meta.add_output('f', 0.)
        meta.default_surrogate = FloatKrigingSurrogate()

        prob = Problem(Group())
        prob.model.add_subsystem('meta', meta, promotes=['x'])
        prob.model.add_subsystem('p', IndepVarComp('x', 0.), promotes=['x'])
        prob.setup(check=False)

        prob['meta.train:x'] = [0., .25, .5, .75, 1.]
        prob['meta.train:f'] = [1., .75, .5, .25, 0.]
        prob['x'] = 0.125
        prob.run()

        Jf = prob.calc_gradient(['x'], ['meta.f'], mode='fwd')
        Jr = prob.calc_gradient(['x'], ['meta.f'], mode='rev')

        assert_rel_error(self, Jf[0][0], -1., 1.e-3)
        assert_rel_error(self, Jr[0][0], -1., 1.e-3)

        stream = cStringIO()
        prob.check_partial_derivatives(out_stream=stream, global_options={'check_type': 'cs'})

        abs_errors = findall('Absolute Error \(.+\) : (.+)', stream.getvalue())
        self.assertTrue(len(abs_errors) > 0)
        for match in abs_errors:
            abs_error = float(match)
            self.assertTrue(abs_error < 1.e-6)

if __name__ == "__main__":
    unittest.main()
