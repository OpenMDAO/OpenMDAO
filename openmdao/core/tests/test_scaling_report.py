"""Define the scaling report tests."""
import unittest

import numpy as np
import openmdao.api as om

from openmdao.utils.testing_utils import use_tempdirs
from openmdao.utils.assert_utils import assert_near_equal


class ExecCompVOI(om.ExecComp):
    # adds all of its inputs as DVs and all of its outputs as constraints
    def setup(self):
        super().setup()

        # add design vars
        rel2meta = self._var_rel2meta
        for name in sorted(self._var_rel_names['input']):
            meta = rel2meta[name]
            self.add_design_var(name, units=meta['units'])

        # add constraints
        for name in sorted(self._var_rel_names['output']):
            meta = rel2meta[name]
            self.add_constraint(name, lower=0., upper=10., units=meta['units'])


class _Obj(object):
    def __init__(self):
        self.dvs = []
        self.cons = []
        self.objs = []


@use_tempdirs
class TestDriverScalingReport(unittest.TestCase):

    def setup_model(self, mult_exp_range=(-10, 4), nins=1, nouts=1, ncomps=10, shape=1):
        assert nins >= 1
        assert nouts >= 1
        assert ncomps >= 1
        assert mult_exp_range[1] > mult_exp_range[0]

        inidxs = np.arange(nins)
        outidxs = np.arange(nouts)

        expected = _Obj()
        expected.objs.append("objective_comp.out")

        p = om.Problem(allow_post_setup_reorder=False)
        model = p.model
        for icomp in range(ncomps):
            exprs = []
            for iout in range(nouts):

                inperm = np.random.permutation(inidxs)
                imults = "+".join([f"in{i} * 10**{(mult_exp_range[1] - mult_exp_range[0]) * np.random.random() + mult_exp_range[0]}" for
                          i in inperm])
                exprs.append(f"out{iout} = {imults}")

            model.add_subsystem(f"comp{icomp}", ExecCompVOI(exprs, shape=shape, do_coloring=False))

            if icomp == 0:
                # add a comp for the objective
                model.add_subsystem("objective_comp", om.ExecComp("out=inp * 2", shape=shape))
                model.add_objective("objective_comp.out")
                model.connect("comp0.out0", "objective_comp.inp")

            s_ins = sorted(f"in{i}" for i in inidxs)
            expected.dvs.extend(f"comp{icomp}.{n}" for n in s_ins)
            s_outs = sorted(f"out{i}" for i in outidxs)
            expected.cons.extend(f"comp{icomp}.{n}" for n in s_outs)

        p.setup()
        return p, expected

    def _check_data(self, data, expected):
        objs = data['oflabels'][0]
        cons = data['oflabels'][1:]
        self.assertEqual(expected.objs, [objs])
        self.assertEqual(expected.cons, cons)
        self.assertEqual(expected.dvs, data['wrtlabels'])
        self.assertEqual(0, len(data['linear']['oflabels']))
        self.assertEqual(len(expected.objs), len(data['obj_table']))
        self.assertEqual(len(expected.cons), len(data['con_table']))
        self.assertEqual(len(expected.dvs), len(data['dv_table']))
        for dvrow in data['dv_table']:
            if dvrow['size'] > 1:
                self.assertEqual(dvrow['size'], len(dvrow['_children']))
                for chrow in dvrow['_children']:
                    self.assertEqual('', chrow['size'])
                    self.assertTrue('_children' not in chrow)

        for conrow in data['con_table']:
            if conrow['size'] > 1:
                self.assertEqual(conrow['size'], len(conrow['_children']))
                for chrow in conrow['_children']:
                    self.assertEqual('', chrow['size'])
                    self.assertTrue('_children' not in chrow)

        for objrow in data['obj_table']:
            if objrow['size'] > 1:
                self.assertEqual(objrow['size'], len(objrow['_children']))
                for chrow in objrow['_children']:
                    self.assertEqual('', chrow['size'])
                    self.assertTrue('_children' not in chrow)

    def test_40x40_in1_out1(self):
        p, expected = self.setup_model(ncomps=40)
        p.final_setup()
        # compute dict totals to make sure we handle that properly
        p.driver._compute_totals()
        data = p.driver.scaling_report(show_browser=False)
        self._check_data(data, expected)

    def test_40x40_in4_out4(self):
        p, expected = self.setup_model(nins=4, nouts=4, ncomps=10)
        p.final_setup()
        data = p.driver.scaling_report(show_browser=False)
        self._check_data(data, expected)

    def test_40x40_in4_out4_shape10(self):
        p, expected = self.setup_model(nins=4, nouts=4, ncomps=10, shape=10)
        p.final_setup()
        data = p.driver.scaling_report(show_browser=False)
        self._check_data(data, expected)

    def test_in100out4_shape10(self):
        p, expected = self.setup_model(nins=100, nouts=4, ncomps=1, shape=10)
        p.final_setup()
        data = p.driver.scaling_report(show_browser=False)
        self._check_data(data, expected)

    def test_in4out100_shape10(self):
        p, expected = self.setup_model(nins=4, nouts=100, ncomps=1, shape=10)
        p.final_setup()
        data = p.driver.scaling_report(show_browser=False)
        self._check_data(data, expected)

    def test_big_subjac(self):
        # this gets slow with larger sizes, e.g. shape=1000 takes > 20 sec to
        # show/hide table rows or show a subjac in the browser
        p, expected = self.setup_model(nins=4, nouts=4, ncomps=1, shape=100)
        p.final_setup()
        data = p.driver.scaling_report(show_browser=False)
        self._check_data(data, expected)


@use_tempdirs
class TestDriverScalingReport2(unittest.TestCase):

    def test_unconstrained(self):
        from openmdao.test_suite.components.paraboloid import Paraboloid

        # build the model
        prob = om.Problem()
        indeps = prob.model.add_subsystem('indeps', om.IndepVarComp())
        indeps.add_output('x', 3.0)
        indeps.add_output('y', -4.0)

        prob.model.add_subsystem('paraboloid', Paraboloid())

        prob.model.connect('indeps.x', 'paraboloid.x')
        prob.model.connect('indeps.y', 'paraboloid.y')

        # setup the optimization
        prob.driver = om.ScipyOptimizeDriver(optimizer='COBYLA')

        prob.model.add_design_var('indeps.x', lower=-50, upper=50)
        prob.model.add_design_var('indeps.y', lower=-50, upper=50)
        prob.model.add_objective('paraboloid.f_xy', index=0)

        prob.setup()
        prob.run_driver()

        # minimum value
        assert_near_equal(prob['paraboloid.f_xy'], -27.33333, 1e-6)

        # location of the minimum
        assert_near_equal(prob['indeps.x'], 6.6667, 1e-4)
        assert_near_equal(prob['indeps.y'], -7.33333, 1e-4)

        # just make sure this doesn't raise an exception
        prob.driver.scaling_report(show_browser=False)


@use_tempdirs
class TestDriverScalingReport3(unittest.TestCase):
    def test_setup_message(self):
        x_train = np.arange(0., 10.)
        y_train = np.arange(10., 20.)
        z_train = x_train ** 2 + y_train ** 2

        p = om.Problem()
        p.model = model = om.Group()

        params = om.IndepVarComp()
        params.add_output('x', val=0.)
        params.add_output('y', val=0.)

        model.add_subsystem('params', params, promotes=['*'])

        sm = om.MetaModelUnStructuredComp(default_surrogate=om.ResponseSurface())
        sm.add_input('x', val=0.)
        sm.add_input('y', val=0.)
        sm.add_output('z', val=0.)

        sm.options['train_x'] = x_train
        sm.options['train_y'] = y_train
        sm.options['train_z'] = z_train

        # With or without the line below does not matter
        # Only when method is set to fd, then RuntimeWarning disappears
        sm.declare_partials('*', '*', method='exact')

        model.add_subsystem('sm', sm, promotes=['*'])

        model.add_design_var('x', lower=0., upper=10.)
        model.add_design_var('y', lower=0., upper=10.)
        model.add_objective('z')

        p.setup()

        with self.assertRaises(RuntimeError) as cm:
            p.driver.scaling_report()

        msg = "Either 'run_model' or 'final_setup' must be called before the scaling report can be generated."
        self.assertEqual(str(cm.exception), msg)

        # Now, make sure it runs run_model automatically as long as we final_setup.
        p.final_setup()
        p.driver.scaling_report(show_browser=False)


@use_tempdirs
class TestDiscreteScalingReport(unittest.TestCase):

    def test_scaling_report(self):
        class ParaboloidDiscrete(om.ExplicitComponent):

            def setup(self):
                self.add_discrete_input('x', val=10)
                self.add_discrete_input('y', val=0)
                self.add_discrete_output('f_xy', val=0)

            def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
                x = discrete_inputs['x']
                y = discrete_inputs['y']
                f_xy = (x - 3.0)**2 + x * y + (y + 4.0)**2 - 3.0
                discrete_outputs['f_xy'] = int(f_xy)

        prob = om.Problem()
        model = prob.model

        # Add independent variables
        indeps = model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
        indeps.add_discrete_output('x', 4)
        indeps.add_discrete_output('y', 3)

        # Add components
        model.add_subsystem('parab', ParaboloidDiscrete(), promotes=['*'])

        # Specify design variable range and objective
        model.add_design_var('x')
        model.add_design_var('y')
        model.add_objective('f_xy')

        samples = [
            [('x', 5), ('y', 1)],
            [('x', 3), ('y', 6)],
            [('x', -1), ('y', 3)],
        ]

        # Setup driver
        prob.driver = om.DOEDriver(om.ListGenerator(samples))

        # run driver
        prob.setup()
        prob.run_driver()
        prob.cleanup()

        # generate scaling report
        prob.driver.scaling_report(show_browser=False)

    def test_bug_2494(self):
        # see Issue #2494, bug in scaling_report was causing damaging side effect
        class OMGroup(om.Group):
            def setup(self):
                ivc = self.add_subsystem("indep_vars", om.IndepVarComp(), promotes=["*"])
                ivc.add_output("length", val=0.0, desc="Length")
                ivc.add_output("width",val=0.0, desc="Width")

                self.add_subsystem("comp",OMComponent(),  promotes=["*"])

        class OMComponent(om.ExplicitComponent):
            def setup(self):
                self.add_discrete_input("definition", val=1, desc="Flag")
                self.add_input("length", val=0.0, desc="Length")
                self.add_input("width", val=0.0, desc="Width")

                self.add_output("area", val=0.0, desc="Area")

            def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
                if discrete_inputs['definition'] == 0:
                    outputs["area"] = inputs['length'] * inputs['width']
                else:
                    outputs["area"] = 2.* inputs['length'] * inputs['width']

        myopt = om.Problem(model=OMGroup(), driver=om.ScipyOptimizeDriver())

        myopt.model.approx_totals(method="fd")
        myopt.model.add_objective("area")
        myopt.model.add_design_var("length",lower=0,upper=10)

        myopt.setup()

        myopt["width"] = 3.
        myopt["length"] = 2.
        myopt.run_driver()

        # generate scaling report
        myopt.driver.scaling_report(show_browser=False)

        # verify access to discrete value
        self.assertEqual(myopt["definition"], 1)


if __name__ == '__main__':
    unittest.main()
