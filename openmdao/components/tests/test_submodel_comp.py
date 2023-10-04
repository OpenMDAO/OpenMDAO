import unittest
from numpy import pi

import openmdao.api as om
from openmdao.utils.mpi import MPI
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials, \
     assert_check_totals
from openmdao.test_suite.groups.parallel_groups import FanIn, FanOut

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None


class TestSubmodelComp(unittest.TestCase):
    def test_submodel_comp(self):
        p = om.Problem()

        model = om.Group()
        model.add_subsystem('supComp', om.ExecComp('z = x**2 + y'),
                            promotes_inputs=['x', 'y'],
                            promotes_outputs=['z'])

        submodel1 = om.Group()
        submodel1.add_subsystem('sub1_ivc_r', om.IndepVarComp('r', 1.),
                                promotes_outputs=['r'])
        submodel1.add_subsystem('sub1_ivc_theta', om.IndepVarComp('theta', pi),
                                promotes_outputs=['theta'])
        submodel1.add_subsystem('subComp1', om.ExecComp('x = r*cos(theta)'),
                                promotes_inputs=['r', 'theta'],
                                promotes_outputs=['x'])

        submodel2 = om.Group()
        submodel2.add_subsystem('sub2_ivc_r', om.IndepVarComp('r', 2),
                                promotes_outputs=['r'])
        submodel2.add_subsystem('sub2_ivc_theta', om.IndepVarComp('theta', pi/2),
                                promotes_outputs=['theta'])
        submodel2.add_subsystem('subComp2', om.ExecComp('y = r*sin(theta)'),
                                promotes_inputs=['r', 'theta'],
                                promotes_outputs=['y'])

        subprob1 = om.Problem()
        subprob1.model.add_subsystem('submodel1', submodel1)
        subprob1.model.promotes('submodel1', any=['*'])

        subprob2 = om.Problem()
        subprob2.model.add_subsystem('submodel2', submodel2)
        subprob2.model.promotes('submodel2', any=['*'])

        subcomp1 = om.SubmodelComp(problem=subprob1,
                                   inputs=['r', 'theta'], outputs=['x'])
        subcomp2 = om.SubmodelComp(problem=subprob2,
                                   inputs=['r', 'theta'], outputs=['y'])

        p.model.add_subsystem('sub1', subcomp1, promotes_inputs=['r','theta'],
                                    promotes_outputs=['x'])
        p.model.add_subsystem('sub2', subcomp2, promotes_inputs=['r','theta'],
                                    promotes_outputs=['y'])
        p.model.add_subsystem('supModel', model, promotes_inputs=['x','y'],
                                    promotes_outputs=['z'])

        p.model.set_input_defaults('r', 1)
        p.model.set_input_defaults('theta', pi)

        p.setup(force_alloc_complex=True)

        p.run_model()
        cpd = p.check_partials(method='cs', out_stream=None)

        assert_near_equal(p.get_val('z'), 1.0)

    def test_no_io(self):
        p = om.Problem()

        model = om.Group()
        model.add_subsystem('supComp', om.ExecComp('z = x**2 + y'),
                            promotes_inputs=['x', 'y'],
                            promotes_outputs=['z'])

        submodel1 = om.Group()
        submodel1.add_subsystem('sub1_ivc_r', om.IndepVarComp('r', 1.),
                                promotes_outputs=['r'])
        submodel1.add_subsystem('sub1_ivc_theta', om.IndepVarComp('theta', pi),
                                promotes_outputs=['theta'])
        submodel1.add_subsystem('subComp1', om.ExecComp('x = r*cos(theta)'),
                                promotes_inputs=['r', 'theta'],
                                promotes_outputs=['x'])

        submodel2 = om.Group()
        submodel2.add_subsystem('sub2_ivc_r', om.IndepVarComp('r', 2),
                                promotes_outputs=['r'])
        submodel2.add_subsystem('sub2_ivc_theta', om.IndepVarComp('theta', pi/2),
                                promotes_outputs=['theta'])
        submodel2.add_subsystem('subComp2', om.ExecComp('y = r*sin(theta)'),
                                promotes_inputs=['r', 'theta'],
                                promotes_outputs=['y'])

        subprob1 = om.Problem()
        subprob1.model.add_subsystem('submodel1', submodel1)
        subprob2 = om.Problem()
        subprob2.model.add_subsystem('submodel2', submodel2)

        subcomp1 = om.SubmodelComp(problem=subprob1)
        subcomp2 = om.SubmodelComp(problem=subprob2)

        p.model.add_subsystem('sub1', subcomp1)
        p.model.add_subsystem('sub2', subcomp2)
        p.model.add_subsystem('supModel', model, promotes_inputs=['x','y'],
                                    promotes_outputs=['z'])

        p.setup(force_alloc_complex=True)

        with self.assertRaises(Exception) as ctx:
            p.set_val('r', 1)
            p.set_val('theta', pi)

            p.run_model()

        msg = '\'<model> <class Group>: Variable "r" not found.\''
        self.assertTrue(str(ctx.exception).startswith(msg))

    def test_variable_alias(self):
        p = om.Problem()
        model = om.Group()

        model.add_subsystem('subsys', om.ExecComp('z = x**2 + y**2'))
        subprob = om.Problem()
        subprob.model.add_subsystem('submodel', model)
        subcomp = om.SubmodelComp(problem=subprob,
                                  inputs=[('submodel.subsys.x', 'a'), ('submodel.subsys.y', 'b')],
                                  outputs=[('submodel.subsys.z', 'c')])

        p.model.add_subsystem('subcomp', subcomp, promotes_inputs=['a', 'b'], promotes_outputs=['c'])
        p.setup()

        p.set_val('a', 1)
        p.set_val('b', 2)

        p.run_model()

        inputs = p.model.subcomp.list_inputs()
        outputs = p.model.subcomp.list_outputs()

        inputs = {inputs[i][0]: inputs[i][1]['val'] for i in range(len(inputs))}
        outputs = {outputs[i][0]: outputs[i][1]['val'] for i in range(len(outputs))}

        assert_near_equal(inputs['a'], 1)
        assert_near_equal(inputs['b'], 2)
        assert_near_equal(outputs['c'], 5)

    def test_unconnected_same_var(self):
        p = om.Problem()

        model = om.Group()

        model.add_subsystem('x1Comp', om.ExecComp('x1 = x*3'))
        model.add_subsystem('x2Comp', om.ExecComp('x2 = x**3'))
        model.connect('x1Comp.x1', 'model.x1')
        model.connect('x2Comp.x2', 'model.x2')
        model.add_subsystem('model', om.ExecComp('z = x1**2 + x2**2'))

        subprob = om.Problem()
        subprob.model.add_subsystem('submodel', model)

        comp = om.SubmodelComp(problem=subprob,
                               inputs=[('submodel.x1Comp.x', 'x'), ('submodel.x2Comp.x', 'y')],
                               outputs=[('submodel.model.z', 'z')])

        p.model.add_subsystem('comp', comp)

        p.model.set_input_defaults('comp.x', 1)
        p.model.set_input_defaults('comp.y', 2)

        p.setup()

        p.run_model()

        inputs = p.model.comp.list_inputs()
        outputs = p.model.comp.list_outputs()

        inputs = {inputs[i][0]: inputs[i][1]['val'] for i in range(len(inputs))}
        outputs = {outputs[i][0]: outputs[i][1]['val'] for i in range(len(outputs))}

        assert_near_equal(inputs['x'], 1)
        assert_near_equal(inputs['y'], 2)
        assert_near_equal(outputs['z'], 73)

    def test_wildcard(self):
        p = om.Problem()
        model = om.Group()

        model.add_subsystem('sub', om.ExecComp('z = x1**2 + x2**2 + x3**2'), promotes=['*'])
        subprob = om.Problem()
        subprob.model.add_subsystem('submodel', model, promotes=['*'])
        comp = om.SubmodelComp(problem=subprob, inputs=['x*'], outputs=['*'])

        p.model.add_subsystem('comp', comp, promotes_inputs=['*'], promotes_outputs=['*'])
        p.setup()

        p.set_val('x1', 1)
        p.set_val('x2', 2)
        p.set_val('x3', 3)

        p.run_model()

        inputs = p.model.comp.list_inputs()
        outputs = p.model.comp.list_outputs()

        inputs = {inputs[i][0]: inputs[i][1]['val'] for i in range(len(inputs))}
        outputs = {outputs[i][0]: outputs[i][1]['val'] for i in range(len(outputs))}

        assert_near_equal(inputs['x1'], 1)
        assert_near_equal(inputs['x2'], 2)
        assert_near_equal(inputs['x3'], 3)
        assert_near_equal(outputs['z'], 14)

    def test_add_io_before_setup(self):
        p = om.Problem()

        model = om.Group()
        model.add_subsystem('supComp', om.ExecComp('z = x**2 + y'),
                            promotes_inputs=['x', 'y'],
                            promotes_outputs=['z'])

        submodel1 = om.Group()
        submodel1.add_subsystem('sub1_ivc_r', om.IndepVarComp('r', 1.),
                                promotes_outputs=['r'])
        submodel1.add_subsystem('sub1_ivc_theta', om.IndepVarComp('theta', pi),
                                promotes_outputs=['theta'])
        submodel1.add_subsystem('subComp1', om.ExecComp('x = r*cos(theta)'))

        submodel2 = om.Group()
        submodel2.add_subsystem('sub2_ivc_r', om.IndepVarComp('r', 2),
                                promotes_outputs=['r'])
        submodel2.add_subsystem('sub2_ivc_theta', om.IndepVarComp('theta', pi/2),
                                promotes_outputs=['theta'])
        submodel2.add_subsystem('subComp2', om.ExecComp('y = r*sin(theta)'))

        subprob1 = om.Problem()
        subprob1.model.add_subsystem('submodel1', submodel1, promotes=['*'])
        subprob2 = om.Problem()
        subprob2.model.add_subsystem('submodel2', submodel2, promotes=['*'])

        comp1 = om.SubmodelComp(problem=subprob1)
        comp2 = om.SubmodelComp(problem=subprob2)

        comp1.add_input('subComp1.r', name='r')
        comp1.add_input('subComp1.theta', name='theta')
        comp2.add_input('subComp2.r', name='r')
        comp2.add_input('subComp2.theta', name='theta')

        comp1.add_output('subComp1.x', name='x')
        comp2.add_output('subComp2.y', name='y')

        p.model.add_subsystem('sub1', comp1, promotes_inputs=['r','theta'],
                                    promotes_outputs=['x'])
        p.model.add_subsystem('sub2', comp2, promotes_inputs=['r','theta'],
                                    promotes_outputs=['y'])
        p.model.add_subsystem('supModel', model, promotes_inputs=['x','y'],
                                    promotes_outputs=['z'])

        p.model.set_input_defaults('r', 1)
        p.model.set_input_defaults('theta', pi)

        p.setup(force_alloc_complex=True)

        p.run_model()
        cpd = p.check_partials(method='cs', out_stream=None)

        assert_near_equal(p.get_val('z'), 1.0)

    def test_add_io_after_setup(self):
        class Subsys1(om.Group):
            def setup(self):
                model = om.Group()
                comp = om.ExecComp('x = r*cos(theta)')
                model.add_subsystem('comp', comp, promotes_inputs=['r', 'theta'],
                                    promotes_outputs=['x'])
                subprob = om.Problem(); subprob.model.add_subsystem('model', model)
                subprob.model.promotes('model', any=['*'])
                self.add_subsystem('submodel1', om.SubmodelComp(problem=subprob))

            def configure(self):
                self._get_subsystem('submodel1').add_input('r')
                self._get_subsystem('submodel1').add_input('theta')
                self._get_subsystem('submodel1').add_output('x')

                self.promotes('submodel1', ['r', 'theta', 'x'])

        class Subsys2(om.Group):
            def setup(self):
                model = om.Group()
                comp = om.ExecComp('y = r*sin(theta)')
                model.add_subsystem('comp', comp, promotes_inputs=['r', 'theta'],
                                    promotes_outputs=['y'])
                subprob = om.Problem(); subprob.model.add_subsystem('model', model)
                subprob.model.promotes('model', any=['*'])
                self.add_subsystem('submodel2', om.SubmodelComp(problem=subprob))

            def configure(self):
                self._get_subsystem('submodel2').add_input('r')
                self._get_subsystem('submodel2').add_input('theta')
                self._get_subsystem('submodel2').add_output('y')

                self.promotes('submodel2', ['r', 'theta', 'y'])

        p = om.Problem()

        model = om.Group()
        model.add_subsystem('supComp', om.ExecComp('z = x**2 + y'),
                            promotes_inputs=['x', 'y'],
                            promotes_outputs=['z'])

        p.model.add_subsystem('sub1', Subsys1(), promotes_inputs=['r', 'theta'],
                              promotes_outputs=['x'])
        p.model.add_subsystem('sub2', Subsys2(), promotes_inputs=['r', 'theta'],
                              promotes_outputs=['y'])
        p.model.add_subsystem('supModel', model, promotes_inputs=['x','y'],
                              promotes_outputs=['z'])

        p.setup(force_alloc_complex=True)

        p.set_val('r', 1)
        p.set_val('theta', pi)

        p.run_model()
        cpd = p.check_partials(method='cs', out_stream=None)

        assert_near_equal(p.get_val('z'), 1)

    def test_invalid_io(self):
        p = om.Problem()

        model = om.Group()
        model.add_subsystem('supComp', om.ExecComp('z = x**2 + y'),
                            promotes_inputs=['x', 'y'],
                            promotes_outputs=['z'])

        submodel1 = om.Group()
        submodel1.add_subsystem('sub1_ivc_r', om.IndepVarComp('r', 1.),
                                promotes_outputs=['r'])
        submodel1.add_subsystem('sub1_ivc_theta', om.IndepVarComp('theta', pi),
                                promotes_outputs=['theta'])
        submodel1.add_subsystem('subComp1', om.ExecComp('x = r*cos(theta)'),
                                promotes_inputs=['r', 'theta'],
                                promotes_outputs=['x'])

        submodel2 = om.Group()
        submodel2.add_subsystem('sub2_ivc_r', om.IndepVarComp('r', 2),
                                promotes_outputs=['r'])
        submodel2.add_subsystem('sub2_ivc_theta', om.IndepVarComp('theta', pi/2),
                                promotes_outputs=['theta'])
        submodel2.add_subsystem('subComp2', om.ExecComp('y = r*sin(theta)'),
                                promotes_inputs=['r', 'theta'],
                                promotes_outputs=['y'])

        subprob1 = om.Problem()
        subprob1.model.add_subsystem('submodel1', submodel1, promotes=['*'])
        subprob2 = om.Problem()
        subprob2.model.add_subsystem('submodel2', submodel2, promotes=['*'])

        comp1 = om.SubmodelComp(problem=subprob1)
        comp2 = om.SubmodelComp(problem=subprob2)

        comp1.add_input('psi')

        p.model.add_subsystem('sub1', comp1)
        p.model.add_subsystem('sub2', comp2)
        p.model.add_subsystem('supModel', model, promotes_inputs=['x','y'],
                              promotes_outputs=['z'])

        with self.assertRaises(Exception) as ctx:
            p.setup(force_alloc_complex=True)

        msg = 'Variable psi not found in model'
        self.assertEqual(str(ctx.exception), msg)

    def test_multiple_setups(self):
        p = om.Problem()

        model = om.Group()
        model.add_subsystem('supComp', om.ExecComp('y = 3*x + 4'),
                            promotes_inputs=['x'], promotes_outputs=['y'])

        submodel = om.Group()
        submodel.add_subsystem('subComp', om.ExecComp('x = 6*z + 3'),
                               promotes_inputs=['z'], promotes_outputs=['x'])

        subprob = om.Problem()
        subprob.model.add_subsystem('submodel', submodel, promotes=['*'])

        comp = om.SubmodelComp(problem=subprob)
        comp.add_input('z')
        comp.add_output('x')

        p.model.add_subsystem('sub', comp, promotes_inputs=['z'],
                              promotes_outputs=['x'])
        p.model.add_subsystem('sup', model, promotes_inputs=['x'],
                              promotes_outputs=['y'])

        p.setup(force_alloc_complex=False)
        p.setup(force_alloc_complex=False)
        p.setup(force_alloc_complex=True)

        p.set_val('z', 1)

        p.run_model()

        assert_near_equal(p.get_val('z'), 1)
        assert_near_equal(p.get_val('x'), 9)
        assert_near_equal(p.get_val('y'), 31)

    def test_multiple_submodel_setups(self):
        model = om.Group()
        model.add_subsystem('supComp', om.ExecComp('y = 3*x + 4'),
                            promotes_inputs=['x'], promotes_outputs=['y'])

        submodel = om.Group()
        submodel.add_subsystem('subComp', om.ExecComp('x = 6*z + 3'),
                               promotes_inputs=['z'], promotes_outputs=['x'])

        subprob = om.Problem()
        subprob.model.add_subsystem('submodel', submodel, promotes=['*'])

        comp = om.SubmodelComp(problem=subprob)
        comp.add_input('z')
        comp.add_output('x')

        comp.setup()
        comp.setup()

    def test_add_io_meta_override(self):
        p = om.Problem()
        subprob = om.Problem()
        subprob.model.add_subsystem('comp', om.ExecComp('x = r*cos(theta)'), promotes=['*'])
        submodel = om.SubmodelComp(problem=subprob)

        submodel.add_input('r', name='new_r', val=20)
        submodel.add_input('theta', name='new_theta', val=0.5)
        submodel.add_output('x', name='new_x', val=100)

        p.model.add_subsystem('submodel', submodel, promotes=['*'])

        p.setup()
        p.final_setup()

        self.assertEqual(p.get_val('new_r'), 20)
        self.assertEqual(p.get_val('new_theta'), 0.5)
        self.assertEqual(p.get_val('new_x'), 100)

    def test_subprob_solver_print(self):
        p = om.Problem()
        subprob = om.Problem()
        subprob.model.add_subsystem('comp', om.ExecComp('x = r*cos(theta)'), promotes=['*'])
        submodel = om.SubmodelComp(problem=subprob)

        submodel.add_input('r', name='new_r', val=20)
        submodel.add_input('theta', name='new_theta', val=0.5)
        submodel.add_output('x', name='new_x', val=100)

        submodel._subprob.set_solver_print(level=3, depth=20, type_='NL')

        p.model.add_subsystem('submodel', submodel, promotes=['*'])

        self.assertTrue((3, 20, 'NL') in p.model.submodel._subprob.model._solver_print_cache)

    def test_complex_step_across_submodel(self):
        p = om.Problem()
        subprob = om.Problem()
        subprob.model.add_subsystem('comp', om.ExecComp('x = r*cos(theta)'), promotes=['*'])
        submodel = om.SubmodelComp(problem=subprob)

        submodel.add_input('r', name='new_r', val=20)
        submodel.add_input('theta', name='new_theta', val=0.5)
        submodel.add_output('x', name='new_x', val=100)

        model = p.model
        model.add_subsystem('submodel', submodel, promotes=['*'])
        model.add_design_var('new_r')
        model.add_design_var('new_theta')
        model.add_objective('new_x')

        p.setup(force_alloc_complex=True)
        p.run_model()

        totals = p.check_totals(method='cs')
        assert_check_totals(totals, atol=1e-11, rtol=1e-11)


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class TestSubmodelCompMPI(unittest.TestCase):
    N_PROCS = 2

    def test_submodel_comp(self):
        p = om.Problem()

        model = p.model

        par = model.add_subsystem('par', om.ParallelGroup())

        par.add_subsystem('subprob1', om.SubmodelComp(problem=om.Problem(model=FanOut()),
                                                      inputs=['p.x'], outputs=['comp2.y', 'comp3.y']))
        par.add_subsystem('subprob2', om.SubmodelComp(problem=om.Problem(model=FanIn()),
                                                      inputs=['p1.x1', 'p2.x2'], outputs=['comp3.y']))

        p.setup(force_alloc_complex=True)
        p.run_model()
        cpd = p.check_partials(method='cs', out_stream=None)
        assert_check_partials(cpd)


if __name__ == '__main__':
    unittest.main()
