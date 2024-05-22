import unittest
import numpy as np

import openmdao.api as om
from openmdao.utils.mpi import MPI
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials, \
     assert_check_totals
from openmdao.test_suite.groups.parallel_groups import FanInGrouped, FanOutGrouped

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None


def build_submodelcomp1(promote=True, **kwargs):
    subprob1 = om.Problem()
    submodel1 = subprob1.model.add_subsystem('submodel1', om.Group())
    submodel1.add_subsystem('sub1_ivc_r', om.IndepVarComp('r', 1.),
                            promotes_outputs=['r'])
    submodel1.add_subsystem('sub1_ivc_theta', om.IndepVarComp('theta', np.pi),
                            promotes_outputs=['theta'])
    if promote:
        promotes = ['*']
    else:
        promotes = ()
    submodel1.add_subsystem('subComp1', om.ExecComp('x = r*cos(theta)'), promotes=promotes)
    subprob1.model.promotes('submodel1', any=['*'])
    return om.SubmodelComp(problem=subprob1, **kwargs)


def build_submodelcomp2(promote=True, **kwargs):
    subprob2 = om.Problem()
    submodel2 = subprob2.model.add_subsystem('submodel2', om.Group())
    submodel2.add_subsystem('sub2_ivc_r', om.IndepVarComp('r', 2),
                            promotes_outputs=['r'])
    submodel2.add_subsystem('sub2_ivc_theta', om.IndepVarComp('theta', np.pi/2),
                            promotes_outputs=['theta'])
    if promote:
        promotes = ['*']
    else:
        promotes = ()
    submodel2.add_subsystem('subComp2', om.ExecComp('y = r*sin(theta)'), promotes=promotes)
    subprob2.model.promotes('submodel2', any=['*'])
    return om.SubmodelComp(problem=subprob2, **kwargs)


class TestSubmodelComp(unittest.TestCase):
    def test_submodel_comp(self):
        p = om.Problem()
        supmodel = om.Group()
        supmodel.add_subsystem('supComp', om.ExecComp('z = x**2 + y'),
                               promotes_inputs=['x', 'y'],
                               promotes_outputs=['z'])

        p.model.add_subsystem('sub1', build_submodelcomp1(inputs=['r', 'theta'], outputs=['x'],
                              do_coloring=True),
                              promotes_inputs=['r','theta'],
                              promotes_outputs=['x'])

        p.model.add_subsystem('sub2', build_submodelcomp2(inputs=['r', 'theta'], outputs=['y'],
                              do_coloring=True),
                              promotes_inputs=['r','theta'],
                              promotes_outputs=['y'])

        p.model.add_subsystem('supModel', supmodel, promotes_inputs=['x','y'],
                              promotes_outputs=['z'])

        p.model.set_input_defaults('r', 1)
        p.model.set_input_defaults('theta', np.pi)

        p.setup(force_alloc_complex=True)

        p.run_model()
        assert_check_partials(p.check_partials(method='cs', out_stream=None))

        assert_near_equal(p.get_val('z'), 1.0)

    def test_no_io(self):
        p = om.Problem()

        model = om.Group()
        model.add_subsystem('supComp', om.ExecComp('z = x**2 + y'),
                            promotes_inputs=['x', 'y'],
                            promotes_outputs=['z'])

        p.model.add_subsystem('sub1', build_submodelcomp1(do_coloring=True))
        p.model.add_subsystem('sub2', build_submodelcomp2(do_coloring=True))
        p.model.add_subsystem('supModel', model, promotes_inputs=['x','y'], promotes_outputs=['z'])

        p.setup(force_alloc_complex=True)

        with self.assertRaises(Exception) as ctx:
            p.set_val('r', 1)

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
                                  outputs=[('submodel.subsys.z', 'c')], do_coloring=True)

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
                               outputs=[('submodel.model.z', 'z')], do_coloring=True)

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

        model.add_subsystem('sub', om.ExecComp(['z = foo**2 + bar**2 + baz**2',
                                                'out = bgd - xyz*foo + baz',
                                                'result = -foo*bgd + bar*xyz']), promotes=['*'])
        subprob = om.Problem()
        subprob.model.add_subsystem('submodel', model, promotes=['*'])
        comp = om.SubmodelComp(problem=subprob, inputs=['x*'], outputs=['*'], do_coloring=True)

        p.model.add_subsystem('comp', comp, promotes_inputs=['*'], promotes_outputs=['*'])
        p.setup()

        p.set_val('foo', 1)
        p.set_val('bar', 2)
        p.set_val('baz', 3)
        p.set_val('bgd', 4)
        p.set_val('xyz', 5)

        p.run_model()

        inputs = p.model.comp.list_inputs()
        outputs = p.model.comp.list_outputs()

        inputs = {inputs[i][0]: inputs[i][1]['val'] for i in range(len(inputs))}
        outputs = {outputs[i][0]: outputs[i][1]['val'] for i in range(len(outputs))}

        assert_near_equal(inputs['foo'], 1)
        assert_near_equal(inputs['bar'], 2)
        assert_near_equal(inputs['baz'], 3)
        assert_near_equal(outputs['z'], 14)

        assert_check_partials(p.check_partials(out_stream=None), atol=2e-6)

    def test_add_io_before_setup(self):
        p = om.Problem()

        model = om.Group()
        model.add_subsystem('supComp', om.ExecComp('z = x**2 + y'),
                            promotes_inputs=['x', 'y'],
                            promotes_outputs=['z'])

        comp1 = build_submodelcomp1(promote=False, do_coloring=True)
        comp2 = build_submodelcomp2(promote=False, do_coloring=True)

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
        p.model.set_input_defaults('theta', np.pi)

        p.setup(force_alloc_complex=True)

        p.run_model()
        assert_check_partials(p.check_partials(out_stream=None))

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
                self.add_subsystem('submodel1', om.SubmodelComp(problem=subprob, do_coloring=True))

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
                self.add_subsystem('submodel2', om.SubmodelComp(problem=subprob, do_coloring=True))

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
        p.set_val('theta', np.pi)

        p.run_model()
        assert_check_partials(p.check_partials(method='cs', out_stream=None))

        assert_near_equal(p.get_val('z'), 1)

    def test_invalid_io(self):
        p = om.Problem()

        model = om.Group()
        model.add_subsystem('supComp', om.ExecComp('z = x**2 + y'),
                            promotes_inputs=['x', 'y'],
                            promotes_outputs=['z'])

        comp1 = build_submodelcomp1(do_coloring=True)
        comp2 = build_submodelcomp2(do_coloring=True)

        comp1.add_input('psi')

        p.model.add_subsystem('sub1', comp1)
        p.model.add_subsystem('sub2', comp2)
        p.model.add_subsystem('supModel', model, promotes_inputs=['x','y'],
                              promotes_outputs=['z'])

        with self.assertRaises(Exception) as ctx:
            p.setup(force_alloc_complex=True)

        msg = "'psi' is not an independent variable in the submodel."
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

        comp = om.SubmodelComp(problem=subprob, do_coloring=True)
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
        top = om.Problem()
        model = top.model
        model.add_subsystem('supComp', om.ExecComp('y = 3*x + 4'),
                            promotes_inputs=['x'], promotes_outputs=['y'])

        submodel = om.Group()
        submodel.add_subsystem('subComp', om.ExecComp('x = 6*z + 3'),
                               promotes_inputs=['z'], promotes_outputs=['x'])

        subprob = om.Problem()
        subprob.model.add_subsystem('submodel', submodel, promotes=['*'])

        comp = om.SubmodelComp(problem=subprob, do_coloring=True)
        comp.add_input('z')
        comp.add_output('x')

        top.setup()
        top.setup()
        top.final_setup()
        top.run_model()

    def test_add_io_meta_override(self):
        p = om.Problem()
        subprob = om.Problem()
        subprob.model.add_subsystem('comp', om.ExecComp('x = r*cos(theta)'), promotes=['*'])
        submodel = om.SubmodelComp(problem=subprob, do_coloring=True)

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
        submodel = om.SubmodelComp(problem=subprob, do_coloring=True)

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
        submodel = om.SubmodelComp(problem=subprob, do_coloring=True)

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

    def test_problem_property(self):
        """Tests the problem property of SubmodelComp"""
        p = om.Problem()
        submodel = om.SubmodelComp(problem=p, do_coloring=True)
        subprob = submodel.problem

        self.assertIsInstance(subprob, om.Problem) # make sure it returns a problem
        self.assertEqual(subprob, p) # make sure it returns correct problem

        with self.assertRaises(AttributeError): # make sure it cannot be assigned
            submodel.problem = om.Problem()


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class TestSubmodelCompMPI(unittest.TestCase):
    N_PROCS = 2

    def test_submodel_comp(self):
        p = om.Problem()

        model = p.model

        par = model.add_subsystem('par', om.ParallelGroup())

        par.add_subsystem('subprob1', om.SubmodelComp(problem=om.Problem(model=FanOutGrouped()),
                                                      inputs=['iv.x'], outputs=['c2.y', 'c3.y'],
                                                      do_coloring=True))
        par.add_subsystem('subprob2', om.SubmodelComp(problem=om.Problem(model=FanInGrouped()),
                                                      inputs=['*'], outputs=['c3.y'],
                                                      do_coloring=True))

        p.setup(force_alloc_complex=True)
        p.run_model()
        assert_check_partials(p.check_partials(method='cs', out_stream=None))

    def test_submodel_with_parallel_group(self):
        p = om.Problem()

        model = p.model

        G = model.add_subsystem('G', om.Group())

        psub = om.Problem()
        par = psub.model.add_subsystem('par', om.ParallelGroup())
        par.add_subsystem('fanout1', FanOutGrouped())
        par.add_subsystem('fanout2', FanOutGrouped())
        G.add_subsystem('subprob1', om.SubmodelComp(problem=psub, inputs=['*'], outputs=['*'],
                                                    do_coloring=False))

        p.setup(force_alloc_complex=True)
        p.run_model()

        assert_near_equal(psub.get_val('par.fanout1.c2.y', get_remote=True), -6.0)
        assert_near_equal(psub.get_val('par.fanout2.c2.y', get_remote=True), -6.0)
        assert_near_equal(psub.get_val('par.fanout1.c3.y', get_remote=True), 15.0)
        assert_near_equal(psub.get_val('par.fanout2.c3.y', get_remote=True), 15.0)

        assert_check_partials(psub.check_partials(method='cs', show_only_incorrect=True)) #, out_stream=None))
        assert_check_partials(p.check_partials(method='cs', show_only_incorrect=True)) #, out_stream=None))

    @unittest.skip("Unskip this after distributed vars work with SubmodelComp")
    def test_submodel_distrib(self):
        p = om.Problem()

        model = p.model

        G = model.add_subsystem('G', om.Group())

        size = 3
        xstart = (np.arange(size) + 1.0) * 5.0
        ystart = (np.arange(size) + 1.0) * 3.0

        psub = om.Problem()
        ivc = psub.model.add_subsystem('ivc', om.IndepVarComp('x', xstart))
        ivc.add_output('y', ystart)

        psub.model.add_subsystem('distcomp1', DistribCompDerivs(size=size))
        psub.model.add_subsystem('distcomp2', DistribCompDerivs(size=size))
        psub.model.connect('ivc.x', 'distcomp1.invec')
        psub.model.connect('ivc.y', 'distcomp2.invec')
        G.add_subsystem('subprob1', om.SubmodelComp(problem=psub, inputs=['*'], outputs=['*'],
                                                    do_coloring=False))

        p.setup(force_alloc_complex=True)
        p.run_model()

        if p.comm.rank == 0:
            assert_near_equal(psub.get_val('distcomp1.outvec', get_remote=False), [10.0, 20.0])
            assert_near_equal(psub.get_val('distcomp2.outvec', get_remote=False), [6.0, 12.0])
        else:
            assert_near_equal(psub.get_val('distcomp1.outvec', get_remote=False), [-45.0])
            assert_near_equal(psub.get_val('distcomp2.outvec', get_remote=False), [-27.0])

        assert_near_equal(psub.get_val('distcomp1.outvec', get_remote=True), [10.0, 20.0, -45.0])
        assert_near_equal(psub.get_val('distcomp2.outvec', get_remote=True), [6.0, 12.0, -27.0])

        assert_check_partials(psub.check_partials(method='cs', show_only_incorrect=False)) #, out_stream=None))

        assert_check_partials(p.check_partials(method='cs', show_only_incorrect=False)) #, out_stream=None))


class IncompleteRelevanceGroup(om.Group):
    def __init__(self, size, **kwargs):
        super().__init__(**kwargs)
        self.size = size

    def setup(self):
        size = self.size
        self.add_subsystem('C1', om.ExecComp('y = 1.5*x', shape=size))
        self.add_subsystem('C2', om.ExecComp('y = 3.0*x', shape=size*2+1))
        self.add_subsystem('C3', om.ExecComp('y = 2.75*x', shape=size))
        self.add_subsystem('C4', om.ExecComp('y = 4.25*x1 - 0.75*x2', shape=size))
        self.add_subsystem('C5', om.ExecComp('y = sum(x)/sum(x**2)', x=np.ones(size*2+1), y=0.0))
        self.connect('C1.y', ['C3.x', 'C4.x1'])
        self.connect('C2.y', 'C4.x2', src_indices=list(range(size)))
        self.connect('C2.y', 'C5.x')


class TestSubmodelColoring(unittest.TestCase):
    def test_submodel_inner_coloring(self):
        # this one has coloring within the submodelcomp

        p = om.Problem()
        model = p.model


        model.add_subsystem('sub', om.SubmodelComp(problem=om.Problem(model=IncompleteRelevanceGroup(3)),
                                                    inputs=['C1.x', 'C2.x'], outputs=['C3.y', 'C4.y', 'C5.y'],
                                                    do_coloring=True))

        p.setup(force_alloc_complex=True)
        p.run_model()

        check = p.check_partials(method='cs', show_only_incorrect=True)
        assert_check_partials(check)

        check = p.check_totals(of=['sub.C3:y', 'sub.C4:y', 'sub.C5:y'],
                               wrt=['sub.C1:x', 'sub.C2:x'], show_only_incorrect=True)
        assert_check_totals(check)

    def test_submodel_inner_outer_coloring(self):
        # this one has coloring within the submodelcomp and the outer problem driver

        p = om.Problem()
        model = p.model


        model.add_subsystem('sub', om.SubmodelComp(problem=om.Problem(model=IncompleteRelevanceGroup(3)),
                                                    inputs=['C1.x', 'C2.x'], outputs=['C3.y', 'C4.y', 'C5.y'],
                                                    do_coloring=True))

        p.driver = om.ScipyOptimizeDriver(optimizer='SLSQP')
        p.driver.declare_coloring(show_summary=True)

        p.model.add_design_var('sub.C1:x')
        p.model.add_design_var('sub.C2:x')

        p.model.add_objective('sub.C5:y', index=0)
        p.model.add_constraint('sub.C4:y', lower=-1)
        p.model.add_constraint('sub.C3:y', upper=99)

        p.setup(force_alloc_complex=True)
        p.run_driver()

        check = p.check_partials(method='cs', show_only_incorrect=True)
        assert_check_partials(check)

        check = p.check_totals(of=['sub.C3:y', 'sub.C4:y', 'sub.C5:y'],
                               wrt=['sub.C1:x', 'sub.C2:x'], show_only_incorrect=True)
        assert_check_totals(check)

class TestSubmodelColoringMultiSubmodelComps(unittest.TestCase):

    def setUp(self):
        from openmdao.utils.general_utils import set_pyoptsparse_opt

        OPT, _ = set_pyoptsparse_opt('SLSQP')
        if OPT is None:
            raise unittest.SkipTest("pyoptsparse is not installed")

    def test_multiple_submodels_inner_outer_coloring(self):
        # this one has coloring within the submodelcomp and the outer problem driver

        p = om.Problem()

        model = p.model

        par = model.add_subsystem('par', om.ParallelGroup(), promotes=['*'])

        par.add_subsystem('sub1', om.SubmodelComp(problem=om.Problem(model=IncompleteRelevanceGroup(3)),
                                                  inputs=['C1.x', 'C2.x'], outputs=['C3.y', 'C4.y', 'C5.y'],
                                                  do_coloring=True))
        par.add_subsystem('sub2', om.SubmodelComp(problem=om.Problem(model=IncompleteRelevanceGroup(3)),
                                                  inputs=['C1.x', 'C2.x'], outputs=['C3.y', 'C4.y', 'C5.y'],
                                                  do_coloring=True))

        p.driver = om.ScipyOptimizeDriver(optimizer='SLSQP')
        p.driver.declare_coloring(show_summary=True)

        p.model.add_design_var('sub1.C1:x')
        p.model.add_design_var('sub1.C2:x')
        p.model.add_design_var('sub2.C1:x')
        p.model.add_design_var('sub2.C2:x')

        p.model.add_objective('sub1.C5:y', index=0)
        p.model.add_constraint('sub2.C5:y', lower=-1)
        p.model.add_constraint('sub1.C4:y', lower=-1)
        p.model.add_constraint('sub1.C3:y', upper=99)
        p.model.add_constraint('sub2.C4:y', lower=-1)
        p.model.add_constraint('sub2.C3:y', upper=99)

        p.setup(force_alloc_complex=True)
        p.run_driver()

        check = p.check_partials(method='cs', show_only_incorrect=True)
        assert_check_partials(check)

        check = p.check_totals(of=['sub1.C3:y', 'sub1.C4:y', 'sub1.C5:y',
                                      'sub2.C3:y', 'sub2.C4:y', 'sub2.C5:y'],
                                 wrt=['sub1.C1:x', 'sub1.C2:x', 'sub2.C1:x', 'sub2.C2:x'], show_only_incorrect=True)
        assert_check_totals(check)


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class TestSubmodelColoringMultiSubmodelCompsMPI(TestSubmodelColoringMultiSubmodelComps):
    N_PROCS = 2


class TestSubmodelOpt(unittest.TestCase):
    def test_submodel_with_opt(self):
        sub_prob = om.Problem()

        # modify paraboloid eqn with parameter "a" that will be constant during internal opt
        sub_prob.model.add_subsystem("paraboloid", om.ExecComp("f = (x-3)**2 + x*y + (y+4)**2 + a"))

        sub_prob.driver = om.ScipyOptimizeDriver(optimizer="SLSQP")

        sub_prob.model.add_design_var("paraboloid.x", lower=-50, upper=50)
        sub_prob.model.add_design_var("paraboloid.y", lower=-50, upper=50)
        sub_prob.model.add_objective("paraboloid.f")

        top_prob = om.Problem()
        submodel = om.SubmodelComp(problem=sub_prob, inputs=[("paraboloid.a", "a")], outputs=[("paraboloid.f", "g")],
                                   do_coloring=False)
        top_prob.model.add_subsystem("submodel", submodel, promotes=["*"])
        top_prob.setup()
        top_prob.set_val("a", val=-3)

        top_prob.run_model()

        assert_near_equal(sub_prob.get_val("paraboloid.f"), -27.33333307, tolerance=1e-8)
        assert_near_equal(sub_prob.get_val("paraboloid.x"), 6.66712855, tolerance=1e-8)
        assert_near_equal(sub_prob.get_val("paraboloid.y"), -7.33324946, tolerance=1e-8)

        # now test that we're not allowed to compute derivatives of a SubmodelComp if it has an optimizer
        with self.assertRaises(Exception) as cm:
            top_prob.compute_totals(of=['g'], wrt=['a'])

        self.assertEqual(cm.exception.args[0],
                         "'submodel' <class SubmodelComp>: Error calling compute_partials(), Can't compute partial "
                         "derivatives of a SubmodelComp with an internal optimizer.")


def build_submodel(subsystem_name):
    p = om.Problem()
    supmodel = om.Group()
    supmodel.add_subsystem('supComp', om.ExecComp('diameter = r * theta'),
                            promotes_inputs=['*'],
                            promotes_outputs=['*', ('diameter', 'aircraft:fuselage:diameter')])

    subprob1 = om.Problem()
    submodel1 = subprob1.model.add_subsystem('submodel1', om.Group(), promotes=['*'])

    submodel1.add_subsystem(subsystem_name, om.ExecComp('x = diameter * 2 * r * theta'),
                            promotes=['*', ('diameter', 'aircraft:fuselage:diameter')])
    submodel1.add_subsystem('b', om.ExecComp('y = mass * donkey_kong'),
                            promotes=['*', ('mass', 'dynamic:mission:mass'),
                                      ('donkey_kong', 'aircraft:engine:donkey_kong')])

    p.model.add_subsystem('supModel', supmodel, promotes_inputs=['*'],
                            promotes_outputs=['*'])


    submodel = om.SubmodelComp(problem=subprob1, inputs=['*'], outputs=['*'], do_coloring=True)
    p.model.add_subsystem('sub1', submodel,
                            promotes_inputs=['*'],
                            promotes_outputs=['*'])

    p.model.set_input_defaults('r', 1.25)
    p.model.set_input_defaults('theta', np.pi)

    p.setup(force_alloc_complex=True)

    p.set_val('r', 1.25)
    p.set_val('theta', 0.5)
    p.set_val('dynamic:mission:mass', 2.0)
    p.set_val('aircraft:engine:donkey_kong', 3.0)
    p.set_val('aircraft:fuselage:diameter', 3.5)

    return p


class TestSubModelBug(unittest.TestCase):

    def test_submodel_bug1(self):
        p = build_submodel(subsystem_name='a')

        p.run_model()

        assert_near_equal(p.get_val('y'), 2.0 * 3.0)
        assert_check_partials(p.check_partials(method='cs', out_stream=None))


    def test_submodel_bug2(self):
        p = build_submodel(subsystem_name='c')

        p.run_model()

        assert_near_equal(p.get_val('y'), 2.0 * 3.0)
        assert_check_partials(p.check_partials(method='cs', out_stream=None))


if __name__ == '__main__':
    unittest.main()
