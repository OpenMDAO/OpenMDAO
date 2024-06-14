import unittest
import numpy as np

import openmdao.api as om
import openmdao.core.problem
import openmdao.utils.hooks as hooks
from openmdao.utils.assert_utils import assert_near_equal


def make_hook(name):
    def hook_func(obj):
        obj.calls.append(name)
    return hook_func


def make_hook_pass_args(name):
    def hook_func(obj, args, kwargs, **kw):
        obj.calls.append((name, args, kwargs))
    return hook_func


def make_hook_pass_return(name):
    def hook_func(obj, ret):
        obj.calls.append((name, ret))
    return hook_func


def make_hook_pass_args_pass_return(name):
    def hook_func(obj, args, kwargs, ret, **kw):
        obj.calls.append((name, args, kwargs, ret))
    return hook_func


def hooks_active(f):
    # turn hooks on and off around a hooks test
    def _wrapper(*args, **kwargs):
        hooks._reset_all_hooks()
        hooks.use_hooks = True
        try:
            f(*args, **kwargs)
        finally:
            hooks.use_hooks = False
    return _wrapper


class HooksTestCase(unittest.TestCase):
    def setUp(self):
        openmdao.core.problem._clear_problem_names()  # need to reset these to simulate separate runs

    def build_model(self, name='problem1', prob=None):
        if prob is None:
            prob = om.Problem(name=name)
        prob.calls = []
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 3.0))
        model.add_subsystem('p2', om.IndepVarComp('y', -4.0))
        model.add_subsystem('comp', om.ExecComp("f_xy=2.0*x+3.0*y"))

        model.connect('p1.x', 'comp.x')
        model.connect('p2.y', 'comp.y')
        prob.setup()
        return prob

    @hooks_active
    def test_ncalls(self):
        hooks._register_hook('final_setup', 'Problem', pre=make_hook('pre_final'), post=make_hook('post_final'), ncalls=2)
        hooks._register_hook('final_setup', 'Problem', pre=make_hook('pre_final2'), post=make_hook('post_final2'))

        prob = self.build_model()
        prob.run_model()
        prob.run_model()
        prob.run_model()

        self.assertEqual(prob.calls, ['pre_final', 'pre_final2', 'post_final', 'post_final2',
                                      'pre_final', 'pre_final2', 'post_final', 'post_final2',
                                      'pre_final2', 'post_final2',
                                     ])

    @hooks_active
    def test_exit(self):
        hooks._register_hook('final_setup', 'Problem', pre=make_hook('pre_final'), post=make_hook('post_final'))
        hooks._register_hook('final_setup', 'Problem', pre=make_hook('pre_final2'), post=make_hook('post_final2'), exit=True)

        prob = self.build_model()
        try:
            prob.run_model()
            prob.run_model()
            prob.run_model()
        except SystemExit:
            self.assertEqual(prob.calls, ['pre_final', 'pre_final2', 'post_final', 'post_final2'])
        else:
            self.fail("sys.exit() was not called")

    @hooks_active
    def test_multiwrap(self):
        post_final = make_hook('post_final')
        pre_final = make_hook('pre_final')
        hooks._register_hook('final_setup', 'Problem', pre=pre_final, post=post_final)
        hooks._register_hook('final_setup', 'Problem', pre=make_hook('pre_final2'), post=make_hook('post_final2'))

        prob = self.build_model()
        prob.run_model()
        prob.run_model()
        prob.run_model()

        self.assertEqual(prob.calls, ['pre_final', 'pre_final2', 'post_final', 'post_final2',
                                      'pre_final', 'pre_final2', 'post_final', 'post_final2',
                                      'pre_final', 'pre_final2', 'post_final', 'post_final2',
                                     ])

        hooks._unregister_hook('final_setup', 'Problem', pre=pre_final, post=False)
        prob.calls = []

        prob.run_model()
        prob.run_model()
        prob.run_model()

        self.assertEqual(prob.calls, ['pre_final2', 'post_final', 'post_final2',
                                      'pre_final2', 'post_final', 'post_final2',
                                      'pre_final2', 'post_final', 'post_final2',
                                     ])

        hooks._unregister_hook('final_setup', 'Problem', pre=True, post=False)
        prob.calls = []

        prob.run_model()
        prob.run_model()
        prob.run_model()

        self.assertEqual(prob.calls, ['post_final', 'post_final2',
                                      'post_final', 'post_final2',
                                      'post_final', 'post_final2',
                                     ])

    @hooks_active
    def test_multiwrap_mixed_None_inst(self):
        pre_final = make_hook('pre_final')
        post_final = make_hook('post_final')
        hooks._register_hook('final_setup', 'Problem', pre=pre_final, post=post_final)
        hooks._register_hook('final_setup', 'Problem', inst_id='problem1', ncalls=2,
                             pre=make_hook('pre_final1'), post=make_hook('post_final1'))
        hooks._register_hook('final_setup', 'Problem', inst_id='problem2',
                             pre=make_hook('pre_final2'), post=make_hook('post_final2'))

        probs = [self.build_model(f"problem{i+1}") for i in range(2)]
        for prob in probs:
            prob.calls = []
            for i in range(3):
                prob.run_model()

        self.assertEqual(probs[0].calls, ['pre_final', 'pre_final1', 'post_final', 'post_final1', 'pre_final', 'pre_final1', 'post_final', 'post_final1', 'pre_final', 'post_final'])
        self.assertEqual(probs[1].calls, ['pre_final', 'pre_final2', 'post_final', 'post_final2', 'pre_final', 'pre_final2', 'post_final', 'post_final2', 'pre_final', 'pre_final2', 'post_final', 'post_final2'])

        hooks._unregister_hook('final_setup', 'Problem', pre=pre_final, post=False)

        for prob in probs:
            prob.calls = []
            for i in range(3):
                prob.run_model()

        self.assertEqual(probs[0].calls, ['post_final', 'post_final', 'post_final'])
        self.assertEqual(probs[1].calls, ['pre_final2', 'post_final', 'post_final2', 'pre_final2', 'post_final', 'post_final2', 'pre_final2', 'post_final', 'post_final2'])

        hooks._unregister_hook('final_setup', 'Problem', pre=True, post=False)

        for prob in probs:
            prob.calls = []
            for i in range(3):
                prob.run_model()

        self.assertEqual(probs[0].calls, ['post_final', 'post_final', 'post_final'])
        self.assertEqual(probs[1].calls, ['post_final', 'post_final2', 'post_final', 'post_final2', 'post_final', 'post_final2'])

    @hooks_active
    def test_multiwrap_mixed_inst_None(self):
        pre_final = make_hook('pre_final')
        post_final = make_hook('post_final')
        hooks._register_hook('final_setup', 'Problem', inst_id='problem1', ncalls=2,
                             pre=make_hook('pre_final1'), post=make_hook('post_final1'))
        hooks._register_hook('final_setup', 'Problem', inst_id='problem2',
                             pre=make_hook('pre_final2'), post=make_hook('post_final2'))
        hooks._register_hook('final_setup', 'Problem', pre=pre_final, post=post_final)

        probs = [self.build_model(f"problem{i+1}") for i in range(2)]
        for prob in probs:
            prob.calls = []
            for i in range(3):
                prob.run_model()

        self.assertEqual(probs[0].calls, ['pre_final1', 'pre_final', 'post_final1', 'post_final', 'pre_final1', 'pre_final', 'post_final1', 'post_final', 'pre_final', 'post_final'])
        self.assertEqual(probs[1].calls, ['pre_final2', 'pre_final', 'post_final2', 'post_final', 'pre_final2', 'pre_final', 'post_final2', 'post_final', 'pre_final2', 'pre_final', 'post_final2', 'post_final'])

        hooks._unregister_hook('final_setup', 'Problem', pre=pre_final, post=False)

        for prob in probs:
            prob.calls = []
            for i in range(3):
                prob.run_model()

        self.assertEqual(probs[0].calls, ['post_final', 'post_final', 'post_final'])
        self.assertEqual(probs[1].calls, ['pre_final2', 'post_final2', 'post_final', 'pre_final2', 'post_final2', 'post_final', 'pre_final2', 'post_final2', 'post_final'])

        hooks._unregister_hook('final_setup', 'Problem', pre=True, post=False)

        for prob in probs:
            prob.calls = []
            for i in range(3):
                prob.run_model()

        self.assertEqual(probs[0].calls, ['post_final', 'post_final', 'post_final'])
        self.assertEqual(probs[1].calls, ['post_final2', 'post_final', 'post_final2', 'post_final', 'post_final2', 'post_final'])

    @hooks_active
    def test_problem_hooks(self):
        hooks._register_hook('setup', 'Problem', pre=make_hook('pre_setup'), post=make_hook('post_setup'))
        hooks._register_hook('final_setup', 'Problem', pre=make_hook('pre_final'), post=make_hook('post_final'))
        hooks._register_hook('run_model', 'Problem', pre=make_hook('pre_run_model'), post=make_hook('post_run_model'))

        prob = self.build_model()

        prob.run_model()
        prob.run_model()
        prob.run_model()

        self.assertEqual(prob.calls, ['pre_setup', 'post_setup',
                                        'pre_run_model', 'pre_final', 'post_final', 'post_run_model',
                                        'pre_run_model', 'pre_final', 'post_final', 'post_run_model',
                                        'pre_run_model', 'pre_final', 'post_final', 'post_run_model',
                                        ])

        np.testing.assert_allclose(prob['comp.f_xy'], -6.0)

        hooks._unregister_hook('setup', 'Problem', pre=False)
        hooks._unregister_hook('final_setup', 'Problem')
        hooks._unregister_hook('run_model', 'Problem', post=False)
        prob.calls = []

        prob.setup()
        prob.run_model()
        prob.run_model()
        prob.run_model()

        self.assertEqual(prob.calls, ['pre_setup', 'post_run_model', 'post_run_model', 'post_run_model'])

        hooks._unregister_hook('setup', 'Problem')

        msg = "No hook found for method 'final_setup' for class 'Problem' and instance 'None'."

        hooks._unregister_hook('run_model', 'Problem')
        prob.calls = []

        prob.setup()
        prob.run_model()
        prob.run_model()

        self.assertEqual(prob.calls, [])

    @hooks_active
    def test_problem_hooks_kwargs(self):

        x0 = 33.0
        y0 = 44.0

        def set_prob_vars_hook_func(prob, **kwargs):
            if 'x0' in kwargs:
                prob['p1.x'] = kwargs['x0']
            if 'y0' in kwargs:
                prob['p2.y'] = kwargs['y0']

        hooks._register_hook('final_setup', 'Problem', pre=set_prob_vars_hook_func, x0=x0, y0=y0)

        prob = self.build_model()

        prob.run_model()

        self.assertEqual(prob['comp.x'], x0)
        self.assertEqual(prob['comp.y'], y0)

    @hooks_active
    def test_problem_hooks_pass_ret(self):

        hooks._register_hook('compute_totals', 'Problem',
                             pre=make_hook('pre_totals'), post=make_hook_pass_return('post_totals'),
                             pass_return=True)

        prob = self.build_model()

        prob.run_model()

        J = prob.compute_totals(of=['comp.f_xy'], wrt=['p1.x', 'p2.y'], return_format='array')
        self.assertEqual(len(prob.calls), 2)
        self.assertEqual(prob.calls[0], 'pre_totals')
        self.assertEqual(prob.calls[1][0], 'post_totals')
        assert_near_equal(prob.calls[1][1], J)

    @hooks_active
    def test_problem_hooks_pass_args(self):

        hooks._register_hook('compute_totals', 'Problem',
                             pre=make_hook_pass_args('pre_totals'), post=make_hook_pass_args('post_totals'),
                             pass_args=True)

        prob = self.build_model()

        prob.run_model()

        of=['comp.f_xy']
        wrt=['p1.x', 'p2.y']
        J = prob.compute_totals(of=of, wrt=wrt, return_format='array')
        self.assertEqual(len(prob.calls), 2)
        self.assertEqual(prob.calls[0],  ('pre_totals', (), {'of': ['comp.f_xy'], 'wrt': ['p1.x', 'p2.y'], 'return_format': 'array'}))
        self.assertEqual(prob.calls[1],  ('post_totals', (), {'of': ['comp.f_xy'], 'wrt': ['p1.x', 'p2.y'], 'return_format': 'array'}))

    @hooks_active
    def test_problem_hooks_pass_args_ret(self):

        hooks._register_hook('compute_totals', 'Problem',
                             pre=make_hook_pass_args('pre_totals'), post=make_hook_pass_args_pass_return('post_totals'),
                             pass_args=True, pass_return=True)

        prob = self.build_model()

        prob.run_model()

        of=['comp.f_xy']
        wrt=['p1.x', 'p2.y']
        J = prob.compute_totals(of=of, wrt=wrt, return_format='array')
        self.assertEqual(len(prob.calls), 2)
        self.assertEqual(prob.calls[0],  ('pre_totals', (), {'of': ['comp.f_xy'], 'wrt': ['p1.x', 'p2.y'], 'return_format': 'array'}))
        self.assertEqual(prob.calls[1][:3],  ('post_totals', (), {'of': ['comp.f_xy'], 'wrt': ['p1.x', 'p2.y'], 'return_format': 'array'}))
        assert_near_equal(prob.calls[1][3], J)

    @hooks_active
    def test_inherited_class_hooks(self):
        def inherited_hook_pre(prob, **kwargs):
            prob.calls.append('inherited_pre')

        def inherited_hook_post(prob, **kwargs):
            prob.calls.append('inherited_post')

        def base_hook_pre(prob, **kwargs):
            prob.calls.append('base_pre')

        def base_hook_post(prob, **kwargs):
            prob.calls.append('base_post')

        hooks._register_hook('final_setup', 'MyProblem', pre=inherited_hook_pre, post=inherited_hook_post)
        hooks._register_hook('final_setup', 'Problem', pre=base_hook_pre, post=base_hook_post)

        prob = self.build_model(prob=MyProblem())

        prob.run_model()
        prob.run_model()
        prob.run_model()

        self.assertEqual(prob.calls, ['inherited_pre', 'base_pre', 'inherited_post', 'base_post', 'inherited_pre',
                                      'base_pre', 'inherited_post', 'base_post', 'inherited_pre', 'base_pre', 'inherited_post', 'base_post'])

        hooks._unregister_hook('final_setup', 'Problem', pre=base_hook_pre, post=False)

        prob.calls = []

        prob.run_model()
        prob.run_model()

        self.assertEqual(prob.calls, ['inherited_pre', 'inherited_post', 'base_post', 'inherited_pre', 'inherited_post', 'base_post'])


class MyProblem(om.Problem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = []

    def run_model(self):
        super().run_model()


if __name__ == '__main__':
    unittest.main()
