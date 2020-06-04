import unittest
import numpy as np

import openmdao.api as om
import openmdao.utils.hooks as hooks
from openmdao.utils.assert_utils import assert_warning


class HooksTestCase(unittest.TestCase):
    def test_rewrap(self):
        # can't test this until we have hooks in Systems or Solvers
        pass

    def test_problem_hooks(self):
        def make_hook(name):
            def hook_func(prob):
                prob.calls.append(name)
            return hook_func

        hooks.use_hooks = True
        hooks._register_hook('setup', 'Problem', pre=make_hook('pre_setup'), post=make_hook('post_setup'))
        hooks._register_hook('final_setup', 'Problem', pre=make_hook('pre_final'), post=make_hook('post_final'))
        hooks._register_hook('run_model', 'Problem', pre=make_hook('pre_run_model'), post=make_hook('post_run_model'))
        try:
            prob = om.Problem()
            prob.calls = []
            model = prob.model

            model.add_subsystem('p1', om.IndepVarComp('x', 3.0))
            model.add_subsystem('p2', om.IndepVarComp('y', -4.0))
            model.add_subsystem('comp', om.ExecComp("f_xy=2.0*x+3.0*y"))

            model.connect('p1.x', 'comp.x')
            model.connect('p2.y', 'comp.y')

            prob.setup()
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

            # already removed final_setup hooks earlier, so expect a warning here
            with assert_warning(UserWarning, msg):
                hooks._unregister_hook('final_setup', 'Problem')

            hooks._unregister_hook('run_model', 'Problem')
            prob.calls = []

            prob.setup()
            prob.run_model()
            prob.run_model()

            self.assertEqual(prob.calls, [])
            self.assertEqual(len(hooks._hooks), 0)  # should be no hooks left

        finally:
            hooks.use_hooks = False
            hooks._reset_all_hooks()


if __name__ == '__main__':
    unittest.main()
