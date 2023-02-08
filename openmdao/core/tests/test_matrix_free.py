import unittest
import openmdao.api as om


class MatrixFreeTestCase(unittest.TestCase):
    def test_dyn_matfree_explicit(self):
        class MyExplComp(om.ExplicitComponent):
            def __init__(self, dyn, **kwargs):
                super().__init__(**kwargs)
                self._dyn = dyn

            def setup(self):
                self.add_input('length', val=1.)
                self.add_input('width', val=1.)
                self.add_output('area', val=1.)
                if self._dyn:
                    self.compute_jacvec_product = self._my_compute_jacvec_product
                    self.matrix_free = True

            def compute(self, inputs, outputs):
                outputs['area'] = inputs['length'] * inputs['width']

            def _my_compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
                if mode == 'fwd':
                    if 'area' in d_outputs:
                        if 'length' in d_inputs:
                            d_outputs['area'] += inputs['width'] * d_inputs['length']
                        if 'width' in d_inputs:
                            d_outputs['area'] += inputs['length'] * d_inputs['width']
                elif mode == 'rev':
                    if 'area' in d_outputs:
                        if 'length' in d_inputs:
                            d_inputs['length'] += inputs['width'] * d_outputs['area']
                        if 'width' in d_inputs:
                            d_inputs['width'] += inputs['length'] * d_outputs['area']

        p = om.Problem()
        comp = p.model.add_subsystem('comp', MyExplComp(dyn=False))
        comp_dyn = p.model.add_subsystem('comp_dyn', MyExplComp(dyn=True))
        p.setup()

        p.run_model()

        self.assertEqual(comp.matrix_free, False)
        self.assertEqual(comp_dyn.matrix_free, True)


    def test_dyn_matfree_implicit(self):
        class MyImplComp(om.ImplicitComponent):
            def __init__(self, dyn, **kwargs):
                super().__init__(**kwargs)
                self._dyn = dyn

            def setup(self):
                self.add_input('b')
                self.add_output('a')
                if self._dyn:
                    self.apply_linear = self._my_apply_linear
                    self.matrix_free = True

            def apply_nonlinear(self, inputs, outputs, residuals):
                residuals['a'] = 6 * outputs['a'] + 1 * inputs['b']

            def _my_apply_linear(self, inputs, outputs,
                            d_inputs, d_outputs, d_residuals, mode):
                if mode == 'fwd':
                    if 'a' in d_residuals:
                        if 'a' in d_outputs:
                            d_residuals['a'] += 6 * d_outputs['a']
                        if 'b' in d_inputs:
                            d_residuals['a'] += 1 * d_inputs['b']
                if mode == 'rev':
                    if 'a' in d_residuals:
                        if 'a' in d_outputs:
                            d_outputs['a'] += 6 * d_residuals['a']
                        if 'b' in d_inputs:
                            d_inputs['b'] += 1 * d_residuals['a']

            def solve_linear(self, d_outputs, d_residuals, mode):
                if mode == 'fwd':
                    d_outputs['a'] = 1./6. * d_residuals['a']
                elif mode == 'rev':
                    d_residuals['a'] = 1./6. * d_outputs['a']

        p = om.Problem()
        comp = p.model.add_subsystem('comp', MyImplComp(dyn=False))
        comp_dyn = p.model.add_subsystem('comp_dyn', MyImplComp(dyn=True))
        p.setup()

        p.run_model()

        self.assertEqual(comp.matrix_free, False)
        self.assertEqual(comp_dyn.matrix_free, True)

