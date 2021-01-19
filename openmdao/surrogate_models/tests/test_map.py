from openmdao.api import Group, Problem, MetaModelUnStructuredComp, NearestNeighbor
from openmdao.utils.assert_utils import assert_near_equal

import numpy as np
import unittest


class CompressorMap(MetaModelUnStructuredComp):

    def __init__(self):
        super().__init__()

        self.add_input('Nc', val=1.0)
        self.add_input('Rline', val=2.0)
        self.add_input('alpha', val=0.0)

        self.add_output('PR', val=1.0, surrogate=NearestNeighbor(interpolant_type='linear'))
        self.add_output('eff', val=1.0, surrogate=NearestNeighbor(interpolant_type='linear'))
        self.add_output('Wc', val=1.0, surrogate=NearestNeighbor(interpolant_type='linear'))


class TestMap(unittest.TestCase):

    def test_comp_map(self):
        # create compressor map and save reference to options (for training data)
        c = CompressorMap()
        m = c.options

        # add compressor map to problem
        p = Problem()
        p.model.add_subsystem('compmap', c)
        p.setup()

        # train metamodel
        Nc = np.array([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1])
        Rline = np.array([1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0])
        alpha = np.array([0.0, 1.0])
        Nc_mat, Rline_mat, alpha_mat = np.meshgrid(Nc, Rline, alpha, sparse=False)

        m['train:Nc'] = Nc_mat.flatten()
        m['train:Rline'] = Rline_mat.flatten()
        m['train:alpha'] = alpha_mat.flatten()

        m['train:PR'] = m['train:Nc']*m['train:Rline']+m['train:alpha']
        m['train:eff'] = m['train:Nc']*m['train:Rline']**2+m['train:alpha']
        m['train:Wc'] = m['train:Nc']**2*m['train:Rline']**2+m['train:alpha']

        # check predicted values
        p['compmap.Nc'] = 0.9
        p['compmap.Rline'] = 2.0
        p['compmap.alpha'] = 0.0
        p.run_model()

        tol = 1e-1
        assert_near_equal(p['compmap.PR'], p['compmap.Nc']*p['compmap.Rline']+p['compmap.alpha'], tol)
        assert_near_equal(p['compmap.eff'], p['compmap.Nc']*p['compmap.Rline']**2+p['compmap.alpha'], tol)
        assert_near_equal(p['compmap.Wc'], p['compmap.Nc']**2*p['compmap.Rline']**2+p['compmap.alpha'], tol)

        p['compmap.Nc'] = 0.95
        p['compmap.Rline'] = 2.1
        p['compmap.alpha'] = 0.0
        p.run_model()

        assert_near_equal(p['compmap.PR'], p['compmap.Nc']*p['compmap.Rline']+p['compmap.alpha'], tol)
        assert_near_equal(p['compmap.eff'], p['compmap.Nc']*p['compmap.Rline']**2+p['compmap.alpha'], tol)
        assert_near_equal(p['compmap.Wc'], p['compmap.Nc']**2*p['compmap.Rline']**2+p['compmap.alpha'], tol)


if __name__ == "__main__":
    unittest.main()
