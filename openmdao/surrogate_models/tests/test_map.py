from openmdao.api import Group, Problem, Component, MetaModel, NearestNeighbor
from openmdao.devtools.testutil import assert_rel_error

import numpy as np
import unittest

class CompressorMap(Group):

    def __init__(self):
        super(CompressorMap, self).__init__()

        compmap = self.add('compmap', MetaModel())

        compmap.add_param('Nc', val=1.0)
        compmap.add_param('Rline', val=2.0)
        compmap.add_param('alpha', val=0.0)

        compmap.add_output('PR', val=1.0, surrogate=NearestNeighbor(interpolant_type='linear'))
        compmap.add_output('eff', val=1.0, surrogate=NearestNeighbor(interpolant_type='linear'))
        compmap.add_output('Wc', val=1.0, surrogate=NearestNeighbor(interpolant_type='linear'))


class TestMap(unittest.TestCase):

    def test_comp_map(self):
        p = Problem()
        p.root = CompressorMap()
        p.setup(check=False)

        Nc = np.array([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1])
        Rline = np.array([1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0])
        alpha = np.array([0.0, 1.0])
        Nc_mat, Rline_mat, alpha_mat = np.meshgrid(Nc, Rline, alpha, sparse=False)

        p['compmap.train:Nc'] = Nc_mat.flatten()
        p['compmap.train:Rline'] = Rline_mat.flatten()
        p['compmap.train:alpha'] = alpha_mat.flatten()

        p['compmap.train:PR'] = p['compmap.train:Nc']*p['compmap.train:Rline']+p['compmap.train:alpha']
        p['compmap.train:eff'] = p['compmap.train:Nc']*p['compmap.train:Rline']**2+p['compmap.train:alpha']
        p['compmap.train:Wc'] = p['compmap.train:Nc']**2*p['compmap.train:Rline']**2+p['compmap.train:alpha']

        p['compmap.Nc'] = 0.9
        p['compmap.Rline'] = 2.0
        p['compmap.alpha'] = 0.0
        p.run()

        tol = 1e-1
        assert_rel_error(self, p['compmap.PR'], p['compmap.Nc']*p['compmap.Rline']+p['compmap.alpha'], tol)
        assert_rel_error(self, p['compmap.eff'], p['compmap.Nc']*p['compmap.Rline']**2+p['compmap.alpha'], tol)
        assert_rel_error(self, p['compmap.Wc'], p['compmap.Nc']**2*p['compmap.Rline']**2+p['compmap.alpha'], tol)

        p['compmap.Nc'] = 0.95
        p['compmap.Rline'] = 2.1
        p['compmap.alpha'] = 0.0
        p.run()

        assert_rel_error(self, p['compmap.PR'], p['compmap.Nc']*p['compmap.Rline']+p['compmap.alpha'], tol)
        assert_rel_error(self, p['compmap.eff'], p['compmap.Nc']*p['compmap.Rline']**2+p['compmap.alpha'], tol)
        assert_rel_error(self, p['compmap.Wc'], p['compmap.Nc']**2*p['compmap.Rline']**2+p['compmap.alpha'], tol)

if __name__ == "__main__":
    unittest.main()
