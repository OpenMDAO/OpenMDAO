from __future__ import division
import numpy
import unittest

from .general_problem import GeneralProblem

class Test(unittest.TestCase):

    def setUp(self):
        self.gps = []

        ngroup_level = [4,2,1]
        gp = GeneralProblem(ngroup_level)
        self.gps.append(gp)

        ngroup_level = [4,1]
        gp = GeneralProblem(ngroup_level)
        self.gps.append(gp)

    def test(self):
        pass
        #self.gps[0].print_all()
        #self.gps[1].print_all()



if __name__ == '__main__':
    unittest.main()
