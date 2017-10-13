from __future__ import print_function
import unittest
from six.moves import range
import numpy as np

import time
from openmdao.api import Problem, Group, ExplicitComponent, IndepVarComp, ExecComp


class Plus(ExplicitComponent):
    def __init__(self, adder):
        super(Plus, self).__init__()
        self.adder = float(adder)

    def setup(self):
        self.add_input('x', np.random.random())
        self.add_output('f1', shape=1)

    def compute(self, inputs, outputs):
        outputs['f1'] = inputs['x'] + self.adder

class Times(ExplicitComponent):
    def __init__(self, scalar):
        super(Times, self).__init__()
        self.scalar = float(scalar)

    def setup(self):
        self.add_input('f1', np.random.random())
        self.add_output('f2', shape=1)

    def compute(self, inputs, outputs):
        outputs['f2'] = inputs['f1'] + self.scalar

class Point(Group):

    def __init__(self, adder, scalar):
        super(Point, self).__init__()
        self.adder = adder
        self.scalar = scalar

    def setup(self):
        self.add_subsystem('plus', Plus(self.adder), promotes=['*'])
        self.add_subsystem('times', Times(self.scalar), promotes=['*'])

class Summer(ExplicitComponent):

    def __init__(self, size):
        super(Summer, self).__init__()
        self.size = size

    def setup(self):
        for i in range(self.size):
            self.add_input('y%d'%i, 0.)

        self.add_output('total', shape=1)

    def compute(self, inputs, outputs):
        tot = 0
        for i in range(self.size):
            tot += inputs['y%d'%i]
        outputs['total'] = tot

class MultiPoint(Group):

    def __init__(self, adders, scalars):
        super(MultiPoint, self).__init__()
        self.adders = adders
        self.scalars = scalars

    def setup(self):

        size = len(self.adders)

        for i,(a,s) in enumerate(zip(self.adders, self.scalars)):
            c_name = 'p%d'%i
            self.add_subsystem(c_name, Point(a,s))
            self.connect(c_name+'.f2','aggregate.y%d'%i)

        self.add_subsystem('aggregate', Summer(size))

class BM(unittest.TestCase):
    """A few 'brute force' multipoint cases (1K, 2K, 5K)"""

    def _setup_bm(self, npts):

        size = npts
        adders =  np.random.random(size)
        scalars = np.random.random(size)

        prob = Problem(MultiPoint(adders, scalars))
        prob.setup(check=False)

        return prob

    def benchmark_setup_2K(self):
        for i in range(3):
            p = self._setup_bm(2000)
            p.final_setup()

    def benchmark_setup_1K(self):
        for i in range(3):
            p = self._setup_bm(1000)
            p.final_setup()

    def benchmark_run_2K(self):
        for i in range(3):
            p = self._setup_bm(2000)
            p.run_model()

    def benchmark_run_1K(self):
        for i in range(3):
            p = self._setup_bm(1000)
            p.run_model()
