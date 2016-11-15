"""Temporary run file for the test components."""
from __future__ import division, print_function
import numpy

from six import iteritems
from six.moves import range
from collections import OrderedDict

import itertools
import unittest

from openmdao.api import Problem
from openmdao.test_suite.components.implicit_components \
    import TestImplCompNondLinear
from openmdao.test_suite.components.explicit_components \
    import TestExplCompNondLinear
from openmdao.test_suite.groups.group import TestGroupFlat
from openmdao.api import DefaultVector, NewtonSolver, ScipyIterativeSolver
from openmdao.parallel_api import PETScVector


class CompTestCase(unittest.TestCase):

    def test_comps(self):
        for key in itertools.product(
                [TestImplCompNondLinear, TestExplCompNondLinear],
                #[TestImplCompNondLinear],
                [DefaultVector, PETScVector],
                ['implicit', 'explicit'],
                range(1, 3),
                range(1, 3),
                [(1,), (2,), (2, 1), (1, 2)],
                ):
            Component = key[0]
            Vector = key[1]
            connection_type = key[2]
            num_var = key[3]
            num_sub = key[4]
            var_shape = key[5]

            print_str = ('%s %s %s %i vars %i subs %s' % (
                Component.__name__,
                Vector.__name__,
                connection_type,
                num_var, num_sub,
                str(var_shape),
            ))

            #print(print_str)

            group = TestGroupFlat(num_sub=num_sub, num_var=num_var,
                                  var_shape=var_shape,
                                  connection_type=connection_type,
                                  Component=Component,
                                  derivatives='dict')
            prob = Problem(group).setup(Vector)
            prob.root.nl_solver = NewtonSolver(
                subsolvers={'linear': ScipyIterativeSolver(
                    ilimit=100,
                )}
            )
            prob.root.suppress_solver_output = True
            fail, rele, abse = prob.run()
            if fail:
                self.fail('re %f ; ae %f ;  ' % (rele, abse) + print_str)


if __name__ == '__main__':
    unittest.main()
