import unittest
from openmdao.utils.testing_utils import use_tempdirs

from openmdao.visualization.opt_report.constraint_plot import in_or_out_of_bounds_plot

@use_tempdirs
class TestConstraintPlot(unittest.TestCase):

    def test_constraint_plot(self):
        # Just make sure it doesn't crash

        # must handle 5 cases if both upper and lower are given:
        #  - value much less than lower
        in_or_out_of_bounds_plot(-10, 2., 4.)

        #  - value a little less than lower
        in_or_out_of_bounds_plot(1, 2., 4.)

        #  - value between lower and upper
        in_or_out_of_bounds_plot(3, 2., 4.)

        #  - value a little greater than upper
        in_or_out_of_bounds_plot(5, 2., 4.)

        #  - value much greater than upper
        in_or_out_of_bounds_plot(10, 2., 4.)

        # also need to handle one-sided constraints
        #    where only one of lower and upper is given
        in_or_out_of_bounds_plot(-2.5, 2., None)
        in_or_out_of_bounds_plot(5, 2., None)
        in_or_out_of_bounds_plot(3, None, 2.0 )
        in_or_out_of_bounds_plot(-1, None, 2.0 )
