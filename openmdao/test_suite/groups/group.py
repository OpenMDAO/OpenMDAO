"""Define the test group classes."""
from __future__ import division, print_function

from openmdao.api import Group


class ParametericTestGroup(Group):
    def __init__(self, **kwargs):

        self.expected_totals = None
        self.total_of = None
        self.total_wrt = None
        self.expected_values = None

        super(ParametericTestGroup, self).__init__(**kwargs)