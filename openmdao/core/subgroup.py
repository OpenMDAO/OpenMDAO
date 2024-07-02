from openmdao.core.group import Group

class UnnamedGroup(Group):
    """
    A System or Group of Systems that is solved as a group.
    """

    def __init__(self, parent, subsystems):
        pass
        # _doutputs
        # _dresiduals
        # _residuals
        # _outputs
        # _inputs
        # comm
        # under_complex_step
        self._setup_from_parent(parent)

    def setup_from_parent(self, parent):
        # take all setup data structures from parent (maybe not even copy)
        # parent must also be updated to include this group and remove the subsystems
        # contianed in this group.
        pass

    def _solve_nonlinear(self):
        pass

    def _solve_linear(self, subsystem):
        pass

    def _get_matvec_scope():
        pass





class VecSlice(object):
    def __init__(self, vec, start, end):
        pass

    def asarray():
        pass

    def set_val(self, value):
        pass

    def __getitem__(self, key):
        pass

    def __setitem__(self, key, value):
        pass

