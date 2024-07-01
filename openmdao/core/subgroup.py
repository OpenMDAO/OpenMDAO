

class SolvedGroup(object):
    """
    A System or Group of Systems that is solved as a group.
    """

    def __init__(self, parent, systems):
        pass
        # _doutputs
        # _dresiduals
        # _residuals
        # _outputs
        # _inputs
        # comm
        # under_complex_step


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

