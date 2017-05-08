"""
Class definition for SurrogateModel, the base class for all surrogate models.
"""

class SurrogateModel(object):
    """
    Base class for surrogate models.
    """

    def __init__(self):
        self.trained = False

    def train(self, x, y):
        self.trained = True

    def predict(self, x):
        if not self.trained:
            msg = "{0} has not been trained, so no prediction can be made."\
                .format(type(self).__name__)
            raise RuntimeError(msg)

    def linearize(self, x):

        msg = "{0} has not defined a jacobian method." \
            .format(type(self).__name__)
        raise RuntimeError(msg)


class MultiFiSurrogateModel(SurrogateModel):
    """
    Base class for surrogate models using multi-fiddelity training data
    """

    def train(self, x, y):
        super(MultiFiSurrogateModel, self).train(x, y)
        self.train_multifi([x], [y])

    def train_multifi(self, x, y):
        """Trains the surrogate model, based on the given
        multi-fidelity training data.

        x: list of (m samples, n inputs) ndarrays
            Values representing the multi-fidelity training case inputs.
        y: list of ndarray
            output training values which corresponds to the multi-fidelity
            training case input given by x.
        """
