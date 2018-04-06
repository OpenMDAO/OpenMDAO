"""
Case generator for Design-of-Experiments Driver implementing the Uniform method..
"""
import numpy as np

from six import moves, iteritems

from openmdao.drivers.doe_driver import DOEGenerator


class UniformGenerator(DOEGenerator):
    """
    DOE case generator implementing the Uniform method.

    Attributes
    ----------
    _num_samples : int
        The number of samples to run.
    _seed : int or None
        Random seed.
    """

    def __init__(self, num_samples=1, seed=None):
        """
        Initialize the UniformGenerator.

        Parameters
        ----------
        num_samples : int, optional
            The number of samples to run. Defaults to 1.

        seed : int or None, optional
            Seed for randon number generator.

        num_par_doe : int, optional
            The number of DOE cases to run concurrently.  Defaults to 1.

        load_balance : bool, Optional
            If True, use rank 0 as master and load balance cases among all of the
            other ranks. Defaults to False.
        """
        super(UniformGenerator, self).__init__()

        self._num_samples = num_samples
        self._seed = seed

    def __call__(self, design_vars):
        """
        Generate case.

        Parameters
        ----------
        design_vars : dict
            Dictionary of design variables for which to generate values.

        Yields
        ------
        dict
            Dictionary of input values for the case.
        """
        if self._seed is not None:
            np.random.seed(self._seed)

        for i in moves.range(self._num_samples):
            sample = {}

            for (name, meta) in iteritems(design_vars):
                values = []

                for k in range(meta['size']):
                    lower = meta['lower']
                    if isinstance(lower, np.ndarray):
                        lower = lower[k]

                    upper = meta['upper']
                    if isinstance(upper, np.ndarray):
                        upper = upper[k]

                    values.append(np.random.uniform(lower, upper))

                sample[name] = np.array(values)

            yield sample
