"""Define the Transfer base class."""

from __future__ import division


class Transfer(object):
    """
    Base Transfer class.

    Attributes
    ----------
    _in_inds : int ndarray
        input indices for the transfer.
    _out_inds : int ndarray
        output indices for the transfer.
    _comm : MPI.Comm or FakeComm
        communicator of the system that owns this transfer.
    """

    # If this is True it will force allocation of rev transfers.  We set this when using
    # coloring when rev coloring is active.
    _need_reverse = False

    def __init__(self, in_vec, out_vec, in_inds, out_inds, comm):
        """
        Initialize all attributes.

        Parameters
        ----------
        in_vec : <Vector>
            pointer to the input vector.
        out_vec : <Vector>
            pointer to the output vector.
        in_inds : int ndarray
            input indices for the transfer.
        out_inds : int ndarray
            output indices for the transfer.
        comm : MPI.Comm or <FakeComm>
            communicator of the system that owns this transfer.
        """
        self._in_inds = in_inds
        self._out_inds = out_inds
        self._comm = comm

        self._initialize_transfer(in_vec, out_vec)

    def __str__(self):
        """
        Return a string representation of the Transfer object.

        Returns
        -------
        str
            String rep of this object.
        """
        try:
            return "%s(in=%s, out=%s" % (self.__class__.__name__, self._in_inds, self._out_inds)
        except Exception as err:
            return "<error during call to Transfer.__str__: %s" % err

    def _initialize_transfer(self, in_vec, out_vec):
        """
        Set up the transfer; do any necessary pre-computation.

        Optionally implemented by the subclass.

        Parameters
        ----------
        in_vec : <Vector>
            reference to the input vector.
        out_vec : <Vector>
            reference to the output vector.
        """
        pass

    def __call__(self, in_vec, out_vec, mode='fwd'):
        """
        Perform transfer.

        Must be implemented by the subclass.

        Parameters
        ----------
        in_vec : <Vector>
            pointer to the input vector.
        out_vec : <Vector>
            pointer to the output vector.
        mode : str
            'fwd' or 'rev'.
        """
        pass
