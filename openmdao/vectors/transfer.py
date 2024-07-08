"""Define the Transfer base class."""


class Transfer(object):
    """
    Base Transfer class.

    Parameters
    ----------
    in_vec : <Vector>
        Pointer to the input vector.
    out_vec : <Vector>
        Pointer to the output vector.
    in_inds : int ndarray
        Input indices for the transfer.
    out_inds : int ndarray
        Output indices for the transfer.

    Attributes
    ----------
    _in_inds : int ndarray
        input indices for the transfer.
    _out_inds : int ndarray
        output indices for the transfer.
    """

    def __init__(self, in_vec, out_vec, in_inds, out_inds):
        """
        Initialize all attributes.
        """
        self._in_inds = in_inds
        self._out_inds = out_inds

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


def _get_xfer_tgt(group, sysname):
    for i, cycle in enumerate(group._cycles):
        if sysname in cycle:
            return f"{group.name}_cycle#{i}"
    else:
        return sysname
