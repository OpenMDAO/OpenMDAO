
import numpy as np

from openmdao.vectors.vector import _full_slices

# What is a vector?
#  ORDER INDEPENDENT
#   - provides a mapping from var name to var value (shaped)
#
#  ORDER DEPENDENT
#   - provides a mapping from var name to a flat array view/slice/range
#   - provides access to a flat array representing all local variables found in a given system
#   - provides access to a virtual array representing all variables (local and remote) found in a
#     given system
#        - can obtain this using offset/size of first var and last var since vars are ordered.


# Where are vector vars accessed by var name vs. by full array?
#  - in components - by (relative) var name
#  - at top level (from problem get/set) by abs var name
#  - problem.compute_jacvec_product by prom var name AND full array
#  - problem.check_partials
#  - matvec_context sets internal set of allowed names
#  - system._abs_get_val
#  - retrieve_data_of_kind
#  - solvers access by full array


_DEFAULT_META_ = {
    'size': 0,
    'shape': None,
    'value': None,
    'discrete': False,
}


class UnorderedVarCollection(object):
    """
    A class to represent the input and output nonlinear vectors prior to vector setup.

    Variables can be set/retrieved by name, but there is no full array available since
    the variables are not ordered.
    """

    def __init__(self):
        self.pathname = None
        self._meta = {}

    def __contains__(self, name):
        return name in self._meta

    def __getitem__(self, name):
        return self._meta[name]['value']

    def __setitem__(self, name, value):
        self._meta[name]['value'] = value

    def _set_parent(self, parent):
        self.pathname = parent.pathname
        # compute abs2meta entries
        # compute prom2meta entries? (or just store mapping and wait until
        #                             prom_context to avoid creating a map we don't need)
        for rel, meta in self._var2meta.items():
            pass

    def _abs_context(self):
        """
        Switch the lookup dict to use absolute names.
        """
        self._meta = self._abs2meta

    def _rel_context(self):
        """
        Switch the lookup dict to use relative names.
        """
        self._meta = self._var2meta

    def size_shape_iter(self):
        """
        Return tuples of the form (name, size, shape).

        This will be used to build Vector instances.
        """
        pass

    def get_meta(self, name, meta_name):
        """
        Return misc metadata for the named variable.
        """
        return self._meta[name][meta_name]

    def add_var(self, name, **kwargs):
        # must add to _var2meta since var name is all we have early on
        self._var2meta[name] = _DEFAULT_META_.copy()
        self._var2meta[name].update(kwargs)
        # when do we populate abs2meta and prom2meta?

    def reshape_var(self, name, shape):
        meta = self._meta[name]
        meta['shape'] = shape
        val = meta['value']
        if val is None:
            meta['value'] = np.ones(shape)
        else:
            val = np.asarray(val)
            if val.shape != shape:
                if val.size == np.prod(shape):
                    meta['value'] = val.reshape(shape)
                else:
                    meta['value'] = np.ones(shape)

    def append(self, vec):
        """
        Add the variables from the given UnorderedVarCollection to this UnorderedVarCollection.

        Parameters
        ----------
        vec : UnorderedVarCollection
            UnorderedVarCollection being added.
        """
        pass

    def set_var(self, name, val, idxs=_full_slice):
        """
        Set the value corresponding to the named variable, with optional indexing.

        Parameters
        ----------
        name : str
            The name of the variable.
        val : float or ndarray
            Scalar or array to set data array to.
        idxs : int or slice or tuple of ints and/or slices.
            The locations where the data array should be updated.
        """
        pass

    @property
    def msginfo(self):
        """
        Our instance pathname, if available, or our class name.  For use in error messages.

        Returns
        -------
        str
            Either our instance pathname or class name.
        """
        if self.pathname == '':
            return f"(<model>)"
        if self.pathname is not None:
            return f"{self.pathname}"
        return f"{type(self).__name__}"

    def asarray(self):
        """
        Raise error indicating that ordering of variables has not yet taken place.
        """
        raise RuntimeError(f"{self.msginfo}: asarray is not allowed yet because variables are "
                           "still unordered. Variables will be ordered later in the setup process.")
