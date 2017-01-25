"""Define the base Jacobian class."""
from __future__ import division
import numpy
from scipy.sparse import coo_matrix, csr_matrix
from six.moves import range

from openmdao.utils.generalized_dict import OptionsDictionary


class Jacobian(object):
    """Base Jacobian class.

    This class provides a dictionary interface for sub-Jacobians and
    performs matrix-vector products when apply_linear is called.

    Attributes
    ----------
    _top_name : str
        name of the system at which we allocate the global Jacobian.
    _assembler : <Assembler>
        pointer to the assembler.
    _system : <System>
        pointer to the system that is currently operating on this Jacobian.
    _subjacs : dict
        dictionary containing the user-supplied sub-Jacobians.
    _int_mtx : <Matrix>
        global internal Jacobian.
    _ext_mtx : <Matrix>
        global external Jacobian.
    _keymap : dict
        Mapping of original (output, input) key to (output, source) in cases
        where the input has src_indices.
    _iter_list : [(out_name, in_name), ...]
        list of output-input pairs to iterate over.
    options : <OptionsDictionary>
        options dictionary.
    """

    def __init__(self, **kwargs):
        """Initialize all attributes.

        Args
        ----
        **kwargs : dict
            options dictionary.
        """
        self._top_name = None
        self._assembler = None
        self._system = None

        self._subjacs = {}
        self._int_mtx = None
        self._ext_mtx = None
        self._keymap = {}
        self._iter_list = None

        self.options = OptionsDictionary()
        self.options.update(kwargs)

    def _key2shape(self, key):
        """Return shape of sub-jacobian for variables making up the key tuple.

        This assumes that no inputs and outputs share the same name,
        so it should only be called from a Component, never from a Group.

        Args
        ----
        key : (str, str)
            output name, input name of sub-Jacobian.

        Returns
        -------
        out_size : int
            local size of the output variable.
        in_size : int
            local size of the input variable.
        """
        out_name, in_name = key
        if in_name in self._system._inputs:
            in_size = self._system._inputs._views_flat[in_name].size
        else:
            in_size = self._system._outputs[in_name].size

        return self._system._outputs._views_flat[out_name].size, in_size

    def _key2unique(self, key):
        """Map output-input local name pair to a unique key.

        This should only be called when self._system is a Component or
        key parts are all outputs.  If the key contains an input name, that
        will not be unique in a Group context.

        Args
        ----
        key : (str, str)
            output name, input name of sub-Jacobian.

        Returns
        -------
        out_path : str
            pathname of output variable.
        in_path : str
            pathname of input variable.
        """
        out_paths = self._system._var_name2path[key[0]]
        in_paths = self._system._var_name2path[key[1]]
        assert (len(in_paths) == 1 and len(out_paths) == 1)
        return (out_paths[0], in_paths[0])

    def _negate(self, key):
        """Multiply this sub-Jacobian by -1.0, for explicit variables.

        Args
        ----
        key : (str, str)
            output name, input name of sub-Jacobian.
        """
        ukey = self._key2unique(key)
        jac = self._subjacs[ukey]

        if isinstance(jac, numpy.ndarray):
            self._subjacs[ukey] = -jac
        elif isinstance(jac, (coo_matrix, csr_matrix)):
            self._subjacs[ukey].data *= -1.0  # DOK not supported
        elif len(jac) == 3:
            self._subjacs[ukey][0] *= -1.0
        elif len(jac) == 2:
            # In this case, negation is not necessary because sparse FD
            # works on the residuals which already contains the negation
            pass

    def _precompute_iter(self):
        """Assemble list of output-input pairs by name.

        Returns
        -------
        list
            List of (output, input) pairs found in the jacobian
            for the current System.
        """
        system = self._system
        pathdict = system._var_pathdict
        pathnames = system._var_allprocs_pathnames
        out_offset = system._var_allprocs_range['output'][0]
        in_offset = system._var_allprocs_range['input'][0]

        iter_list = []
        for re_ind in system._var_myproc_indices['output']:
            re_path = pathnames['output'][re_ind - out_offset]

            for out_ind in system._var_myproc_indices['output']:
                out_path = pathnames['output'][out_ind - out_offset]

                if (re_path, out_path) in self._subjacs:
                    iter_list.append((pathdict[re_path].name,
                                      pathdict[out_path].name))

            for in_ind in system._var_myproc_indices['input']:
                in_path = pathnames['input'][in_ind - in_offset]

                if (re_path, in_path) in self._subjacs:
                    iter_list.append((pathdict[re_path].name,
                                      pathdict[in_path].name))

        return iter_list

    def __contains__(self, key):
        """Map output-input pairs names to indices.

        Args
        ----
        key : (str, str)
            output name, input name of sub-Jacobian.

        Returns
        -------
        boolean
            return whether sub-Jacobian has been defined.
        """
        return self._key2unique(key) in self._subjacs

    def __iter__(self):
        """Return iterator from pre-computed _iter_list.

        Returns
        -------
        listiterator
            iterator returning (out_name, in_name) pairs.
        """
        if self._iter_list is None:
            self._iter_list = self._precompute_iter()
        return iter(self._iter_list)

    def __setitem__(self, key, jac):
        """Set sub-Jacobian.

        Args
        ----
        key : (str, str)
            output name, input name of sub-Jacobian.
        jac : int or float or ndarray or sparse matrix
            sub-Jacobian as a scalar, vector, array, or AIJ list or tuple.
        """
        system = self._system
        shape = self._key2shape(key)
        ukey = self._key2unique(key)
        typ = system._var_pathdict[ukey[1]].typ

        if numpy.isscalar(jac) or isinstance(jac, numpy.ndarray):
            jac = numpy.atleast_2d(jac).reshape(shape)
            # numpy.promote_types will choose the smallest dtype that can contain both arguments
            safe_dtype = numpy.promote_types(jac.dtype, float)
            jac = jac.astype(safe_dtype, copy=False)
        elif isinstance(jac, (coo_matrix, csr_matrix)):
            pass
        elif isinstance(jac, (tuple, list)):
            if len(jac) != 3:
                raise ValueError("Sub-jacobian of type '%s' for key %s has "
                                 "the wrong size (%d)." %
                                 (type(jac).__name__, key, len(jac)))
            if isinstance(jac, tuple):
                jac = list(jac)
        else:
            raise TypeError("Sub-jacobian of type '%s' for key %s is "
                            "not supported." % (type(jac).__name__, key))

        ind = system._var_myproc_names['output'].index(key[0])
        r_factor = system._scaling_to_norm['residual'][ind, 1]

        ind = system._var_myproc_names[typ].index(key[1])
        c_factor = system._scaling_to_norm[typ][ind, 1]

        if r_factor != c_factor:
            if isinstance(jac, numpy.ndarray):
                jac *= r_factor / c_factor
            elif isinstance(jac, (coo_matrix, csr_matrix)):
                jac.data *= r_factor / c_factor
            elif len(jac) == 3:
                jac[0] *= r_factor / c_factor

        self._subjacs[ukey] = jac

    def __getitem__(self, key):
        """Get sub-Jacobian.

        Args
        ----
        key : (str, str)
            output name, input name of sub-Jacobian.

        Returns
        -------
        jac : ndarray or spmatrix or list[3]
            sub-Jacobian as an array, sparse mtx, or AIJ/IJ list or tuple.
        """
        system = self._system
        ukey = self._key2unique(key)
        jac = self._subjacs[ukey]
        typ = system._var_pathdict[ukey[1]].typ

        ind = system._var_myproc_names['output'].index(key[0])
        r_factor = system._scaling_to_phys['residual'][ind, 1]

        ind = system._var_myproc_names[typ].index(key[1])
        c_factor = system._scaling_to_phys[typ][ind, 1]

        if r_factor != c_factor:
            if isinstance(jac, numpy.ndarray):
                jac *= r_factor / c_factor
            elif isinstance(jac, (coo_matrix, csr_matrix)):
                jac.data *= r_factor / c_factor
            elif len(jac) == 3:
                jac[0] *= r_factor / c_factor

        return jac

    def _initialize(self):
        """Allocate the global matrices."""
        pass

    def _update(self):
        """Read the user's sub-Jacobians and set into the global matrix."""
        pass

    def _apply(self, d_inputs, d_outputs, d_residuals, mode):
        """Compute matrix-vector product.

        Args
        ----
        d_inputs : Vector
            inputs linear vector.
        d_outputs : Vector
            outputs linear vector.
        d_residuals : Vector
            residuals linear vector.
        mode : str
            'fwd' or 'rev'.
        """
        pass
