"""Define the Component class."""

from __future__ import division

import fnmatch
import numpy
from itertools import product
from six import string_types, iteritems
from scipy.sparse import issparse

from openmdao.core.system import System, PathData
from openmdao.jacobians.global_jacobian import SUBJAC_META_DEFAULTS


class Component(System):
    """Base Component class; not to be directly instantiated.

    Attributes
    ----------
    _var2meta : dict
        A mapping of local variable name to its metadata.
    """

    def __init__(self, **kwargs):
        """Initialize all attributes.

        Args
        ----
        **kwargs: dict of keyword arguments
            available here and in all descendants of this system.
        """
        super(Component, self).__init__(**kwargs)
        self._var2meta = {}

    def add_input(self, name, val=1.0, shape=None, indices=None, units=None, desc='', var_set=0):
        """Add an input variable to the component.

        Args
        ----
        name : str
            name of the variable in this component's namespace.
        val : float or list or tuple or ndarray
            The initial value of the variable being added in user-defined units. Default is 1.0.
        shape : int or tuple or list or None
            Shape of this variable, only required if indices not provided and val is not an array.
            Default is None.
        indices : int or list of ints or tuple of ints or int ndarray or None
            The indices of the source variable to transfer data from.
            If val is given as an array_like object, the shapes of val and indices must match.
            A value of None implies this input depends on all entries of source. Default is None.
        units : str or None
            Units in which this input variable will be provided to the component during execution.
            Default is None, which means it has no units.
        desc : str
            description of the variable
        var_set : hashable object
            For advanced users only. ID or color for this variable, relevant for reconfigurability.
            Default is 0.
        """
        # First, type check all arguments
        if not isinstance(name, str):
            raise TypeError('The name argument should be a string')
        if not numpy.isscalar(val) and not isinstance(val, (list, tuple, numpy.ndarray)):
            raise TypeError('The val argument should be a float, list, tuple, or ndarray')
        if shape is not None and not isinstance(shape, (int, tuple, list)):
            raise TypeError('The shape argument should be an int, tuple, or list')
        if indices is not None and not isinstance(indices, (int, list, tuple, numpy.ndarray)):
            raise TypeError('The indices argument should be an int, list, tuple, or ndarray')
        if units is not None and not isinstance(units, str):
            raise TypeError('The units argument should be a str or None')

        if shape is not None:
            if isinstance(shape, int):
                shape = (shape,)
            elif isinstance(shape, list):
                shape = tuple(shape)
        # Next check that shapes are consistent
        if not numpy.isscalar(val):
            val_shape = numpy.atleast_1d(val).shape
            # 1. val and shape
            if shape is not None and val_shape != shape:
                raise ValueError('The val argument is an array, but val.shape != shape.')
            # 2. val and indices
            if indices is not None and val_shape != numpy.atleast_1d(indices).shape:
                raise ValueError('The val and indices are arrays, but val.shape != indices.shape.')
        if shape is not None:
            # 3. shape and indices
            if indices is not None and shape != numpy.atleast_1d(indices).shape:
                raise ValueError('The val argument is an array, but val.shape != indices.shape.')

        metadata = {}

        # val: taken as is
        metadata['value'] = val

        # shape: if not given, infer from val (if array) or indices, else assume scalar
        if shape is not None:
            metadata['shape'] = shape
        elif not numpy.isscalar(val):
            metadata['shape'] = numpy.atleast_1d(val).shape
        elif indices is not None:
            metadata['shape'] = numpy.atleast_1d(indices).shape
        else:
            metadata['shape'] = (1,)

        # indices: None or ndarray
        if indices is None:
            metadata['indices'] = None
        else:
            metadata['indices'] = numpy.atleast_1d(indices)

        # units: taken as is
        metadata['units'] = units

        # desc: taken as is
        metadata['desc'] = desc

        # var_set: taken as is
        metadata['var_set'] = var_set

        self._var_allprocs_names['input'].append(name)
        self._var_myproc_names['input'].append(name)
        self._var_myproc_metadata['input'].append(metadata)
        self._var2meta[name] = metadata

    def add_output(self, name, val=1.0, shape=None, units=None, res_units=None, desc='',
                   lower=None, upper=None, ref=1.0, ref0=0.0,
                   res_ref=1.0, res_ref0=0.0, var_set=0):
        """Add an output variable to the component.

        Args
        ----
        name : str
            name of the variable in this component's namespace.
        val : float or list or tuple or ndarray
            The initial value of the variable being added in user-defined units. Default is 1.0.
        shape : int or tuple or list or None
            Shape of this variable, only required if indices not provided and val is not an array.
            Default is None.
        units : str or None
            Units in which the output variables will be provided to the component during execution.
            Default is None, which means it has no units.
        res_units : str or None
            Units in which the residuals of this output will be given to the user when requested.
            Default is None, which means it has no units.
        desc : str
            description of the variable.
        lower : float or list or tuple or ndarray or None
            lower bound(s) in user-defined units. It can be (1) a float, (2) an array_like
            consistent with the shape arg (if given), or (3) an array_like matching the shape of
            val, if val is array_like. A value of None means this output has no lower bound.
            Default is None.
        upper : float or list or tuple or ndarray or None
            upper bound(s) in user-defined units. It can be (1) a float, (2) an array_like
            consistent with the shape arg (if given), or (3) an array_like matching the shape of
            val, if val is array_like. A value of None means this output has no upper bound.
            Default is None.
        ref : float
            Scaling parameter. The value in the user-defined units of this output variable when
            the scaled value is 1. Default is 1.
        ref0 : float
            Scaling parameter. The value in the user-defined units of this output variable when
            the scaled value is 0. Default is 0.
        res_ref : float
            Scaling parameter. The value in the user-defined res_units of this output's residual
            when the scaled value is 1. Default is 1.
        res_ref0 : float
            Scaling parameter. The value in the user-defined res_units of this output's residual
            when the scaled value is 0. Default is 0.
        var_set : hashable object
            For advanced users only. ID or color for this variable, relevant for reconfigurability.
            Default is 0.
        """
        # First, type check all arguments
        if not isinstance(name, str):
            raise TypeError('The name argument should be a string')
        if not numpy.isscalar(val) and not isinstance(val, (list, tuple, numpy.ndarray)):
            raise TypeError('The val argument should be a float, list, tuple, or ndarray')
        if shape is not None and not isinstance(shape, (int, tuple, list)):
            raise TypeError('The shape argument should be an int, tuple, or list')
        if units is not None and not isinstance(units, str):
            raise TypeError('The units argument should be a str or None')
        if res_units is not None and not isinstance(res_units, str):
            raise TypeError('The res_units argument should be a str or None')
        if lower is not None and not numpy.isscalar(lower) and \
                not isinstance(lower, (list, tuple, numpy.ndarray)):
            raise TypeError('The lower argument should be a float, list, tuple, or ndarray')
        if upper is not None and not numpy.isscalar(upper) and \
                not isinstance(upper, (list, tuple, numpy.ndarray)):
            raise TypeError('The upper argument should be a float, list, tuple, or ndarray')
        for item in [ref, ref0, res_ref, res_ref]:
            if not numpy.isscalar(item):
                raise TypeError('The %s argument should be a float' % (item.__name__))

        if shape is not None:
            if isinstance(shape, int):
                shape = (shape,)
            elif isinstance(shape, list):
                shape = tuple(shape)
        # Next check that shapes are consistent between val and shape, if necessary
        if not numpy.isscalar(val):
            if shape is not None and numpy.atleast_1d(val).shape != shape:
                raise ValueError('The val argument is an array, but val.shape != shape.')

        metadata = {}

        # val: taken as is
        metadata['value'] = val

        # shape: if not given, infer from val (if array) or indices, else assume scalar
        if shape is not None:
            metadata['shape'] = shape
        elif not numpy.isscalar(val):
            metadata['shape'] = numpy.atleast_1d(val).shape
        else:
            metadata['shape'] = (1,)

        # units, res_units: taken as is
        metadata['units'] = units
        metadata['res_units'] = res_units

        # desc: taken as is
        metadata['desc'] = desc

        # lower, upper: check the shape if necessary
        if lower is not None and not numpy.isscalar(lower) and \
                numpy.atleast_1d(lower).shape != metadata['shape']:
            raise ValueError('The lower argument has the wrong shape')
        if upper is not None and not numpy.isscalar(upper) and \
                numpy.atleast_1d(upper).shape != metadata['shape']:
            raise ValueError('The upper argument has the wrong shape')
        metadata['lower'] = lower
        metadata['upper'] = upper

        # ref, ref0, res_ref, res_ref0: taken as is
        metadata['ref'] = ref
        metadata['ref0'] = ref0
        metadata['res_ref'] = res_ref
        metadata['res_ref0'] = res_ref0

        # var_set: taken as is
        metadata['var_set'] = var_set

        self._var_allprocs_names['output'].append(name)
        self._var_myproc_names['output'].append(name)
        self._var_myproc_metadata['output'].append(metadata)
        self._var2meta[name] = metadata

    def declare_partials(self, of, wrt, dependent=True,
                         rows=None, cols=None, val=None):
        """Store subjacobian metadata for later use.

        Args
        ----
        of : str or list of str
            The name of the residual(s) that derivatives are being computed for.
            May also contain a glob pattern.
        wrt : str or list of str
            The name of the variables that derivatives are taken with respect to.
            This can contain the name of any input or output variable.
            May also contain a glob pattern.
        dependent : bool(True)
            If False, specifies no dependence between the output(s) and the
            input(s). This is only necessary in the case of a sparse global
            jacobian, because if 'dependent=False' is not specified and
            set_subjac_info is not called for a given pair, then a dense
            matrix of zeros will be allocated in the sparse global jacobian
            for that pair.  In the case of a dense global jacobian it doesn't
            matter because the space for a dense subjac will always be
            allocated for every pair.
        rows : ndarray of int or None
            Row indices for each nonzero entry.  For sparse subjacobians only.
        cols : ndarray of int or None
            Column indices for each nonzero entry.  For sparse subjacobians only.
        val : float or ndarray of float
            Value of subjacobian.  If rows and cols are not None, this will
            contain the values found at each (row, col) location in the subjac.

        """
        oflist = [of] if isinstance(of, string_types) else of
        wrtlist = [wrt] if isinstance(wrt, string_types) else wrt

        # If only one of rows/cols is specified
        if (rows is None) ^ (cols is None):
            raise ValueError('If one of rows/cols is specified, then both must be specified')

        if val is not None and not issparse(val):
            val = numpy.atleast_1d(val)
            # numpy.promote_types  will choose the smallest dtype that can contain both arguments
            safe_dtype = numpy.promote_types(val.dtype, float)
            val = val.astype(safe_dtype, copy=False)

        if rows is not None:
            if isinstance(rows, (list, tuple)):
                rows = numpy.array(rows, dtype=int)
            if isinstance(cols, (list, tuple)):
                cols = numpy.array(cols, dtype=int)

            if rows.shape != cols.shape:
                raise ValueError('rows and cols must have the same shape,'
                                 ' rows: {}, cols: {}'.format(rows.shape, cols.shape))

            if val is not None and val.shape != (1,) and rows.shape != val.shape:
                raise ValueError('If rows and cols are specified, val must be a scalar or have the '
                                 'same shape, val: {}, rows/cols: {}'.format(val.shape, rows.shape))

        for of in oflist:
            for wrt in wrtlist:
                meta = {
                    'rows': rows,
                    'cols': cols,
                    'value': val,
                    'dependent': dependent,
                }
                # matching names/glob patterns will be resolved later because
                # we don't know if all variables have been declared at this
                # point.
                key = (of, wrt)
                if key in self._subjacs_info:
                    meta2 = self._subjacs_info[key]
                else:
                    meta2 = SUBJAC_META_DEFAULTS.copy()
                meta2.update(meta)
                self._subjacs_info[key] = meta2

    def _iter_partials_matches(self):
        """Generate all (of, wrt) name pairs to add to jacobian."""
        outs = self._var_allprocs_names['output']
        ins = self._var_allprocs_names['input']
        tvlists = (('output', outs), ('input', ins))

        for (of_pattern, wrt_pattern), meta in iteritems(self._subjacs_info):
            of_matches = fnmatch.filter(outs, of_pattern)
            for typ, vnames in tvlists:
                wrt_matches = fnmatch.filter(vnames, wrt_pattern)
                for (of, wrt) in product(of_matches, wrt_matches):
                    yield (of, wrt), meta, typ

    def _check_partials_meta(self, key, meta):
        """Check a given partial derivative and metadata for the correct shapes.

        Args
        ----
        key : tuple(str,str)
            The of/wrt pair defining the partial derivative.
        meta : dict
            Metadata dictionary from declare_partials.
        """
        of, wrt = key
        if meta['dependent']:
            out_size = numpy.prod(self._var2meta[of]['shape'])
            in_size = numpy.prod(self._var2meta[wrt]['shape'])
            rows = meta['rows']
            cols = meta['cols']
            if rows is not None:
                if rows.min() < 0:
                    msg = '{}: d({})/d({}): row indices must be non-negative'
                    raise ValueError(msg.format(self.pathname, of, wrt))
                if cols.min() < 0:
                    msg = '{}: d({})/d({}): col indices must be non-negative'
                    raise ValueError(msg.format(self.pathname, of, wrt))
                if rows.max() >= out_size or cols.max() >= in_size:
                    msg = '{}: d({})/d({}): Expected {}x{} but declared at least {}x{}'
                    raise ValueError(msg.format(
                        self.pathname, of, wrt,
                        out_size, in_size,
                        rows.max() + 1, cols.max() + 1))
            elif meta['value'] is not None:
                val = meta['value']
                val_shape = val.shape
                if len(val_shape) == 1:
                    val_out, val_in = val_shape[0], 1
                else:
                    val_out, val_in = val.shape
                if val_out > out_size or val_in > in_size:
                    msg = '{}: d({})/d({}): Expected {}x{} but val is {}x{}'
                    raise ValueError(msg.format(
                        self.pathname, of, wrt,
                        out_size, in_size,
                        val_out, val_in))

    def _set_partials_meta(self):
        """Set subjacobian info into our jacobian."""
        with self._jacobian_context() as J:
            for key, meta, typ in self._iter_partials_matches():
                self._check_partials_meta(key, meta)
                J._set_partials_meta(key, meta)

    def _setup_variables(self, recurse=False):
        """Assemble variable metadata and names lists.

        Sets the following attributes:
            _var_allprocs_names
            _var_myproc_names
            _var_myproc_metadata
            _var_pathdict
            _var_name2path

        Args
        ----
        recurse : boolean
            Ignored.
        """
        super(Component, self)._setup_variables(False)

        # set up absolute path info
        self._var_pathdict = {}
        self._var_name2path = {'input': {}, 'output': {}}
        for typ in ['input', 'output']:
            names = self._var_allprocs_names[typ]
            if self.pathname:
                self._var_allprocs_pathnames[typ] = paths = [
                    '.'.join((self.pathname, n)) for n in names
                ]
            else:
                self._var_allprocs_pathnames[typ] = paths = names
            for idx, name in enumerate(names):
                path = paths[idx]
                self._var_pathdict[path] = PathData(name, idx, idx, typ)
                if typ is 'input':
                    self._var_name2path[typ][name] = (path,)
                else:
                    self._var_name2path[typ][name] = path

    def _setup_vector(self, vectors, vector_var_ids, use_ref_vector):
        r"""Add this vector and assign sub_vectors to subsystems.

        Sets the following attributes:

        - _vectors
        - _vector_transfers
        - _inputs*
        - _outputs*
        - _residuals*
        - _transfers*

        \* If vec_name is 'nonlinear'

        Args
        ----
        vectors : {'input': <Vector>, 'output': <Vector>, 'residual': <Vector>}
            <Vector> objects corresponding to 'name'.
        vector_var_ids : ndarray[:]
            integer array of all relevant variables for this vector.
        use_ref_vector : bool
            if True, allocate vectors to store ref. values.
        """
        super(Component, self)._setup_vector(vectors, vector_var_ids,
                                             use_ref_vector)

        # Components must load their initial input and output values into the
        # vectors.

        # Note: It's possible for meta['value'] to not match
        #       meta['shape'], and input and output vectors are sized according
        #       to shape, so if, for example, value is not specified it
        #       defaults to 1.0 and the shape can be anything, resulting in the
        #       value of 1.0 being broadcast into all values in the vector
        #       that were allocated according to the shape.
        if vectors['input']._name is 'nonlinear':
            names = self._var_myproc_names['input']
            inputs = self._inputs
            for i, meta in enumerate(self._var_myproc_metadata['input']):
                inputs[names[i]] = meta['value']

        if vectors['output']._name is 'nonlinear':
            names = self._var_myproc_names['output']
            outputs = self._outputs
            for i, meta in enumerate(self._var_myproc_metadata['output']):
                outputs[names[i]] = meta['value']
