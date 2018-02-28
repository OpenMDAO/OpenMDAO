*****************************************
Using src_indices with Promoted Variables
*****************************************

Inputs and outputs can be connected by promoting them both to the same name, but what
if your output is an array and you only want to connect part of it to your
input?  If you connect variables via promotion, you must set :code:`src_indices` when
you call :code:`add_input` to add your input variable to its component.  Another
argument, :code:`flat_src_indices` is a boolean that determines whether the entries
of the :code:`src_indices` array are interpreted as indices into the flattened source
or as tuples or lists of indices into the unflattened source.  The default
of :code:`flat_src_indices=False` assumes indices map into an unflattened source.

Usage
-----

1. Connect an independent array variable to two different components where
each component gets part of the array.

.. embed-code::
    openmdao.core.tests.test_group.TestGroup.test_promote_src_indices
    :layout: interleave

2. A distributed component that promotes its input and receives certain
entries of the source array based on its rank.  Note that negative indices are
supported.

.. embed-code::
    openmdao.core.tests.test_distribcomp.TestGroupMPI.test_promote_distrib
    :layout: interleave

3. The source array is shape (4,3) and the input array is shape (2,2)

.. embed-code::
    openmdao.core.tests.test_group.TestGroup.test_promote_src_indices_nonflat
    :layout: interleave

4. If the source array is shape (4,3), the input is scalar, and we want to
connect it to the (3, 1) entry of the source, then the :code:`add_input`
call might look like the following if we use flat :code:`src_indices`:

.. code::

    self.add_input('x', 1.0, src_indices=[10], shape=1, flat_src_indices=True)


If we instead use the default setting of :code:`flat_src_indices=False`, we must specify
the input shape since a scalar input value alone doesn't necessarily indicate that
the input variable is scalar.  For example, in the case below, if we didn't know the
input shape, we wouldn't know if it was scalar and connected to a 2-D source
array, or if it was shape (1,2) and connected to a flat source array.

.. code::

    self.add_input('x', 1.0, src_indices=np.array([[3,1]]), shape=1)

5.  If the source array is flat and the input is shape (2,2), the :code:`add_input`
call might look like this:

.. code::

    self.add_input('x', src_indices=[[0, 10], [7, 4]], shape=(2,2))


.. note::

    If the source array is flat, we allow the use of flat src_indices even
    without setting :code:`flat_src_indices=True`.
