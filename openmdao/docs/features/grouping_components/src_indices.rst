
Using src_indices with promoted variables
-----------------------------------------

Inputs and outputs can be connected by promoting them to the same name, but what
if your output is an array and you only want to connect part of it to your
input?  If you connect variables via promotion, your must
set *src_indices* when you add your input variable to its component.

Usage
+++++

1. Connect an independent array variable to two different components where
each component gets part of the array.

.. embed-test:: openmdao.core.tests.test_group.TestGroup.test_promote_src_indices

2. A distributed component that promotes its input and receives certain
entries of the source array based on its rank.

.. embed-code:: openmdao.core.tests.test_group.TestGroupMPI.test_promote_distrib

3. The source array is shape (4,3) and the input array is shape (2,2)

.. embed-test:: openmdao.core.tests.test_group.TestGroup.test_promote_src_indices_nonflat

4. If the source array is shape (4,3) and the input is scalar, the *add_input*
call might look like the following.  We must specify the input shape here
since a scalar input value alone doesn't necessarily indicate that the input
variable is scalar.  For example, in the case below if we didn't know the
input shape, we wouldn't know if it was scalar and connected to a 2-D source
array, or if it was shape (1,2) and connected to a flat source array.

.. code::

    self.add_input('x', 1.0, src_indices=numpy.array([[3,1]]), shape=1)

5.  If the source array is flat and the input is shape (2,2), the *add_input*
call might look like this:

.. code::

    self.add_input('x', src_indices=[[0, 10], [7, 4]], shape=(2,2))
