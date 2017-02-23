
Using src_indices with promoted variables
-----------------------------------------

Inputs and outputs can be connected by promoting them to the same name, but what
if your output is an array and you only want to connect part of it to your
input?  If you connect variables via promotion, your only choice is to
set *src_indices* when you add your input variable to its component.


Usage
+++++

1. Connect an independent array variable to two different components where
each component gets part of the array.

.. embed-test::
    openmdao.core.tests.test_group.TestGroup.test_promote_src_indices

2. A distributed component that promotes its input and receives certain
entries of the source array based on its rank.

.. embed-code::
    openmdao.core.tests.test_group.TestGroupMPI.test_promote_distrib

3. When the source array is not flat, src_indices must be an array with a
shape that matches the shape of the input variable with the addition of an
extra dimension of size equal to the number of dimensions of the source array.

.. embed-test::
    openmdao.core.tests.test_group.TestGroup.test_promote_src_indices_nonflat
