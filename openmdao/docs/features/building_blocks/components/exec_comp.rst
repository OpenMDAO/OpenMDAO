.. _feature_exec_comp:
.. index:: ExecComp Example

********
ExecComp
********

`ExecComp` is a component that provides a shortcut for building an ExplicitComponent that
represents a set of simple mathematical relationships between inputs and outputs. The ExecComp
automatically takes care of all of the component API methods, so you just need to instantiate
it with an equation. Derivatives are also automatically determined using the complex step
method.  Because of this, functions available for use in ExecComp are limited to the following
numpy and scipy functions:

=========================  ====================================
Function                   Description
=========================  ====================================
abs(x)                     Absolute value of x
acos(x)                    Inverse cosine of x
acosh(x)                   Inverse hyperbolic cosine of x
arange(start, stop, step)  Array creation
arccos(x)                  Inverse cosine of x
arccosh(x)                 Inverse hyperbolic cosine of x
arcsin(x)                  Inverse sine of x
arcsinh(x)                 Inverse hyperbolic sine of x
arctan(x)                  Inverse tangent of x
asin(x)                    Inverse sine of x
asinh(x)                   Inverse hyperbolic sine of x
atan(x)                    Inverse tangent of x
cos(x)                     Cosine of x
cosh(x)                    Hyperbolic cosine of x
dot(x, y)                  Dot-product of x and y
e                          Euler's number
erf(x)                     Error function
erfc(x)                    Complementary error function
exp(x)                     Exponential function
expm1(x)                   exp(x) - 1
factorial(x)               Factorial of all numbers in x
fmax(x, y)                 Element-wise maximum of x and y
fmin(x, y)                 Element-wise minimum of x and y
inner(x, y)                Inner product of arrays x and y
isinf(x)                   Element-wise detection of np.inf
isnan(x)                   Element-wise detection of np.nan
kron(x, y)                 Kronecker product of arrays x and y
linspace(x, y, N)          Numpy linear spaced array creation
log(x)                     Natural logarithm of x
log10(x)                   Base-10 logarithm of x
log1p(x)                   log(1+x)
matmul(x, y)               Matrix multiplication of x and y
maximum(x, y)              Element-wise maximum of x and y
minimum(x, y)              Element-wise minimum of x and y
ones(N)                    Create an array of ones
outer(x, y)                Outer product of x and y
pi                         Pi
power(x, y)                Element-wise x**y
prod(x)                    The product of all elements in x
sin(x)                     Sine of x
sinh(x)                    Hyperbolic sine of x
sum(x)                     The sum of all elements in x
tan(x)                     Tangent of x
tanh(x)                    Hyperbolic tangent of x
tensordot(x, y)            Tensor dot product of x and y
zeros(N)                   Create an array of zeros
=========================  ====================================

For example, here is a simple component that takes the input and adds one to it.

.. embed-test::
    openmdao.components.tests.test_exec_comp.TestExecComp.test_feature_simple

You can also declare an ExecComp with arrays for inputs and outputs, but when you do, you must also
pass in a correctly-sized array as an argument to the ExecComp call. This can be the initial value
in the case of unconnected inputs, or just an empty array with the correct size.

.. embed-test::
    openmdao.components.tests.test_exec_comp.TestExecComp.test_feature_array

Functions from the math library are available for use in the expression strings.

.. embed-test::
    openmdao.components.tests.test_exec_comp.TestExecComp.test_feature_math

You can also access built-in Numpy functions by using the prefix "numpy." with the function name.

.. embed-test::
    openmdao.components.tests.test_exec_comp.TestExecComp.test_feature_numpy

You can also declare metadata like 'units', 'upper', or 'lower' on the inputs and outputs. Here is an example
where we declare all our inputs to be inches to trigger conversion from a variable expressed in feet in one
connection source.

.. embed-test::
    openmdao.components.tests.test_exec_comp.TestExecComp.test_feature_metadata

.. tags:: ExecComp, Examples