.. _feature_unit_testing_partials:

********************************
Unit Testing Partial Derivatives
********************************

If you want to check the implementations of a Component's partial derivatives as part of a unit test,
you can make use of a custom assert function, :code:`assert_check_partials`.

.. autofunction:: openmdao.utils.assert_utils.assert_check_partials
    :noindex:


In your unit test, after calling :code:`check_partials` on a :code:`Component`, you can call the
:code:`assert_check_partials` function with the returned value from :code:`check_partials`.


Usage
-----

.. embed-test::
    openmdao.utils.tests.test_assert_utils.TestAssertUtils.test_assert_check_partials_no_exception_expected
