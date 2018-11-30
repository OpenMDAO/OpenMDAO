.. _feature_check_partials_subset:

***************************************************
Checking Partial Derivatives on a Subset of a Model
***************************************************

Includes and Excludes
---------------------

When you have a model with a large number of components, you may want to reduce the number of components you
check so that the output is small and readable. The `check_partials` method has two arguments: "includes" and
"excludes" that help you specify a reduced set. Both of these arugments are lists of strings that default to None. If you
specify "includes", and give it a list containing strings, then only the components whose full pathnames match one of the patterns in those strings
are included in the check. Wildcards are acceptable in the string patterns. Likewise, if you specify excludes, then components whose pathname matches
the given patterns will be excluded from the check.

You can use both arguments together to hone in on the precise set of components you wish to check.

.. embed-code::
    openmdao.core.tests.test_check_derivs.TestCheckPartialsFeature.test_includes_excludes
    :layout: interleave
