Setting Up Travis CI for Your OpenMDAO Plugin
=============================================

Continuous integration using Travis CI is a way to make sure all your building and testing protocols remain
functioning over time. For general assistance on how to get your initial Travis account set up, and how to make a .travis.yml
file, see `this guide <https://docs.travis-ci.com/user/getting-started/>`_.

When working with an OpenMDAO-dependent project, there are several specific settings one must be mindful of,
as well as technical nuances that must be included in the .travis.yml file

.travis.yml
-----------

When you run your doc-build on CI, and you happen to be embedding any code from the openmdao project, that code will be
executed to generate output for the documentation. That code will not execute up on the CI platform unless you have first installed
everything that OpenMDAO needs to run. So installing just what your project needs will not be enough.


Coverage
--------

Coverage is a tool that shows developers how many lines of their code that are being executed by your current test suite.

To use this tool, set up an account at the website coveralls.io, and then activate the appropriate github repo for coverage results.
This sets up the site to receive results. The other side is taken care of in your .travis.yml file, as I will show in OpenMDAO's example:

.. code-block:: python

    script:
        testflo -n 1 openmdao --coverage  --coverpkg openmdao --cover-omit \*tests/\*  --cover-omit \*devtools/\* --cover-omit \*test_suite/\* --cover-omit \*docs/\*;

    after_success:
        coveralls --rcfile=../../.coveragerc --output=coveralls.json;
        sed 's/\/home\/travis\/miniconda\/lib\/python'"$PY"'\/site-packages\///g' < coveralls.json > coveralls-upd.json;
        coveralls --upload=coveralls-upd.json;

The point is to use testflo to run things, and set the `coverpkg` to your project, and the `cover-omit` dirs to exclude from coverage, use coveralls
to collect the data, and then send the results to coveralls.io.

Caching
-------

The concept of build caching on Travis CI is intended to speed up the build, and therefore the entire build/test cycle on Travis CI.
By caching the builds of dependencies/requirements that rarely change, we can get right to of various dependencies to speed up the build and the docbuild.
This topic probably requires a document of its own, coming soon.
