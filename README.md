[![GitHub Actions Test Badge][17]][18]
[![Coveralls Badge][13]][14]
[![PyPI version][10]][11]
[![PyPI Monthly Downloads][12]][11]

# [OpenMDAO][0]

OpenMDAO is an open-source high-performance computing platform for
systems analysis and multidisciplinary optimization, written in Python.
It enables you to decompose your models, making them easier to build and
maintain, while still solving them in a tightly coupled manner with
efficient parallel numerical methods.

The OpenMDAO project is primarily focused on supporting gradient-based
optimization with analytic derivatives to allow you to explore large
design spaces with hundreds or thousands of design variables, but the
framework also has a number of parallel computing features that can
work with gradient-free optimization, mixed-integer nonlinear
programming, and traditional design space exploration.

If you are using OpenMDAO, please [cite][20] us!

## Documentation

Documentation for the latest development version can be found [here][2].

Documentation for all released versions can be found [here][3].

## Important Notice

While the API is relatively stable, **OpenMDAO** remains in active development.
There will be periodic changes to the API.
User's are encouraged to pin their version of OpenMDAO to a recent release and
update periodically.

## Install OpenMDAO

You have two options for installing **OpenMDAO**, (1) from the
[Python Package Index (PyPI)][1], and (2) from the [GitHub repository][4].

**OpenMDAO** includes several optional sets of dependencies including:
`test` for installing the developer tools (e.g., testing, coverage),
`docs` for building the documentation and
`visualization` for some extra visualization tools.
Specifying `all` will include all of the optional dependencies.

### Install from [PyPI][1]

This is the easiest way to install **OpenMDAO**. To install only the runtime
dependencies:

    pip install openmdao

To install all the optional dependencies:

    pip install openmdao[all]

### Install from a Cloned Repository

This allows you to install **OpenMDAO** from a local copy of the source code.

    git clone http://github.com/OpenMDAO/OpenMDAO
    cd OpenMDAO
    pip install .

If you would like to make changes to **OpenMDAO** it is recommended you
install it in *[editable][16]* mode (i.e., development mode) by adding the `-e`
flag when calling `pip`, this way any changes you make to the source code will
be included when you import **OpenMDAO** in *Python*. You will also want to
install the packages necessary for running **OpenMDAO**'s tests and documentation
generator.  You can install everything needed for development by running:

    pip install -e OpenMDAO[all]

## OpenMDAO Versions

**OpenMDAO 3.x.y** represents the current, supported version. It requires Python 3.8
or later and is maintained [here][4]. To upgrade to the latest release, run:

    pip install --upgrade openmdao

**OpenMDAO 2.10.x** was the last version to support Python 2.x and is no longer supported.
To install this older release, run:

    pip install "openmdao<3"

**OpenMDAO 1.7.4** was an earlier version of OpenMDAO and is also no longer supported.
The code repository is now named **OpenMDAO1**, and has moved [here][5]. To install it, run:

    pip install "openmdao<2"

The legacy **OpenMDAO v0.x** (versions 0.13.0 and older) of the
**OpenMDAO-Framework** are [here][6].

## Test OpenMDAO

Users are encouraged to run the unit tests to ensure **OpenMDAO** is performing
correctly.  In order to do so, you must install the testing dependencies.

1. Install **OpenMDAO** and its testing dependencies:

    `pip install openmdao[test]`

    > Alternatively, you can clone the repository, as explained
    [here](#install-from-a-cloned-repository), and install the development
    dependencies as described [here](#install-the-developer-dependencies).

2. Run tests:

    `testflo openmdao -n 1`

3. If everything works correctly, you should see a message stating that there
were zero failures.  If the tests produce failures, you are encouraged to report
them as an [issue][7].  If so, please make sure you include your system spec,
and include the error message.

    > If tests fail, please include your system information, you can obtain
    that by running the following commands in *python* and copying the results
    produced by the last line.

        import platform, sys

        info = platform.uname()
        (info.system, info.version), (info.machine, info.processor), sys.version

    > Which should produce a result similar to:

        (('Windows', '10.0.17134'),
         ('AMD64', 'Intel64 Family 6 Model 94 Stepping 3, GenuineIntel'),
         '3.6.6 | packaged by conda-forge | (default, Jul 26 2018, 11:48:23) ...')

## Build the Documentation for OpenMDAO

Documentation for the latest version can always be found [here][2], but if you would like to build a local copy you can find instructions to do so [here][19].

[0]: http://openmdao.org/ "OpenMDAO"
[1]: https://pypi.org/project/openmdao/ "OpenMDAO @PyPI"

[2]: http://openmdao.org/newdocs/versions/latest "Latest Docs"
[3]: http://openmdao.org/docs "Archived Docs"

[4]: https://github.com/OpenMDAO/OpenMDAO "OpenMDAO Git Repo"
[5]: https://github.com/OpenMDAO/OpenMDAO1 "OpenMDAO 1.x Git Repo"
[6]: https://github.com/OpenMDAO/OpenMDAO-Framework "OpenMDAO Framework Git Repo"

[7]: https://github.com/OpenMDAO/OpenMDAO/issues/new "Make New OpenMDAO Issue"

[8]: https://help.github.com/articles/changing-a-remote-s-url/ "Update Git Remote URL"

[10]: https://badge.fury.io/py/openmdao.svg "PyPI Version"
[11]: https://badge.fury.io/py/openmdao "OpenMDAO @PyPI"

[12]: https://img.shields.io/pypi/dm/openmdao "PyPI Monthly Downloads"

[13]: https://coveralls.io/repos/github/OpenMDAO/OpenMDAO/badge.svg?branch=master "Coverage Badge"
[14]: https://coveralls.io/github/OpenMDAO/OpenMDAO?branch=master "OpenMDAO @Coveralls"

[15]: https://en.wikipedia.org/wiki/Software_release_life_cycle#Beta "Wikipedia Beta"

[16]: https://setuptools.readthedocs.io/en/latest/setuptools.html#development-mode "Pip Editable Mode"

[17]: https://github.com/OpenMDAO/OpenMDAO/actions/workflows/openmdao_test_workflow.yml/badge.svg "Github Actions Badge"
[18]: https://github.com/OpenMDAO/OpenMDAO/actions "Github Actions"

[19]: http://openmdao.org/newdocs/versions/latest/other_useful_docs/developer_docs/doc_build.html

[20]: https://openmdao.org/newdocs/versions/latest/other/citing.html
