[![TravisCI Badge][9]][10]
[![AppVeyor Badge][11]][12]
[![Coveralls Badge][13]][14]

# [OpenMDAO][0]

## Documentation

Documentation for the latest version can be found [here][2].

Documentation archives for prior versions can be found [here][3].

## Important Notice

While the API is relatively stable, **OpenMDAO** remains in active development.
There will be periodic changes to the API.
User's are encouraged to pin their version of OpenMDAO to a recent release and
update periodically.

### OpenMDAO Versions

**OpenMDAO 3.x.y** represents the current version and is no longer
considered **[BETA][15]**.  It requires Python 3.6 or later and is
maintained [here][4]. 
To install the latest release, run `pip install --update openmdao`.

**OpenMDAO 2.10.x** is the last version to support Python2.x and will
only receive critical bug fixes going forward.
To install this older release, run `pip install "openmdao<3"`
(the quotes around `openmdao<3` are required).

> **PLEASE NOTE**: This repository was previously named **OpenMDAO/blue**. 
If you had cloned that repository, please update your repository name and
remotes to reflect these changes. You can find instructions [here][8].

The **OpenMDAO 1.7.4** code repository is now named **OpenMDAO1**, and has moved
[here][5]. To install it, run: `pip install "openmdao<2"`
(the quotes around `openmdao<2` are required).

The legacy **OpenMDAO v0.x** (versions 0.13.0 and older) of the 
**OpenMDAO-Framework** are [here][6].

## Install OpenMDAO

You have two options for installing **OpenMDAO**, (1) from the
[Python Package Index (PyPI)][1], and (2) from the [GitHub repository][4].

**OpenMDAO** includes several optional sets of dependencies: 
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
    pip install OpenMDAO

#### Install for Development
If you would like to make changes to **OpenMDAO** it is recommended you
install it in *[editable][16]* mode (i.e., development mode) by adding the `-e`
flag when calling `pip`, this way any changes you make to the source code will
be included when you import **OpenMDAO** in *Python*. You will also want to
install the packages necessary for running **OpenMDAO**'s tests and documentation
generator.  You can install everything needed for development by running:

    pip install -e OpenMDAO[all]

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

> You will need **make** to build the documentation.  If you are using Windows,
you can install [Anaconda](https://www.anaconda.com/download/) and install
**make** by running: `conda install make`.

1. Make sure you have cloned the repository with the source code:
    > Follow the [instructions](#install-from-a-cloned-repository) for
    installing **OpenMDAO** from a cloned repository.

2. Install **OpenMDAO** and the dependencies required to build the
   documentation:

    `pip install OpenMDAO[docs]`

3. Change to the docs directory:

    `cd OpenMDAO/openmdao/docs`

4. Run the command to auto-generate the documentation.

    `make clean; make all`

This will build the docs into `openmdao/docs/_build/html`.  You can browse the
documentation by opening `openmdao/docs/_build/html/index.html` with your web
browser.


[0]: http://openmdao.org/ "OpenMDAO"
[1]: https://pypi.org/project/openmdao/ "OpenMDAO @PyPI"

[2]: http://openmdao.org/twodocs/versions/latest "Latest Docs"
[3]: http://openmdao.org/twodocs "Archived Docs"

[4]: https://github.com/OpenMDAO/OpenMDAO "OpenMDAO Git Repo"
[5]: https://github.com/OpenMDAO/OpenMDAO1 "OpenMDAO 1.x Git Repo"
[6]: https://github.com/OpenMDAO/OpenMDAO-Framework "OpenMDAO Framework Git Repo"

[7]: https://github.com/OpenMDAO/OpenMDAO/issues/new "Make New OpenMDAO Issue"

[8]: https://help.github.com/articles/changing-a-remote-s-url/ "Update Git Remote URL"

[9]: https://travis-ci.org/OpenMDAO/OpenMDAO.svg?branch=master "TravisCI Badge"
[10]: https://travis-ci.org/OpenMDAO/OpenMDAO "OpenMDAO @TravisCI"
[11]: https://ci.appveyor.com/api/projects/status/33kct0irhbgcg8m1?svg=true "Build Badge"
[12]: https://ci.appveyor.com/project/OpenMDAO/blue/branch/master "OpenMDAO @AppVeyor"
[13]: https://coveralls.io/repos/github/OpenMDAO/OpenMDAO/badge.svg?branch=master "Coverage Badge"
[14]: https://coveralls.io/github/OpenMDAO/OpenMDAO?branch=master "OpenMDAO @Coveralls"

[15]: https://en.wikipedia.org/wiki/Software_release_life_cycle#Beta "Wikipedia Beta"

[16]: https://setuptools.readthedocs.io/en/latest/setuptools.html#development-mode "Pip Editable Mode"
