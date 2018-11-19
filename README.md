[![TravisCI Badge][9]][10]
[![AppVeyor Badge][11]][12]
[![Coveralls Badge][13]][14]

# [OpenMDAO 2][0]
*This version of **OpenMDAO** is in Development Status **[BETA][15]**.*

## Documentation
Documentation for the latest version can be found [here][2].

Documentation archives for prior versions can be found [here][3].

## Important Notice
While the API is MOSTLY stable, the **OpenMDAO** development team reserves the
right to update it and other elements of the code as needed.

The team will be making frequent updates to the source code, so users are
encouraged to regularly pull for updates.

### OpenMDAO Versions
> **PLEASE NOTE**: Until recently, this repository was named **OpenMDAO/blue**. 
If you had cloned that repository, please update your repository name and
remotes to reflect these changes. You can find instructions [here][8].

The **OpenMDAO 2.x.y** code has taken the name **OpenMDAO**, and is maintained
[here][4]. To install the latest release, run `pip install --update openmdao`.

The **OpenMDAO 1.7.4** code repository is now named **OpenMDAO1**, and has moved
[here][5]. To install it, run: `pip install "openmdao<2"` (the quotes around 
`openmdao<2` are required). 

The legacy **OpenMDAO v0.x** (versions 0.13.0 and older) of the 
**OpenMDAO-Framework** are [here][6].

### Missing Features from OpenMDAO 1.7.4
Be aware that this new version of **OpenMDAO** is in Development Status **BETA**
and not all the features of 1.7.4 are currently available.

Here is a list of things that have not yet been developed in 2.x:

* Pass-by-object variables
* Automatic ordering of groups/components based on data connections
* Parallel Finite Difference
* File variables
* Active-set constraint calculation disabling
* Brent Solver
* CaseRecording using CSV, HDF5, and dump recorders (SqliteRecorder and 
WebRecorder are currently supported)

## Install OpenMDAO 2
You have two options for installing **OpenMDAO**, (1) from the
[Python Package Index (PyPI)][1], and (2) from the [GitHub repository][4].

**OpenMDAO** includes two optional sets of dependencies, one for installing
the developer tools (e.g., testing, coverage), and one for building the
documentation.

### Install from [PyPI][1]
This is the easiest way to install **OpenMDAO**.
> `pip install openmdao`

To install the testing dependencies, run:
> `pip install openmdao[develop]`

### Install from a Cloned Repository
This allows you to install **OpenMDAO** from a local copy of the source code.
> `git clone http://github.com/OpenMDAO/OpenMDAO`
>
> `pip install OpenMDAO`

#### Install the Developer Dependencies
This includes the packages necessary for running **OpenMDAO**'s tests.  
> `pip install OpenMDAO[develop]`

If you would like to make changes to **OpenMDAO** it is recommended you
install it in *[editable][16]* mode (i.e., development mode) so any changes you
make to the source code will be reflected when you import **OpenMDAO** in
*Python*.
> `pip install -e OpenMDAO[develop]`

## Test OpenMDAO 2
Users are encourage to run the unit tests to ensure **OpenMDAO** is performing
correctly.  In order to do so, you must install the testing dependencies.

1. Install **OpenMDAO** and its testing dependencies:

    > `pip install openmdao[develop]`

    > Alternatively, you can clone the repository, as explained
    [here](#install-from-a-cloned-repository), and install the development
    dependencies as described [here](#install-the-developer-dependencies).

2. Run tests:

    > `testflo openmdao -n 1`

3. If everything works correctly, you should see a message stating that there 
were zero failures.  If you see failures, you are encouraged to report
it as an [issue][7].  If so, please make sure you include your system spec,
and include the error message.

    > If tests fail, please include your system information, you can obtain
    that by running the following commands in *python* and copying the results
    produced by the last line.
    ```python
    import platform, sys
    info = platform.uname()
    (info.system, info.version), (info.machine, info.processor), sys.version
    ```
    > Which should produce a result similar to:
    ```python
    (('Windows', '10.0.17134'),
     ('AMD64', 'Intel64 Family 6 Model 94 Stepping 3, GenuineIntel'),
     '3.6.6 | packaged by conda-forge | (default, Jul 26 2018, 11:48:23) ...')
    ```

## Build the Documentation for OpenMDAO 2
> You will need **make** to build the documentation.  If you are using Windows,
you can install [Anaconda](https://www.anaconda.com/download/) and install
**make** by running: `conda install make`.

1. Make sure you have cloned the repository with the source code:
    > Follow the [instructions](#install-from-a-cloned-repository) for
    installing **OpenMDAO** from a cloned repository.

2. Install **OpenMDAO** and the dependencies required to build the documentation :
    > `pip install OpenMDAO[docs]`

3. Change to the docs directory:
    > `cd OpenMDAO/openmdao/docs`

4. Run the command to auto-generate the documentation.
    > `make clean; make all`

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
