[![Build Status][9]][10]
[![Build status][11]][12]
[![Coverage Status][13]][14]

# OpenMDAO Versions
> **PLEASE NOTE**: Until recently, this repository was named **OpenMDAO/blue**. 
If you had cloned that repository, please update your repository name and 
remotes to reflect these changes. You can find instructions [here][7].

The **OpenMDAO 2.x.y** code has taken the name **OpenMDAO**, and is maintained 
[here][4]. To install the latest release, run `pip install --update openmdao`.

The **OpenMDAO 1.7.4** code repository is now named **OpenMDAO1**, and has moved
[here][5]. To install it, run: `pip install "openmdao<2"` (the quotes around 
`openmdao<2` are required). 

The legacy **OpenMDAO v0.x** (versions 0.13.0 and older) of the 
**OpenMDAO-Framework** are [here][8].


# OpenMDAO 2
This is an ALPHA version of **OpenMDAO 2**.

The latest docs can be found [here][2].

Archived versions of the docs can be found [here][3].

## Important Note
While the API is MOSTLY stable, we reserve the right to change things as needed.

We will be making frequent updates to this code. If you’re going to try it,
make sure you pull these updates often.


## Features of OpenMDAO 1.7.4 Not Yet in 2.x.y

Be aware that this is an Alpha.
Not all the features of 1.7.4 exist in 2.x.y yet.

Here is a list of things that have not yet been developed in 2.x:

* Pass-by-object variables
* Automatic ordering of groups/components based on data connections
* Parallel Finite Difference
* File variables
* Active-set constraint calculation disabling
* Brent Solver
* CaseRecording using CSV, HDF5, and dump recorders (SqliteRecorder and 
WebRecorder are currently supported)

# Installation Instructions
You have two means for installing OpenMDAO, from the Python Package Index (PyPI)
or from a clone of the [GitHub repository](https://github.com/OpenMDAO/OpenMDAO).
### Install from [PyPI][1]
This is the easist way to install **OpenMDAO**.
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
install it in *editable* mode (i.e., development mode).
> `pip install -e OpenMDAO[develop]`

If you are planning to change the Developer mode will allow you to make changes
to the OpenMDAO code

#### Install the documentation making dependencies
> `pip install OpenMDAO[docs]`

# Documentation Building Instructions
> Make sure you followed the [instructions](#install-from-a-cloned-repository)
for installing **OpenMDAO** from a cloned repository, and the 
[instructions](#install-the-documentation-making-dependencies) for installing 
the documentation making dependencies

> You will need **make** to build the documentation.  If you are using Windows,
you can install [Anaconda](https://www.anaconda.com/download/) and install 
**make** by running: `conda install make`.

Change to the docs directory:

> `cd OpenMDAO/openmdao/docs`
>
> `make clean; make all`

This will build the docs into `openmdao/docs/_build/html`.

Then, just open  `openmdao/docs/_build/html/index.html` in a browser see the 
docs.


# Testing Instructions
1. Install OpenMDAO and its testing dependencies:

    >`pip install openmdao[develop]`

2. Run tests:

    > `testflo openmdao -n 1`

3. If everything works correctly, you should see a message that there 
were no failures.  If you see failures, you are encouraged to report
it as an [issue][6].  If so, please make sure you include your system spec,
and include the error message.

    > If tests fail, please include your system information, you can obtain that
    by running the following commands in *python* and copying the results 
    produced by the last line.
    ```python
    >>> import platform
    >>> import sys
    >>>
    >>> info = platform.uname()
    >>> (info.system, info.version), (info.machine, info.processor), sys.version
    ```
    > Which should produce a result similar to:
    ```python
    (('Windows', '10.0.17134'),
     ('AMD64', 'Intel64 Family 6 Model 94 Stepping 3, GenuineIntel'),
      '3.6.6 | packaged by conda-forge | (default, Jul 26 2018, 11:48:23) [MSC v.1900 64 bit (AMD64)]')
    ```


[1]: https://pypi.org/project/openmdao/ "PyPI"
[2]: http://openmdao.org/twodocs/versions/latest "Latest Docs"
[3]: http://openmdao.org/twodocs "Archived Docts"
[4]: https://github.com/OpenMDAO/OpenMDAO "OpenMDAO Git Repo"
[5]: https://github.com/OpenMDAO/OpenMDAO1 "OpenMDAO 1.x Git Repo"
[6]: https://github.com/OpenMDAO/OpenMDAO/issues/new "Make New Issue"
[7]: https://help.github.com/articles/changing-a-remote-s-url/ "Remote URL Update"
[8]: https://github.com/OpenMDAO/OpenMDAO-Framework "OpenMDAO Framework Git Repo"
[9]: https://travis-ci.org/OpenMDAO/OpenMDAO.svg?branch=master "TravisCI Badge"
[10]: https://travis-ci.org/OpenMDAO/OpenMDAO "OpenMDAO @TravisCI"
[11]: https://ci.appveyor.com/api/projects/status/33kct0irhbgcg8m1?svg=true "Build Badge"
[12]: https://ci.appveyor.com/project/OpenMDAO/blue/branch/master "OpenMDAO @AppVeyor"
[13]: https://coveralls.io/repos/github/OpenMDAO/OpenMDAO/badge.svg?branch=master "Coverage Badge"
[14]: https://coveralls.io/github/OpenMDAO/OpenMDAO?branch=master "OpenMDAO @Coveralls"
