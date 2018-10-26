.. _`travis_ci_setup`:


Setting Up Travis CI for Your Project
=====================================

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

You can use the .travis.yml file in our template repository to get you started for your project.
That file looks something like this:

.. code-block:: none

    sudo: false

    os:
      - linux

    env:
      - PY=2.7
      - PY=3.6 UPLOAD_DOCS=1

    language: generic

    addons:
      apt:
        sources:
        - ubuntu-toolchain-r-test
        packages:
        - gfortran
        - libblas-dev
        - liblapack-dev
        - libopenmpi-dev
        - openmpi-bin

    before_install:
    - if [ "$PY" = "2.7" ];  then wget "https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh" -O miniconda.sh; fi
    - if [ "$PY" = "3.6" ];  then wget "https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh" -O miniconda.sh; fi
    - chmod +x miniconda.sh;
    - ./miniconda.sh -b  -p /home/travis/miniconda;
    - export PATH=/home/travis/miniconda/bin:$PATH;
    - if  [ "$TRAVIS_REPO_SLUG" = "OpenMDAO/dymos" ] && [ "$TRAVIS_PULL_REQUEST" = "false" ]; then
        MASTER_BUILD=1;
      fi

    install:
    - conda install --yes python=$PY numpy scipy nose sphinx mock swig pip;
    - pip install --upgrade pip
    - sudo apt-get install gfortran
    - pip install numpy==1.14.1
    - pip install scipy==1.0.0
    - pip install mpi4py
    - pip install matplotlib
    - pip install nose
    - pip install networkx
    - pip install testflo
    - pip install pyyaml
    - pip install coveralls
    - pip install --user travis-sphinx;

    # install pyoptsparse
    - git clone https://github.com/OpenMDAO/pyoptsparse.git;
    - cd pyoptsparse;
    - python setup.py install;
    - cd ..;

    # install MBI
    - git clone https://github.com/OpenMDAO/MBI.git;
    - cd MBI;
    - python setup.py build install;
    - cd ..;

    # install OpenMDAO in developer mode so we have access to its sphinx extensions
    - git clone https://github.com/OpenMDAO/OpenMDAO.git;
    - cd OpenMDAO;
    - pip install -e .;
    - cd ..;

    # install your project itself in developer mode.
    - pip install -e .;

    script:
    - testflo -n 1 your_project --pre_announce --coverage --coverpkg dymos;
    - travis-sphinx build --source=your_project/docs;

    after_success:
    - if [ "$MASTER_BUILD" ] && [ "$UPLOAD_DOCS" ]; then
        travis-sphinx deploy;
      fi
    - coveralls;

Coverage
--------

Coverage is a tool that shows developers how many lines of their code that are being executed by your current test suite.

To use this tool, set up an account at the website `coveralls.io <https://coveralls.io>`_, login using your Github credentials,
and then activate the appropriate github repo for coverage results. This sets up the site to receive results. The other side
of the equation is taken care of in your .travis.yml file, as I will show here using OpenMDAO's example:

.. code-block:: python

    script:
        testflo -n 1 openmdao --coverage  --coverpkg openmdao --cover-omit \*tests/\*  --cover-omit \*devtools/\* --cover-omit \*test_suite/\* --cover-omit \*docs/\*;

    after_success:
        coveralls

The point of the above example is to:
    #. Use testflo to run the test suite,
    #. Use testflo to set the `coverpkg` to your project, (collect coverage data on that package)
    #. Use tesflo to `cover-omit` directories you'd like to exclude from coverage, and
    #. Use coveralls to send coverage data to coveralls.io

For more information on testflo, please see `testflo on pypi <https://pypi.org/project/testflo>`_
