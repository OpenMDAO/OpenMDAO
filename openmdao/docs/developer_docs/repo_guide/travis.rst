.. _`travis_ci_setup`:


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

The point of the above example is to use testflo to run things, and set the `coverpkg` to your project, and the `cover-omit` dirs to exclude from coverage, use coveralls
to collect the data, and then send the results to coveralls.io.

Caching
-------

The concept of build caching on Travis CI is intended to speed up the build, and therefore the entire build/test cycle on Travis CI.
By caching the builds of dependencies/requirements that rarely change, we can get right to our various dependencies to speed up the build
and the docbuild for our everyday testing.

Certain commonly-used things can be easily cached, using code near the top of your .travis.yml file that looks like this:

.. code-block:: none

    cache:
      apt: true
      directories:
        - $HOME/.cache/pip
        - $HOME/pyoptsparse
        - $HOME/miniconda
        - $HOME/miniconda/lib/python$PY/site-packages/pyoptsparse

Later in your .travis.yml file, you need to check for a cached version before you install, or don't install an item.
Read the comments for some not-so-intuitive news on what caching does the first time through.

.. code-block:: none

    before_install:

    # Check for existence of files to determine if cache exists
    # If the dir doesn't exist, but is slated to be cached later,
    # Travis unhelpfully creates it, which then causes "dir already exists"
    # errors when you go to actually install the thing, so we must non-intuitively
    # delete the file before re-creating it later.
    - if [ -f $HOME/miniconda/bin/python$PY ]; then
        echo "cached miniconda found -- nothing to do";
      else
        NOT_CACHED_CONDA=1;
        rm -rf $HOME/miniconda;
      fi

Finally, a last thing to cache might be something private, like in OpenMDAO's case, the code for SNOPT, to be used inside
our pyoptsparse install. To do this, we need to keep our private code in a private location, then do the following:

    #. Set up passwordless entrance to the secure location with the SNOPT source.
    #. Copy the source into the proper directory on Travis so it can be built and subsequently cached.

In fulfillment of #1, let's get a key decrypted, placed, chmodded, and added for passwordless access to a remote site:
(for full instructions, see :ref:`Advanced Operations, steps 1-3 <advanced_operations_automation>`.)

.. code-block:: none

    - if [ "$MASTER_BUILD" ]; then
        openssl aes-256-cbc -K $encrypted_74d70a284b7d_key -iv $encrypted_74d70a284b7d_iv -in travis_deploy_rsa.enc -out /tmp/travis_deploy_rsa -d;
        eval "$(ssh-agent -s)";
        chmod 600 /tmp/travis_deploy_rsa;
        ssh-add /tmp/travis_deploy_rsa;
        echo -e "Host web543.webfaction.com\n\tStrictHostKeyChecking no\n" >> ~/.ssh/config;
      fi

In fulfillment of #2, set $SNOPT_LOCATION to be an encrypted variable in your Travis CI settings that contains the
secret location of the private code.
Then we will check, and if the cache doesn't exist, we will copy it in from the secret location, and
then, following a successful build/test, it will get cached.

.. code-block:: none

    - if [ "$NOT_CACHED_PYOPTSPARSE" ]; then
        git clone https://github.com/OpenMDAO/pyoptsparse.git;
        cd pyoptsparse;

        if [ "$MASTER_BUILD" ]; then
          cd pyoptsparse/pySNOPT/source;
          scp -r "$SNOPT_LOCATION" .;
          cd ../../..;
        fi

        python setup.py install;
        cd ..;
      fi

.. note::

    There is one potentially-confusing complication to this whole process of caching of a private item. The use of an encrypted variable as described above is not allowed
    by Travis on pull requests--Travis determines bringing in encrypted variables to be a security vulnerability. In other words, encrypted stuff won't work during a PR.
    Only after that PR has been merged by a repo owner, then, during the subsequent master build, the encrypted items will work,
    and will be cached if THAT master build/test is successful.  Once the encrypted item builds and caches on master, subsequent pull-request builds WILL have
    the cached private item in their caches, because the PR builds derive their caches from the master cache.