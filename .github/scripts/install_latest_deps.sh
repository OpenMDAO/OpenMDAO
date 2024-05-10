# this script will populate a conda environment with the latest versions
# (including pre-release versions) of all OpenMDAO dependencies

conda upgrade -c conda-forge --all -y

echo "============================================================="
echo "Upgrade to latest pip"
echo "============================================================="
python -m pip install --upgrade pip

echo "============================================================="
echo "Install latest setuptools"
echo "============================================================="
python -m pip install --upgrade --pre setuptools

echo "============================================================="
echo "Install latest versions of NumPy/SciPy"
echo "============================================================="
python -m pip install --upgrade --pre numpy
python -m pip install --upgrade --pre scipy

# remember versions so we can check them later
NUMPY_VER=`python -c "import numpy; print(numpy.__version__)"`
SCIPY_VER=`python -c "import scipy; print(scipy.__version__)"`
echo "NUMPY_VER=$NUMPY_VER" >> $GITHUB_ENV
echo "SCIPY_VER=$SCIPY_VER" >> $GITHUB_ENV

echo "============================================================="
echo "Install latest versions of 'required' dependencies"
echo "============================================================="
python -m pip install --upgrade --pre networkx
python -m pip install --upgrade --pre requests
python -m pip install --upgrade --pre packaging

echo "============================================================="
echo "Install latest versions of 'docs' dependencies"
echo "============================================================="
python -m pip install --upgrade --pre matplotlib
python -m pip install --upgrade --pre numpydoc
python -m pip install --upgrade --pre jupyter-book
python -m pip install --upgrade --pre sphinx-sitemap
python -m pip install --upgrade --pre ipyparallel

echo "============================================================="
echo "Install latest versions of 'doe' dependencies"
echo "============================================================="
python -m pip install --upgrade --pre pyDOE3

echo "============================================================="
echo "Install latest versions of 'notebooks' dependencies"
echo "============================================================="
python -m pip install --upgrade --pre notebook
python -m pip install --upgrade --pre ipympl

echo "============================================================="
echo "Install latest versions of 'visualization' dependencies"
echo "============================================================="
python -m pip install --upgrade --pre bokeh
python -m pip install --upgrade --pre colorama

echo "============================================================="
echo "Install latest versions of 'test' dependencies"
echo "============================================================="
python -m pip install --upgrade --pre parameterized
python -m pip install --upgrade --pre numpydoc
python -m pip install --upgrade --pre pycodestyle
python -m pip install --upgrade --pre pydocstyle
python -m pip install --upgrade --pre testflo
python -m pip install --upgrade --pre websockets
python -m pip install --upgrade --pre aiounittest
echo "the latest version of playwright (1.38.0) is pinned to"
echo "greenlet 2.0.2 which does not build properly for python 3.12"
echo "https://github.com/microsoft/playwright-python/issues/2096"
python -m pip install --upgrade --pre playwright  || echo "Skipping playwright, no GUI testing!"
python -m pip install --upgrade --pre num2words

echo "============================================================="
echo "Install latest versions of other optional packages"
echo "============================================================="
python -m pip install --upgrade --pre pyparsing psutil objgraph pyxdsm
python -m pip install --upgrade --pre jax jaxlib

echo "============================================================="
echo "Install latest PETSc"
echo "============================================================="
conda install mpi4py petsc!=3.21.1 petsc4py -q -y

echo "============================================================="
echo "Check MPI and PETSc installation"
echo "============================================================="
export OMPI_MCA_rmaps_base_oversubscribe=1
export OMPI_MCA_btl=^openib

echo "OMPI_MCA_rmaps_base_oversubscribe=1" >> $GITHUB_ENV
echo "OMPI_MCA_btl=^openib" >> $GITHUB_ENV

echo "-----------------------"
echo "Quick test of mpi4py:"
mpirun -n 3 python -c "from mpi4py import MPI; print(f'Rank: {MPI.COMM_WORLD.rank}')"
echo "-----------------------"
echo "Quick test of petsc4py:"
mpirun -n 3 python -c "import numpy; from mpi4py import MPI; comm = MPI.COMM_WORLD; \
                       import petsc4py; petsc4py.init(); \
                       x = petsc4py.PETSc.Vec().createWithArray(numpy.ones(5)*comm.rank, comm=comm);  \
                       print(x.getArray())"
echo "-----------------------"

echo "============================================================="
echo "Display conda environment"
echo "============================================================="
conda info
conda list
