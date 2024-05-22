# this script will populate a conda environment with the latest versions
# (including pre-release versions) of all OpenMDAO dependencies

echo "============================================================="
echo "Upgrade to latest pip"
echo "============================================================="
python -m pip install --upgrade pip

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
echo "Install latest versions of 'jax' dependencies"
echo "============================================================="
python -m pip install --upgrade --pre jax jaxlib

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
python -m pip install --upgrade --pre playwright
python -m pip install --upgrade --pre num2words

echo "============================================================="
echo "Install latest versions of other optional packages"
echo "============================================================="
python -m pip install --upgrade --pre pyparsing psutil objgraph pyxdsm
python -m pip install --upgrade --pre jax jaxlib

echo "============================================================="
echo "Display conda environment"
echo "============================================================="
conda info
conda list
