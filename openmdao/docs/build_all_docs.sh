#!/bin/bash
echo "NOTE: if 'Bad config' error occurs during ipcluster startup, try deleting the $HOME/.ipython/profile_mpi directory"
rm -rf openmdao_book/_srcdocs openmdao_book/_build
export OLD_OPENMDAO_REPORTS=${OPENMDAO_REPORTS}
export OPENMDAO_REPORTS=0

python build_source_docs.py;
jupyter-book build openmdao_book || export OPENMDAO_REPORTS=${OLD_OPENMDAO_REPORTS}
python copy_build_artifacts.py;
