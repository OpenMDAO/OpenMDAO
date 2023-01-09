#!/bin/bash
echo "NOTE: if 'Bad config' error occurs during ipcluster startup, try deleting the $HOME/.ipython/profile_mpi directory"
rm -rf openmdao_book/_srcdocs openmdao_book/_build
export OLD_OPENMDAO_REPORTS=${OPENMDAO_REPORTS}
export OPENMDAO_REPORTS=0
python build_source_docs.py 2>&1 | tee bld_src_docs.log;
jupyter-book build -W --keep-going openmdao_book 2>&1 | tee openmdao_book_build.log || export OPENMDAO_REPORTS=${OLD_OPENMDAO_REPORTS}
python copy_build_artifacts.py 2>&1 | tee cpy_bld_artifacts.log;
