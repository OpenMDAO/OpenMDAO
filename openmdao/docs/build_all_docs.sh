#!/bin/bash
rm -rf openmdao_book/_srcdocs openmdao_book/_build
python build_source_docs.py;
export OLD_OPENMDAO_REPORTS=${OPENMDAO_REPORTS}
export OPENMDAO_REPORTS=0
jupyter-book build -W --keep-going openmdao_book || export OPENMDAO_REPORTS=${OLD_OPENMDAO_REPORTS}
python copy_build_artifacts.py;
