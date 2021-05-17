#!/bin/bash
rm -rf openmdao_book/_srcdocs openmdao_book/_build
python build_source_docs.py;
jupyter-book build -W --keep-going openmdao_book
python copy_build_artifacts.py;
