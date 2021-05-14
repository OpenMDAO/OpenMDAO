#!/bin/bash
python build_source_docs.py;
jupyter-book build -W --keep-going openmdao_book
python copy_build_artifacts.py;
