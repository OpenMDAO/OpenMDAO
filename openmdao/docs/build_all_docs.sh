#!/bin/bash
if [ ! -d "./openmdao_book/OpenMDAO/" ]; then
    echo "Temporarily installing OpenMDAO repo to build the source docs";
    cd openmdao_book/;
    git clone https://github.com/OpenMDAO/OpenMDAO.git;
    cd ..;
fi
python build_source_docs.py;
rm -rf openmdao_book/OpenMDAO;
#jupyter-book build -W --keep-going openmdao_book
jupyter-book build openmdao_book
python copy_build_artifacts.py;
