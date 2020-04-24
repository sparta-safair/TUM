#!/bin/bash

if [ -e ~/bin/metis-5.1.0 ]
then printf "\nMETIS library folder exists already\n"
else
    printf "\nDownloading METIS library ...\n\n"
    mkdir ~/bin && cd ~/bin

    wget http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/metis-5.1.0.tar.gz
    echo "Extracting metis-5.1.0 from gzip ..."
    gunzip metis-5.1.0.tar.gz
    tar -xvf metis-5.1.0.tar

    echo "Installing METIS ..."
    cd metis-5.1.0
    make config shared=1
    make
fi
