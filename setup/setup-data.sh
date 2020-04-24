#!/bin/bash

if [ -e data/dblp ]
then echo "DBLP dataset exists"
else
    echo "Dowloading DBLP dataset ..."
    cd ../data && mkdir dblp && cd dblp

    wget https://snap.stanford.edu/data/bigdata/communities/com-dblp.ungraph.txt.gz
    echo "Extracting com-dblp.ungraph.txt from gzip ..."
    gunzip com-dblp.ungraph.txt.gz
    cd ../..
fi
