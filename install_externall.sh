#!/bin/bash

echo "Installing dependencies"
echo "Installing ACE 0.9.34"
wget http://sweaglesw.org/linguistics/ace/download/ace-0.9.34-x86-64.tar.gz -P ./external/
tar -zxvf ./external/ace-0.9.34-x86-64.tar.gz -C ./external/
rm ./external/ace-0.9.34-x86-64.tar.gz

echo "Installing ERG 2018"
wget http://sweaglesw.org/linguistics/ace/download/erg-2018-x86-64-0.9.34.dat.bz2 -P ./external/
bzip2  -d ./external/erg-2018-x86-64-0.9.34.dat.bz2

echo "Installing Utool 3.1.1"
wget https://www.coli.uni-saarland.de/projects/chorus/utool/download/Utool-3.1.1.jar -P ./external/ --no-check-certificate
