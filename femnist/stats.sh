#!/usr/bin/env bash

NAME="femnist"

cd ../utils

python.exe stats.py --name $NAME

cd ../$NAME