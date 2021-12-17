#!/bin/bash

source ./py-env/bin/activate

current_directory=$(pwd)

python py-env/py-code/main.py $current_directory
