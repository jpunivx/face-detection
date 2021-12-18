#!/bin/bash

python3 -m venv ./py-env

cd ./py-env

source ./bin/activate

pip install -r requirements.txt

current_directory=$(pwd)

python3 ./py-code/main.py $current_directory

deactivate
