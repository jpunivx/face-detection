#!/bin/bash

python3 -m venv ./py-env

cd ./py-env

source ./bin/activate

pip install -r requirements_base.txt

pip install -r requirements.txt

cd ..

current_directory=$(pwd)

python3 ./py-env/py-code/main.py $current_directory

deactivate
