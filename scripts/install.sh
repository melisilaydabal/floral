#!/bin/bash

repo_path=$~/floral    # adjust this after you clone the repository in your local machine

virtualenv -p python3 ./floral_venv  # path to venv

pip install -r ./requirements.txt
source ./floral_venv/bin/activate

##deactivate
