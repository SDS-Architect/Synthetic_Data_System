#!/bin/bash
echo "Create SDS enviroment and install? (y/n)?"
read build_env_choice

if [[($build_env_choice == "n")]]; then
echo "You can build your own custom enviroment if you want"

fi

### Building the env
if [[($build_env_choice == "y")]]; then
echo "Building env"

conda init bash
tset

echo ""
echo 'Creating the SDS enviroment'

conda create -n 'sds_env' python=3.6 conda

conda activate sds_env

pip install --upgrade pip

pip install --no-cache-dir SDS-0.1a0-py3-none-any.whl

echo ""
echo "Completed install"

fi
