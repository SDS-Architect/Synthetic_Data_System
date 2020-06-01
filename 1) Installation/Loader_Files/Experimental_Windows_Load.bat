@echo off
rem This file should auto-create a new conda env for the user
rem bonus points: auto-loads whl

SET /P CONT="Create SDS enviroment? (y/n)?"

if %CONT% == n (echo Exiting & exit /B) else (GOTO SETUP_ENV)


:SETUP_ENV
echo "Creating the enviroment"
call conda create --name sds_env python=3.5 conda || GOTO :ERROR
call conda activate sds_env || GOTO :ERROR
echo "Installing the SDS"
call pip install --no-cache-dir SDS-0.1a0-py3-none-any.whl || GOTO :ERROR
echo ""
echo "Completed install"
:ERROR
echo Failed with error #%errorlevel%.
exit /B
