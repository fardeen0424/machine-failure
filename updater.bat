:: Windows updater file for updating and installing all the required packages from requirements.txt file

:: This script updates existing packages and installs new ones if there are any in requirements.txt

:: To execute this file, open cmd and run the following command in cmd: updater.bat

@echo off

rem Activate the virtual environment
call venv\Scripts\activate

rem Update all installed packages and install any new ones from requirements.txt
echo Updating and installing required packages...
pip install --upgrade -r requirements.txt

echo Hooray! All packages are up to date.
