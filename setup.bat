:: Windows setup file for creating a virtual environment and all the required packages from requirements.txt file

:: To execute this file open cmd and run the following command in cmd : setup.bat


@echo off
rem Check if the virtual environment folder exists
if not exist venv (
    echo Creating virtual environment and required packages no neeed to do anything...
    pip install virtualenv 
    virtualenv venv
)

rem Activate the virtual environment
call venv\Scripts\activate

rem Install the required packages without any prompts
echo Installing required packages...
pip install -r requirements.txt 

echo Hooray Setup complete! Run the project...
