#!/bin/bash 

# File for creating a virtual environment and all the required packages from requirements.txt file

# For MacOS and Linux distributions

# Make the script executable by running the following command in the terminal: chmod +x setup.sh

# Run the script by executing: ./setup.sh

# Check if the virtual environment folder exists

if [ ! -d "venv" ]; then
    echo "Creating virtual environment and installing required packages, no need to do anything..."
    pip install virtualenv
    virtualenv venv
fi

# Activate the virtual environment

source venv/bin/activate

# Install the required packages

echo "Installing required packages..."
pip install -r requirements.txt

echo "Hooray Setup complete! Run the project..."
