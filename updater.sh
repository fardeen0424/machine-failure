#!/bin/bash

# Bash updater file for updating and installing all the required packages from requirements.txt file

# This script updates existing packages and installs new ones if there are any in requirements.txt

# For MacOS and Linux distributions

# Make the script executable by running the following command in the terminal: chmod +x updater.sh

# Run the script by executing: ./updater.sh

# Activate the virtual environment
source venv/bin/activate

# Update all installed packages and install any new ones from requirements.txt
echo "Updating and installing required packages..."
pip install --upgrade -r requirements.txt

echo "Hooray! All packages are up to date."
