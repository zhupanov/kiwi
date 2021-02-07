#!/usr/bin/env bash

set -euo pipefail

project_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. >/dev/null 2>&1 && pwd )"
cd "$project_dir"

echo "Installing python dependencies"

# Install Python libraries required in prod
python3 -m pip install --no-cache-dir --requirement "$project_dir/requirements.txt"

# Install Python libraries required in dev environment (linters, etc.)
python3 -m pip install --no-cache-dir --requirement "$project_dir/requirements_dev.txt"

ubuntu=$(lsb_release -a | grep 'Release:' | awk '{print $2}' | sed 's/\./ /' | awk '{print $1}')
echo "=========================="
echo "Detected Ubuntu version $ubuntu"
python3 --version
