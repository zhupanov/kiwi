#!/usr/bin/env bash

set -euo pipefail

project_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. >/dev/null 2>&1 && pwd )"
cd "$project_dir/python"

all_python_tests=$(find . -type f | grep -E '^.*_test.py$')
echo "========================================================================================================"
echo "$all_python_tests"

for python_test in $all_python_tests
do
  echo "--------------------------------------------------------------------------------------------------------"
  echo "=== Running $python_test"
  python3 "$python_test"
done
