#!/usr/bin/env bash

set +o pipefail

project_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. >/dev/null 2>&1 && pwd )"

if [ "$#" -ne "1" ]
then
  echo "Usage: $0 <python-file-to-lint>"
  exit 1
fi

python_script=$1

found_issues=0

echo "--------------------------------------------------------------------------------------------------------"
echo "=== Linting $python_script"
python3 -m pylint --rcfile "$project_dir/pylintrc" "$python_script"

ret=$?
if [ "$ret" -ne "0" ]
then
  found_issues=1
fi

echo "--------------------------------------------------------------------------------------------------------"
echo "=== Running MyPy check of $python_script"
python3 -m mypy \
  --follow-imports=silent \
  --strict \
  --ignore-missing-imports \
  --disallow-untyped-defs \
  --disallow-incomplete-defs \
  --no-implicit-optional \
  --warn-redundant-casts \
  --warn-unused-ignores \
  --warn-return-any \
  --warn-unreachable \
  --strict-equality \
  --show-error-context \
  --show-error-codes \
  --pretty \
  "$python_script"

ret=$?
if [ "$ret" -ne "0" ]
then
  found_issues=1
fi

if [ "$found_issues" -ne "0" ]
then
  exit 1
fi
