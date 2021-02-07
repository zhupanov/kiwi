#!/usr/bin/env bash

set +o pipefail

project_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. >/dev/null 2>&1 && pwd )"

all_scripts=$(find "$project_dir/python" -type f -name '*.py' | grep -v 'mock-good-bad-api' | grep -v '\.git')
echo "========================================================================================================"
echo "$all_scripts"

found_issues=0

# print pylint version
python3 -m pylint --version

for python_script in $all_scripts
do
  "$project_dir/bin/lint_one_python.sh" "$python_script"
  ret=$?
  if [ "$ret" -ne "0" ]
  then
    found_issues=1
  fi
done

echo "========================================================================================================"
echo "=== Running Flake8 checks of $all_scripts"
echo "========================================================================================================"

# print flake8 version
flake8 --version

# stop the build if there are Python syntax errors or undefined names
flake8 ./python --count --select=E9,F63,F7,F82 --show-source --statistics
ret=$?
if [ "$ret" -ne "0" ]
then
  found_issues=1
fi

# exit-zero treats all errors as warnings. The GitHub editor is 127 chars wide
flake8 ./python --count --max-complexity=10 --max-line-length=127 --statistics
ret=$?
if [ "$ret" -ne "0" ]
then
  found_issues=1
fi

echo "========================================================================================================"
echo "=== grepping for TODO in all python scripts:"
echo "========================================================================================================"
for python_script in $all_scripts
do
  grep -Hin "TODO" "$python_script"
done

RED='\033[1;31m'
GREEN='\033[1;32m'
NO_COLOR='\033[0m'

if [ "$found_issues" -ne "0" ]
then
  echo -e "${RED}BAD! SOME ISSUES FOUND.${NO_COLOR}"
  exit 1
else
  echo -e "${GREEN}GOOD! NO ISSUES FOUND.${NO_COLOR}"
fi
