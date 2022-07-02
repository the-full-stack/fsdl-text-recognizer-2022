#!/bin/bash
set -uo pipefail
set +e

FAILURE=false

# apply safety checking to the production environment
echo "safety (failure is tolerated)"
FILE=requirements/prod.txt
if [ -f "$FILE" ]; then
    # We're in the main repo
    safety check -r requirements/prod.txt
else
    # We're in the labs repo
    safety check -r ../requirements/prod.txt
fi

# apply automatic formatting
echo "black"
pre-commit run black || FAILURE=true

# check for python code style violations, see .flake8 for details
echo "flake8"
pre-commit run flake8 || FAILURE=true

# check for shell scripting style violations
echo "shellcheck"
pre-commit run shellcheck || FAILURE=true

# check python types
echo "mypy"
pre-commit run mypy || FAILURE=true

if [ "$FAILURE" = true ]; then
  echo "Linting failed"
  exit 1
fi
echo "Linting passed"
exit 0
