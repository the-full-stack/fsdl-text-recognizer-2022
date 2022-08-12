#!/bin/bash
set -uo pipefail
set +e

# provide escaped regex patterns, e.g. "99_\|03_", as CLI args to override these
SELECT_PATTERN="${1:-.*}"
IGNORE_PATTERN="${2:-99_}"
# to match all, use ".*" and "\$-" (see discussion at https://stackoverflow.com/questions/2930182/regex-to-not-match-any-characters)
# use https://regex101.com/r/yjcs1Z/3 to an interactive regex testing tool

# use environment variables to control notebook bootstrapping behavior, see notebook setup cells and https://fsdl.me/bootstrap-gist
FSDL_REPO=$(basename "$(git rev-parse --show-toplevel)")
export FSDL_REPO

# setting a maximum allowable runtime, in seconds, for any single cell
#  note that this will be hardware dependent, so we add a 20% buffer above the 5 minute target
MAX_RUNTIME=360

# create an array of NOTEBOOK filenames by...
mapfile -t NOTEBOOKS <<< "$(
	find notebooks -maxdepth 1 | grep \.ipynb$ | grep -v nbconvert                 | grep "$SELECT_PATTERN"  | grep -v "$IGNORE_PATTERN" | sort)"
	# 1) searching ./notebooks 2) for .ipynbs  3) that aren't nbconvert files and  4) match the selector and 5) don't match the filter   6) and sorting them

echo "Testing these notebooks: ${NOTEBOOKS[*]}"
FAILURE=false
for NOTEBOOK in "${NOTEBOOKS[@]}"
do  # loop over notebooks, executing each one
    echo "Testing $NOTEBOOK"
    jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=$MAX_RUNTIME --execute "$NOTEBOOK" || FAILURE=true
done

if [ "$FAILURE" = true ]; then
    echo "Notebook tests failed"
    exit 1
fi
echo "Notebook tests passed"
