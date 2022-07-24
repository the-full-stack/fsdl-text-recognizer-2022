#!/bin/bash
set -uo pipefail
set +e

# provide an escaped regex pattern, e.g. "99_\|03_", to override
# use /(\?\!)/, an unmatchable regex (from https://codegolf.stackexchange.com/questions/18393), to match all
IGNORE_PATTERN="${1:-99_}"

# use environment variables to control notebook bootstrapping behavior, see notebook setup cells and https://fsdl.me/bootstrap-gist
FSDL_REPO="$(basename "$(pwd)")"
export FSDL_REPO

find notebooks -maxdepth 1 | grep \.ipynb$ | grep -v nbconvert | grep -v "$IGNORE_PATTERN" | xargs jupyter nbconvert --to notebook --execute
