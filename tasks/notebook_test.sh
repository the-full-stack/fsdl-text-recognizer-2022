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

# look inside notebooks dir for .ipynbs that are not nbconvert files and that match the pattern and not the ignore and pass them to nbconvert to run
find notebooks -maxdepth 1 | grep \.ipynb$ | grep -v nbconvert | grep "$SELECT_PATTERN" | grep -v "$IGNORE_PATTERN" | xargs jupyter nbconvert --to notebook --execute
