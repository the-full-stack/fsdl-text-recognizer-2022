#!/bin/bash
set -uo pipefail
set +e

FAILURE=false

# unit tests check whether current best model is working, so we stage it
python ./training/stage_model.py --fetch --entity cfrye59 --from_project fsdl-text-recognizer-2021-training --to_project fsdl-testing-2022-ci || FAILURE=true
# pytest configuration in pyproject.toml
python -m pytest || FAILURE=true

./training/tests/test_run_experiment.sh || FAILURE=true

if [ "$FAILURE" = true ]; then
  echo "Unit tests failed"
  exit 1
fi
echo "Unit tests passed"
exit 0
