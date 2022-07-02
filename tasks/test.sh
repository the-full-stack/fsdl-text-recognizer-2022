#!/bin/bash
set -uo pipefail
set +e

FAILURE=false

./tasks/unit_test.sh || FAILURE=true
./tasks/integration_test.sh || FAILURE=true

if [ "$FAILURE" = true ]; then
  echo "Tests failed"
  exit 1
fi
echo "Tests passed"
exit 0
