#!/bin/bash

UP_TO=${1-10}
python admin/tasks/subset_repo_for_labs.py --output_dir ../fsdl-text-recognizer-2022-labs --up_to "$UP_TO" && black ../fsdl-text-recognizer-2022-labs
