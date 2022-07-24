#!/usr/bin/env python
"""
Script to generate directories (or git branches) corresponding to subsets of the repo appropriate for different labs.

The script creates a subset of files corresponding to labs with index less than or equal than the one given,
as specified in lab_specific_files.yml

Furthermore, it also strips out text between blocks like
    # Your code below (Lab01)
    # <content>
    # Your code above (Lab01)
for labs with index greater than or equal to the one given.

It also strips text between blocks like
    # Hide lines below until Lab 02
    # <content>
    # Hide lines above until Lab 02
for labs with index greater than the one given

and the strip-delineating lines themselves are also stripped out.

NOTE that the stripping is only performed on .py files.
"""
import argparse
import glob
import os
from pathlib import Path
import re
import shutil

import yaml

MAX_LAB_NUMBER = 10
REPO_DIRNAME = Path(__file__).resolve().parents[2]
INFO_FILENAME = REPO_DIRNAME / "admin" / "tasks" / "lab_specific_files.yml"
SOLUTION_VERSION_LABS = True


def subset_repo(info, output_dirname, up_to=MAX_LAB_NUMBER):
    """See module docstring."""
    # Clear output dir
    output_dir = Path(output_dirname)
    if output_dir.exists():
        for directory in glob.glob(f"{str(output_dir)}/lab*"):
            shutil.rmtree(directory)
        if os.path.exists(output_dir / "data"):
            shutil.rmtree(output_dir / "data")

    # Data
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(REPO_DIRNAME / "data" / "raw", output_dir / "data" / "raw")

    # Common files
    _copy_common_files(info, output_dir)

    # Common readme files
    shutil.copy("instructions/labs-readme.md", output_dir / "readme.md")
    setup_dir = output_dir / "setup"
    setup_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy("instructions/setup/readme.md", setup_dir)
    for filename in glob.glob("instructions/setup/*.png"):
        shutil.copy(filename, setup_dir)

    # Overview notebook
    shutil.copy("notebooks/overview.ipynb", output_dir / "overview.ipynb")

    # Lab files
    for lab_number in info.keys():
        try:
            int(lab_number)
            lab_number_str = _to_str(lab_number)
            if lab_number > up_to:
                continue
        except ValueError:
            continue
        lab_output_dir = output_dir / f"lab{lab_number_str}"
        lab_output_dir.mkdir(parents=True)

        new_paths = _copy_files_for_lab(info, lab_number, lab_output_dir)

        _process_new_files(new_paths, lab_number, filter_your_code=(not SOLUTION_VERSION_LABS))

        # local image files
        for filename in glob.glob(f"instructions/lab{lab_number_str}/*.png"):
            shutil.copy(filename, lab_output_dir)

        # testing script
        shutil.copy("tasks/notebook_test.sh", lab_output_dir / ".notebook_test.sh")


def _filter_your_code_blocks(lines, lab_number):
    """
    Strip out stuff between "Your code here" blocks.
    """
    if lab_number == MAX_LAB_NUMBER:
        lab_numbers_to_strip = _to_str(lab_number)
    else:
        lab_numbers_to_strip = f"[{'|'.join(_to_str(num) for num in range(lab_number, MAX_LAB_NUMBER))}]"
    beginning_comment = rf"# Your code below \(Lab {lab_numbers_to_strip}\)"
    ending_comment = rf"# Your code above \(Lab {lab_numbers_to_strip}\)"
    filtered_lines = []
    filtering = False
    for line in lines:
        if not filtering:
            filtered_lines.append(line)
        if re.search(beginning_comment, line):
            filtering = True
            filtered_lines.append("")
        if re.search(ending_comment, line):
            filtered_lines.append(line)
            filtering = False
    return filtered_lines


def _filter_hidden_blocks(lines, lab_number):
    if lab_number == MAX_LAB_NUMBER:
        return lines
    if lab_number + 1 == MAX_LAB_NUMBER:
        lab_numbers_to_hide = _to_str(MAX_LAB_NUMBER)
    else:
        lab_numbers_to_hide = "(" + "|".join([_to_str(num) for num in range(lab_number + 1, MAX_LAB_NUMBER)]) + ")"
    beginning_comment = rf"# Hide lines below until Lab {lab_numbers_to_hide}"
    ending_comment = rf"# Hide lines above until Lab {lab_numbers_to_hide}"
    filtered_lines = []
    filtering = False
    for line in lines:
        if re.search(beginning_comment, line):
            filtering = True
        if re.search(ending_comment, line):
            filtering = False
            continue
        if not filtering:
            filtered_lines.append(line)
    return filtered_lines


def _filter_filters(lines):
    filter_comments = ["# Hide lines", "# Your code above", "# Your code below"]
    any_filters = "|".join(filter_comments)
    filtered_lines = []
    for line in lines:
        if re.search(any_filters, line):
            continue
        else:
            filtered_lines.append(line)
    return filtered_lines


def _replace_data_dirname(lines):
    filtered_lines = []
    for line in lines:
        if 'Path(__file__).resolve().parents[2] / "data"' in line:
            line = line.replace(".parents[2]", ".parents[3]")
        filtered_lines.append(line)
    return filtered_lines


def _copy_common_files(info, output_dir):
    for path in info["common"]:
        new_path = output_dir / path
        new_path.parents[0].mkdir(parents=True, exist_ok=True)
        shutil.copy(path, new_path)


def _copy_files_for_lab(info, lab_number, lab_output_dir):
    selected_paths = sum([info.get(number, []) for number in range(lab_number + 1)], [])
    new_paths = []
    for path in selected_paths:
        new_path = lab_output_dir / path
        new_path.parents[0].mkdir(parents=True, exist_ok=True)
        shutil.copy(path, new_path)
        new_paths.append(new_path)
    return new_paths


def _process_new_files(
    new_paths, lab_number, filter_your_code=True, filter_hidden=True, filter_filters=True, replace_data_dirname=True
):
    for path in new_paths:
        if path.suffix != ".py":
            continue

        with open(path) as f:
            lines = f.read().split("\n")

        if filter_your_code:
            lines = _filter_your_code_blocks(lines, lab_number)
        if filter_hidden:
            lines = _filter_hidden_blocks(lines, lab_number)
        if replace_data_dirname:
            lines = _replace_data_dirname(lines)
        if filter_filters:
            lines = _filter_filters(lines)

        with open(path, "w") as f:
            f.write("\n".join(lines))


def _to_str(lab_number):
    return str(lab_number).zfill(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dirname", default="_labs", help="Where to output the lab subset directories. Default is ./_labs"
    )
    parser.add_argument(
        "--up_to", default=MAX_LAB_NUMBER, help="Produce labs up to this number. Default is {MAX_LAB_NUMBER}.", type=int
    )
    args = parser.parse_args()

    with open(INFO_FILENAME) as f:
        info = yaml.full_load(f.read())

    subset_repo(info, args.output_dirname, args.up_to)


if __name__ == "__main__":
    main()
