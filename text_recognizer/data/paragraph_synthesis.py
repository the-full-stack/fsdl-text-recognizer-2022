"""Utilities for generating synthetic paragraphs from real lines."""
import random
from typing import Any, List, Sequence, Tuple

import numpy as np
from PIL import Image as image
from PIL.Image import Image as Image
from pytorch_lightning.utilities.rank_zero import rank_zero_info

from text_recognizer.data.iam_paragraphs import get_dataset_properties
import text_recognizer.metadata.iam_synthetic_paragraphs as metadata


NEW_LINE_TOKEN = metadata.NEW_LINE_TOKEN


def build_paragraph_from_indices(line_crops, line_labels):
    """Given a list of images and labels of lines of text, create an image and a label of a paragraph of text."""
    paragraph_properties = get_dataset_properties()

    para_label = NEW_LINE_TOKEN.join(line_labels)

    if len(para_label) > paragraph_properties["label_length"]["max"]:
        rank_zero_info("Label longer than longest label in original IAM Paragraphs dataset -- dropping")
        return (None, None)

    para_crop = join_line_crops_to_form_paragraph(line_crops)

    max_para_shape = paragraph_properties["crop_shape"]["max"]
    if para_crop.height > max_para_shape[0] or para_crop.width > max_para_shape[1]:
        rank_zero_info("Crop larger than largest crop in original IAM Paragraphs dataset -- dropping")
        return (None, para_label)

    return para_crop, para_label


def generate_synthetic_paragraphs(
    line_crops: List[Image], line_labels: List[str], max_size: int = 9
) -> Tuple[List[Image], List[str]]:
    """Generate synthetic paragraphs and corresponding labels by randomly joining subsets of lines.

    A paragraph is an ordered set, aka a list, of lines. So we synthesize a new dataset of paragraphs
    by drawing from our dataset of lines in a different order and chunking them differently -- creating
    a new partition of the original dataset of lines. We repeat this in order to increase the amount of data.
    """
    # we operate at the level of indices while defining the partition.
    indices = list(range(len(line_labels)))

    # we start by generating a "one-line paragraph" for each line in the dataset
    single_line_indices = [[idx] for idx in indices]

    # then we split the data into shorter paragraphs with just a few lines
    short_paragraph_indices = generate_random_partition(indices, min_size=2, max_size=max_size // 2)

    # and then into longer paragraphs
    long_paragraph_indices = generate_random_partition(indices, min_size=(max_size // 2) + 1, max_size=max_size)

    # and finally, paragraphs with lengths drawn uniformly at random from two lines to maximum_size
    uniform_paragraph_indices = generate_random_partition(values=indices, min_size=2, max_size=max_size)

    # and we combine them together to get our new dataset of synthetic paragraphs
    all_paragraph_indices = (
        single_line_indices + short_paragraph_indices + long_paragraph_indices + uniform_paragraph_indices
    )

    _report_paragraph_sizes(all_paragraph_indices)

    # then we use the indices of the lines to pick out the cropped lines and their labels and join them together
    para_crops, para_labels = _build_paragraphs_from_indices(all_paragraph_indices, line_crops, line_labels)

    return para_crops, para_labels


def join_line_crops_to_form_paragraph(line_crops: Sequence[Image]) -> Image:
    """Horizontally stack line crops and return a single image forming the paragraph."""
    crop_shapes = np.array([_.size[::-1] for _ in line_crops])
    para_height = crop_shapes[:, 0].sum()
    para_width = crop_shapes[:, 1].max()

    para_image = image.new(mode="L", size=(para_width, para_height), color=0)
    current_height = 0
    for line_crop in line_crops:
        para_image.paste(line_crop, box=(0, current_height))
        current_height += line_crop.height
    return para_image


def _build_paragraphs_from_indices(
    all_paragraph_indices: List[List[int]], line_crops, line_labels
) -> Tuple[List[Image], List[str]]:
    """For each list of line indices defining a paragraph, synthesize an image and a label using the lines' label and image."""
    para_crops, para_labels = [], []
    for paragraph_line_indices in all_paragraph_indices:

        para_line_crops = [line_crops[idx] for idx in paragraph_line_indices]
        para_line_labels = [line_labels[idx] for idx in paragraph_line_indices]

        para_crop, para_label = build_paragraph_from_indices(para_line_crops, para_line_labels)

        if para_crop is None or para_label is None:
            continue
        else:
            para_crops.append(para_crop)
            para_labels.append(para_label)

    assert len(para_crops) == len(para_labels)

    return para_crops, para_labels


def generate_random_partition(values, min_size: int, max_size: int):
    """Randomly partitions the provided list of values, returning a list of sublists.

    Parameters
    ----------
    values
        A list of values to partition.
    min_size
        The minimum size of all except one of the elements of the partition, the last.
    max_size
        The maximum size of the elements of the partition.
    """
    shuffled_values = values.copy()
    random.shuffle(shuffled_values)

    partition, current_idx, total = [], 0, 0
    while current_idx < len(values):
        num_values = random.randint(min_size, max_size)
        chunk = shuffled_values[current_idx : current_idx + num_values]
        partition.append(chunk)
        current_idx += num_values
        total += len(chunk)

    # confirm we didn't lose any elements
    assert total == len(values)

    return partition


def _report_paragraph_sizes(paragraph_indices: List[List[int]]):
    lengths, counts = _count_lengths(paragraph_indices)
    for length, count in zip(lengths, counts):
        rank_zero_info(f"{count} samples with {length} lines")


def _count_lengths(paragraphs: List[List[Any]]):
    lengths, counts = np.unique([len(pargraph) for pargraph in paragraphs], return_counts=True)
    return lengths, counts
