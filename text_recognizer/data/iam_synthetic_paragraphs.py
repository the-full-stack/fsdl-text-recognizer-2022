"""IAM Synthetic Paragraphs Dataset class."""
import argparse
import random
from typing import Any, List, Sequence, Tuple

import numpy as np
from PIL import Image
from pytorch_lightning.utilities.rank_zero import rank_zero_info

from text_recognizer.data.base_data_module import load_and_print_info
from text_recognizer.data.iam import IAM
from text_recognizer.data.iam_lines import line_crops_and_labels, load_line_crops_and_labels, save_images_and_labels
from text_recognizer.data.iam_paragraphs import (
    get_dataset_properties,
    IAMParagraphs,
)
from text_recognizer.data.util import BaseDataset, convert_strings_to_labels, resize_image
import text_recognizer.metadata.iam_synthetic_paragraphs as metadata
from text_recognizer.stems.paragraph import ParagraphStem

IMAGE_SCALE_FACTOR = metadata.IMAGE_SCALE_FACTOR
NEW_LINE_TOKEN = metadata.NEW_LINE_TOKEN

PROCESSED_DATA_DIRNAME = metadata.PROCESSED_DATA_DIRNAME


class IAMSyntheticParagraphs(IAMParagraphs):
    """IAM Handwriting database synthetic paragraphs."""

    def __init__(self, args: argparse.Namespace = None):
        super().__init__(args)
        self.trainval_transform.scale_factor = 1  # we perform rescaling ahead of time, in prepare_data
        self.transform.scale_factor = 1

    def prepare_data(self, *args, **kwargs) -> None:
        """
        Prepare IAM lines such that they can be used to generate synthetic paragraphs dataset in setup().
        This method is IAMLines.prepare_data + resizing of line crops.
        """
        if PROCESSED_DATA_DIRNAME.exists():
            return
        rank_zero_info(
            "IAMSyntheticParagraphs.prepare_data: preparing IAM lines for synthetic IAM paragraph creation..."
        )

        iam = IAM()
        iam.prepare_data()

        for split in ["train", "val"]:  # synthetic dataset is only used in training phase
            rank_zero_info(f"Cropping IAM line regions and loading labels for {split} data split...")
            crops, labels = line_crops_and_labels(iam, split)

            crops = [resize_image(crop, IMAGE_SCALE_FACTOR) for crop in crops]

            rank_zero_info(f"Saving images and labels at {PROCESSED_DATA_DIRNAME} for {split} data split...")
            save_images_and_labels(crops, labels, split, PROCESSED_DATA_DIRNAME)

    def setup(self, stage: str = None) -> None:
        rank_zero_info(f"IAMSyntheticParagraphs.setup({stage}): Loading trainval IAM paragraph regions and lines...")

        def _load_dataset(split: str, transform: ParagraphStem) -> BaseDataset:
            line_crops, line_labels = load_line_crops_and_labels(split, PROCESSED_DATA_DIRNAME)
            X, para_labels = generate_synthetic_paragraphs(line_crops=line_crops, line_labels=line_labels)
            Y = convert_strings_to_labels(strings=para_labels, mapping=self.inverse_mapping, length=self.output_dims[0])
            return BaseDataset(X, Y, transform=transform)

        if stage == "fit" or stage is None:
            self.data_train = _load_dataset(split="train", transform=self.trainval_transform)
            self.data_val = _load_dataset(split="val", transform=self.transform)

        if stage == "train_only":
            self.data_train = _load_dataset(split="train", transform=self.trainval_transform)

    def __repr__(self) -> str:
        """Print info about the dataset."""
        basic = (
            "IAM Synthetic Paragraphs Dataset\n"
            f"Num classes: {len(self.mapping)}\n"
            f"Input dims : {self.input_dims}\n"
            f"Output dims: {self.output_dims}\n"
        )
        if self.data_train is None:
            return basic

        x, y = next(iter(self.train_dataloader()))
        data = (
            f"Train/val/test sizes: {len(self.data_train)}, {len(self.data_val)}, 0\n"
            f"Train Batch x stats: {(x.shape, x.dtype, x.min(), x.mean(), x.std(), x.max())}\n"
            f"Train Batch y stats: {(y.shape, y.dtype, y.min(), y.max())}\n"
        )
        return basic + data


def generate_synthetic_paragraphs(
    line_crops: List[Image.Image], line_labels: List[str], max_batch_size: int = 12
) -> Tuple[List[Image.Image], List[str]]:
    """
    Generate synthetic paragraphs and corresponding labels by randomly joining different subsets of crops.
    These synthetic paragraphs are generated such that the number of paragraphs with 1 line of text is greater
    than the number of paragraphs with 2 lines of text is greater than the number of paragraphs with 3 lines of text
    and so on.
    """
    paragraph_properties = get_dataset_properties()

    indices = list(range(len(line_labels)))
    assert max_batch_size < paragraph_properties["num_lines"]["max"]

    batched_indices_list = [[_] for _ in indices]  # batch_size = 1, len = 9462
    batched_indices_list.extend(
        generate_random_batches(values=indices, min_batch_size=2, max_batch_size=(1 * max_batch_size) // 4)
    )
    batched_indices_list.extend(
        generate_random_batches(values=indices, min_batch_size=2, max_batch_size=(2 * max_batch_size) // 4)
    )
    batched_indices_list.extend(
        generate_random_batches(values=indices, min_batch_size=2, max_batch_size=(3 * max_batch_size) // 4)
    )
    batched_indices_list.extend(
        generate_random_batches(values=indices, min_batch_size=2, max_batch_size=max_batch_size)
    )
    batched_indices_list.extend(
        generate_random_batches(
            values=indices, min_batch_size=(2 * max_batch_size) // 4 + 1, max_batch_size=max_batch_size
        )
    )
    batched_indices_list.extend(
        generate_random_batches(
            values=indices, min_batch_size=(3 * max_batch_size) // 4 + 1, max_batch_size=max_batch_size
        )
    )
    # assert sorted(list(itertools.chain(*batched_indices_list))) == indices

    unique, counts = np.unique([len(_) for _ in batched_indices_list], return_counts=True)
    for batch_len, count in zip(unique, counts):
        rank_zero_info(f"{count} samples with {batch_len} lines")

    para_crops, para_labels = [], []
    for para_indices in batched_indices_list:
        para_label = NEW_LINE_TOKEN.join([line_labels[i] for i in para_indices])
        if len(para_label) > paragraph_properties["label_length"]["max"]:
            print("Label longer than longest label in original IAM Paragraphs dataset - hence dropping")
            continue

        para_crop = join_line_crops_to_form_paragraph([line_crops[i] for i in para_indices])
        max_para_shape = paragraph_properties["crop_shape"]["max"]
        if para_crop.height > max_para_shape[0] or para_crop.width > max_para_shape[1]:
            print("Crop larger than largest crop in original IAM Paragraphs dataset - hence dropping")
            continue

        para_crops.append(para_crop)
        para_labels.append(para_label)

    assert len(para_crops) == len(para_labels)
    return para_crops, para_labels


def join_line_crops_to_form_paragraph(line_crops: Sequence[Image.Image]) -> Image.Image:
    """Horizontally stack line crops and return a single image forming the paragraph."""
    crop_shapes = np.array([_.size[::-1] for _ in line_crops])
    para_height = crop_shapes[:, 0].sum()
    para_width = crop_shapes[:, 1].max()

    para_image = Image.new(mode="L", size=(para_width, para_height), color=0)
    current_height = 0
    for line_crop in line_crops:
        para_image.paste(line_crop, box=(0, current_height))
        current_height += line_crop.height
    return para_image


def generate_random_batches(values: List[Any], min_batch_size: int, max_batch_size: int) -> List[List[Any]]:
    """
    Generate random batches of elements in values without replacement and return the list of all batches. Batch sizes
    can be anything between min_batch_size and max_batch_size including the end points.
    """
    shuffled_values = values.copy()
    random.shuffle(shuffled_values)

    start_id = 0
    grouped_values_list = []
    while start_id < len(shuffled_values):
        num_values = random.randint(min_batch_size, max_batch_size)
        grouped_values_list.append(shuffled_values[start_id : start_id + num_values])
        start_id += num_values
    assert sum([len(_) for _ in grouped_values_list]) == len(values)
    return grouped_values_list


if __name__ == "__main__":
    load_and_print_info(IAMSyntheticParagraphs)
