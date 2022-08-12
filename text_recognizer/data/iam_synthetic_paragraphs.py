"""IAM Synthetic Paragraphs Dataset class."""
import argparse
import random
from typing import Any, Callable, cast, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from pytorch_lightning.utilities.rank_zero import rank_zero_info
import torch

from text_recognizer.data.base_data_module import load_and_print_info
from text_recognizer.data.iam import IAM
from text_recognizer.data.iam_lines import (
    generate_line_crops_and_labels,
    load_processed_line_crops,
    load_processed_line_labels,
    save_images_and_labels,
)
from text_recognizer.data.iam_paragraphs import IAMParagraphs
from text_recognizer.data.util import convert_strings_to_labels
import text_recognizer.metadata.iam_synthetic_paragraphs as metadata


NEW_LINE_TOKEN = metadata.NEW_LINE_TOKEN

PROCESSED_DATA_DIRNAME = metadata.PROCESSED_DATA_DIRNAME


class IAMSyntheticParagraphs(IAMParagraphs):
    """IAM Handwriting database synthetic paragraphs."""

    def __init__(self, args: argparse.Namespace = None):
        super().__init__(args)
        self.line_crops: Optional[List[Image.Image]] = None
        self.line_labels: Optional[List[str]] = None
        self.trainval_transform.scale_factor = 1  # we perform rescaling ahead of time, in prepare_data

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

        for split in ["train"]:  # synthetic dataset is only used in training phase
            rank_zero_info(f"Cropping IAM line regions and loading labels for {split} data split...")
            crops, labels = generate_line_crops_and_labels(iam, split)
            save_images_and_labels(crops, labels, split, PROCESSED_DATA_DIRNAME)

    def setup(self, stage: str = None) -> None:
        rank_zero_info(f"IAMSyntheticParagraphs.setup({stage}): Loading train IAM paragraph regions and lines...")

        if stage == "fit" or stage is None:
            self._load_processed_crops_and_labels()
            self.data_train = IAMSyntheticParagraphsDataset(
                line_crops=cast(List[Image.Image], self.line_crops),
                line_labels=cast(List[str], self.line_labels),
                dataset_len=64 * 8 * 40,
                inverse_mapping=self.inverse_mapping,
                input_dims=self.input_dims,
                output_dims=self.output_dims,
                transform=self.trainval_transform,
            )

    def _load_processed_crops_and_labels(self):
        if self.line_crops is None:
            self.line_crops = load_processed_line_crops("train", PROCESSED_DATA_DIRNAME)
        if self.line_labels is None:
            self.line_labels = load_processed_line_labels("train", PROCESSED_DATA_DIRNAME)

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
            f"Train/val/test sizes: {len(self.data_train)}, 0, 0\n"
            f"Train Batch x stats: {(x.shape, x.dtype, x.min(), x.mean(), x.std(), x.max())}\n"
            f"Train Batch y stats: {(y.shape, y.dtype, y.min(), y.max())}\n"
        )
        return basic + data


class IAMSyntheticParagraphsDataset(torch.utils.data.Dataset):
    """Base Dataset class that simply processes data and targets through optional transforms.

    Parameters
    ----------
    data
        commonly these are torch tensors, numpy arrays, or PIL Images
    targets
        commonly these are torch tensors or numpy arrays
    transform
        function that takes a datum and returns the same
    target_transform
        function that takes a target and returns the same
    """

    def __init__(
        self,
        line_crops: List[Image.Image],
        line_labels: List[str],
        dataset_len: int,
        inverse_mapping: dict,
        input_dims: Tuple[int, ...],
        output_dims: Tuple[int, ...],
        transform: Callable = None,
    ) -> None:
        super().__init__()
        self.line_crops = line_crops
        self.line_labels = line_labels
        assert len(self.line_crops) == len(self.line_labels)

        self.ids = list(range(len(self.line_labels)))
        self.dataset_len = dataset_len
        self.inverse_mapping = inverse_mapping
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.transform = transform
        self.min_num_lines, self.max_num_lines = 1, 15

        self.seed_set = False

    def __len__(self) -> int:
        """Return length of the dataset."""
        return self.dataset_len

    def _set_seed(self, seed):
        if not self.seed_set:
            print(f"Setting seed to {seed} for worker {torch.utils.data.get_worker_info()}")
            random.seed(seed)
            self.seed_set = True

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Return a datum and its target, after processing by transforms."""
        # Since shuffle is True for train dataloaders, the first index will be different on different GPUs
        self._set_seed(index)
        num_lines = random.randint(self.min_num_lines, self.max_num_lines)
        indices = random.sample(self.ids, k=num_lines)

        while True:
            datum = join_line_crops_to_form_paragraph([self.line_crops[i] for i in indices])
            labels = NEW_LINE_TOKEN.join([self.line_labels[i] for i in indices])

            if (
                (len(labels) <= self.output_dims[0] - 2)
                and (datum.height <= self.input_dims[1])
                and (datum.width <= self.input_dims[2])
            ):
                break
            indices = indices[:-1]

        if self.transform is not None:
            datum = self.transform(datum)

        length = self.output_dims[0]
        target = convert_strings_to_labels(strings=[labels], mapping=self.inverse_mapping, length=length)[0]

        return datum, target


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


if __name__ == "__main__":
    load_and_print_info(IAMSyntheticParagraphs)
