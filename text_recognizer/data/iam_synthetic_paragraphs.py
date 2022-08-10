"""IAM Synthetic Paragraphs Dataset class."""
import argparse

from boltons.cacheutils import cachedproperty
from pytorch_lightning.utilities.rank_zero import rank_zero_info

from text_recognizer.data.base_data_module import load_and_print_info
from text_recognizer.data.iam import IAM
from text_recognizer.data.iam_lines import (
    generate_line_crops_and_labels,
    load_processed_line_crops,
    load_processed_line_labels,
    save_images_and_labels,
)
from text_recognizer.data.iam_paragraphs import IAMParagraphs
from text_recognizer.data.paragraph_synthesis import generate_synthetic_paragraphs
from text_recognizer.data.util import BaseDataset, convert_strings_to_labels
import text_recognizer.metadata.iam_synthetic_paragraphs as metadata


PROCESSED_DATA_DIRNAME = metadata.PROCESSED_DATA_DIRNAME


class IAMSyntheticParagraphs(IAMParagraphs):
    """IAM Handwriting database synthetic paragraphs."""

    def __init__(self, args: argparse.Namespace = None):
        super().__init__(args)

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

    def setup(self, stage: str = None):
        rank_zero_info(f"IAMSyntheticParagraphs.setup({stage}): Loading trainval IAM paragraph regions and lines...")

        if stage == "fit" or stage is None:
            self.data_train = self._synthesize_dataset()

    def _synthesize_dataset(self):
        X, para_labels = generate_synthetic_paragraphs(line_crops=self.line_crops, line_labels=self.line_labels)
        Y = convert_strings_to_labels(strings=para_labels, mapping=self.inverse_mapping, length=self.output_dims[0])
        return BaseDataset(X, Y, transform=self.trainval_transform)

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

    @cachedproperty
    def line_crops(self):
        return load_processed_line_crops("train", PROCESSED_DATA_DIRNAME)

    @cachedproperty
    def line_labels(self):
        return load_processed_line_labels("train", PROCESSED_DATA_DIRNAME)


if __name__ == "__main__":
    load_and_print_info(IAMSyntheticParagraphs)
