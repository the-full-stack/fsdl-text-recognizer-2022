"""IAM Synthetic Paragraphs Dataset class."""
import argparse
import random
from typing import Any, Callable, List, Sequence, Tuple

# from boltons.cacheutils import cachedproperty
import numpy as np
from PIL import Image
import torch
from pytorch_lightning.utilities.rank_zero import rank_zero_info

from text_recognizer.data.base_data_module import load_and_print_info
from text_recognizer.data.iam import IAM
from text_recognizer.data.iam_lines import (
    line_crops_and_labels,
    load_line_crops,
    load_line_labels,
    save_images_and_labels,
)
from text_recognizer.data.iam_paragraphs import (
    get_dataset_properties,
    IAMParagraphs,
)
from text_recognizer.data.util import BaseDataset, convert_strings_to_labels, resize_image
import text_recognizer.metadata.iam_synthetic_paragraphs as metadata


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

        for split in ["train"]:  # synthetic dataset is only used in training phase
            rank_zero_info(f"Cropping IAM line regions and loading labels for {split} data split...")
            crops, labels = line_crops_and_labels(iam, split)

            crops = [resize_image(crop, IMAGE_SCALE_FACTOR) for crop in crops]
            save_images_and_labels(crops, labels, split, PROCESSED_DATA_DIRNAME)

    def setup(self, stage: str = None) -> None:
        rank_zero_info(f"IAMSyntheticParagraphs.setup({stage}): Loading train IAM paragraph regions and lines...")

        if stage == "fit" or stage is None:
            self.data_train = IAMSyntheticParagraphsDataset(
                dataset_len=self.batch_size * max(self.num_gpus, 1) * 10,
                inverse_mapping=self.inverse_mapping,
                target_length=self.output_dims[0],
                transform=self.trainval_transform,
            )

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

    def __init__(self, dataset_len: int, inverse_mapping: dict, target_length: int, transform: Callable = None) -> None:
        super().__init__()
        self.line_crops = load_line_crops("train", PROCESSED_DATA_DIRNAME)  # should these be passed externally?
        self.line_labels = load_line_labels("train", PROCESSED_DATA_DIRNAME)  # should these be passed externally?
        assert len(self.line_crops) == len(self.line_labels)

        self.ids = list(range(len(self.line_labels)))
        self.dataset_len = dataset_len
        self.inverse_mapping = inverse_mapping
        self.target_length = target_length
        self.transform = transform
        self.min_num_lines, self.max_num_lines = 1, 12

        # each worker will have its PyTorch seed set to base_seed + worker_id
        worker_info = torch.utils.data.get_worker_info()
        print("IAMSyntheticParagraphsDataset.__init__():worker_info", worker_info)
        if worker_info is not None:
            print(f"Setting seed to {worker_info.seed} for worker {worker_info}")
            random.seed(worker_info.seed)
        print("IAMSyntheticParagraphsDataset.__init__():self.dataset_len", self.dataset_len)

    def __len__(self) -> int:
        """Return length of the dataset."""
        print("IAMSyntheticParagraphsDataset.__len__():self.dataset_len", self.dataset_len)
        return self.dataset_len

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Return a datum and its target, after processing by transforms."""
        # set seed
        num_lines = random.randint(self.min_num_lines, self.max_num_lines)
        indices = random.sample(self.ids, k=num_lines)
        print(f"IAMSyntheticParagraphsDataset.__getitem__({index}):indices: {indices}")
        datum = join_line_crops_to_form_paragraph([self.line_crops[i] for i in indices])

        labels = NEW_LINE_TOKEN.join([self.line_labels[i] for i in indices])
        target = convert_strings_to_labels(strings=[labels], mapping=self.inverse_mapping, length=self.target_length)[0]

        # if len(target) > paragraph_properties["label_length"]["max"]:
        #     print("Label longer than longest label in original IAM Paragraphs dataset - hence dropping")
        #     continue
        # max_para_shape = paragraph_properties["crop_shape"]["max"]
        # if datum.height > max_para_shape[0] or datum.width > max_para_shape[1]:
        #     print("Crop larger than largest crop in original IAM Paragraphs dataset - hence dropping")
        #     continue

        if self.transform is not None:
            datum = self.transform(datum)

        return datum, target


# def generate_synthetic_paragraphs(
#     line_crops: List[Image.Image], line_labels: List[str], max_batch_size: int = 12
# ) -> Tuple[List[Image.Image], List[str]]:
#     """
#     Generate synthetic paragraphs and corresponding labels by randomly joining different subsets of crops.
#     These synthetic paragraphs are generated such that the number of paragraphs with 1 line of text is greater
#     than the number of paragraphs with 2 lines of text is greater than the number of paragraphs with 3 lines of text
#     and so on.
#     """
#     paragraph_properties = get_dataset_properties()

#     indices = list(range(len(line_labels)))
#     assert max_batch_size < paragraph_properties["num_lines"]["max"]

#     batched_indices_list = [[_] for _ in indices]  # batch_size = 1, len = 9462
#     batched_indices_list.extend(
#         generate_random_batches(values=indices, min_batch_size=2, max_batch_size=(1 * max_batch_size) // 4)
#     )
#     batched_indices_list.extend(
#         generate_random_batches(values=indices, min_batch_size=2, max_batch_size=(2 * max_batch_size) // 4)
#     )
#     batched_indices_list.extend(
#         generate_random_batches(values=indices, min_batch_size=2, max_batch_size=(3 * max_batch_size) // 4)
#     )
#     batched_indices_list.extend(
#         generate_random_batches(values=indices, min_batch_size=2, max_batch_size=max_batch_size)
#     )
#     batched_indices_list.extend(
#         generate_random_batches(
#             values=indices, min_batch_size=(2 * max_batch_size) // 4 + 1, max_batch_size=max_batch_size
#         )
#     )
#     batched_indices_list.extend(
#         generate_random_batches(
#             values=indices, min_batch_size=(3 * max_batch_size) // 4 + 1, max_batch_size=max_batch_size
#         )
#     )
#     # assert sorted(list(itertools.chain(*batched_indices_list))) == indices

#     unique, counts = np.unique([len(_) for _ in batched_indices_list], return_counts=True)
#     for batch_len, count in zip(unique, counts):
#         rank_zero_info(f"{count} samples with {batch_len} lines")

#     para_crops, para_labels = [], []
#     for para_indices in batched_indices_list:
#         para_label = NEW_LINE_TOKEN.join([line_labels[i] for i in para_indices])
#         if len(para_label) > paragraph_properties["label_length"]["max"]:
#             print("Label longer than longest label in original IAM Paragraphs dataset - hence dropping")
#             continue

#         para_crop = join_line_crops_to_form_paragraph([line_crops[i] for i in para_indices])
#         max_para_shape = paragraph_properties["crop_shape"]["max"]
#         if para_crop.height > max_para_shape[0] or para_crop.width > max_para_shape[1]:
#             print("Crop larger than largest crop in original IAM Paragraphs dataset - hence dropping")
#             continue

#         para_crops.append(para_crop)
#         para_labels.append(para_label)

#     assert len(para_crops) == len(para_labels)
#     return para_crops, para_labels


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


# def generate_random_batches(values: List[Any], min_batch_size: int, max_batch_size: int) -> List[List[Any]]:
#     """
#     Generate random batches of elements in values without replacement and return the list of all batches. Batch sizes
#     can be anything between min_batch_size and max_batch_size including the end points.
#     """
#     shuffled_values = values.copy()
#     random.shuffle(shuffled_values)

#     start_id = 0
#     grouped_values_list = []
#     while start_id < len(shuffled_values):
#         num_values = random.randint(min_batch_size, max_batch_size)
#         grouped_values_list.append(shuffled_values[start_id : start_id + num_values])
#         start_id += num_values
#     assert sum([len(_) for _ in grouped_values_list]) == len(values)
#     return grouped_values_list


if __name__ == "__main__":
    load_and_print_info(IAMSyntheticParagraphs)
