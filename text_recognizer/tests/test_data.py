"""Test submodules of the data module."""
import os
import shutil

import numpy as np
import pytest

from text_recognizer.data import emnist


@pytest.mark.data
class TestDataset:
    """Tests downloading and setup of a dataset."""


emnist_dirs = [emnist.PROCESSED_DATA_DIRNAME, emnist.DL_DATA_DIRNAME]


@pytest.fixture(scope="module")
def emnist_dataset():
    _remove_if_exist(emnist_dirs)
    dataset = emnist.EMNIST()
    dataset.prepare_data()
    return dataset


def _exist(dirs):
    return all(os.path.exists(dir) for dir in dirs)


def _remove_if_exist(dirs):
    for dir in dirs:
        shutil.rmtree(dir, ignore_errors=True)


class TestEMNIST(TestDataset):
    """Tests downloading and properties of the EMNIST dataset."""

    dirs = emnist_dirs

    def test_prepare_data(self, emnist_dataset):
        """Tests whether the prepare_data method has produced the expected directories."""
        assert _exist(self.dirs)

    def test_setup(self, emnist_dataset):
        """Tests features of the fully set up dataset."""
        dataset = emnist_dataset
        dataset.setup()
        assert all(map(lambda s: hasattr(dataset, s), ["x_trainval", "y_trainval", "x_test", "y_test"]))
        splits = [dataset.x_trainval, dataset.y_trainval, dataset.x_test, dataset.y_test]
        assert all(map(lambda attr: type(attr) == np.ndarray, splits))
        observed_train_frac = len(dataset.data_train) / (len(dataset.data_train) + len(dataset.data_val))
        assert np.isclose(observed_train_frac, emnist.TRAIN_FRAC)
        assert dataset.input_dims[-2:] == dataset.x_trainval[0].shape  # ToTensor() adds a dimension
        assert len(dataset.output_dims) == len(dataset.y_trainval.shape)  # == 1
