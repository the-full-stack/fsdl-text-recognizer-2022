"""Module containing submodules for each dataset.

Each dataset is defined as a class in that submodule.

The datasets should have a .config method that returns
any configuration information needed by the model.

Most datasets define their constants in a submodule
of the metadata module that is parallel to this one in the
hierarchy.
"""
from .util import BaseDataset
from .base_data_module import BaseDataModule
from .mnist import MNIST

# Hide lines below until Lab 02
from .emnist import EMNIST
from .emnist_lines import EMNISTLines

# Hide lines above until Lab 02

# Hide lines below until Lab 03
from .iam_paragraphs import IAMParagraphs

# Hide lines above until Lab 03

# Hide lines below until Lab 04
from .iam_lines import IAMLines

# Hide lines above until Lab 04

# Hide lines below until Lab 05
from .fake_images import FakeImageData

# Hide lines above until Lab 05

# Hide lines below until Lab 06
from .iam_synthetic_paragraphs import IAMSyntheticParagraphs
from .iam_original_and_synthetic_paragraphs import IAMOriginalAndSyntheticParagraphs

# Hide lines above until Lab 06
