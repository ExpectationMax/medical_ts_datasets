"""Test the physionet 2012 dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow_datasets import testing
from medical_ts_datasets.physionet_2019 import Physionet2019

from .utils import FAKE_DATA_DIR


class Physionet2019Test(testing.DatasetBuilderTestCase):
    """Test Physionet 2012 dataset."""

    DATASET_CLASS = Physionet2019
    EXAMPLE_DIR = os.path.join(FAKE_DATA_DIR, 'physionet_2019')
    SPLITS = {
        "train": 2,
        # "test": 2
    }
    DL_EXTRACT_RESULT = {
        'train-1': '',
        'train-2': '',
    }


if __name__ == "__main__":
    testing.test_main()
