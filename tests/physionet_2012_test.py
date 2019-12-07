"""Test the physionet 2012 dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow_datasets import testing
from medical_ts_datasets.physionet_2012 import Physionet2012

from .utils import FAKE_DATA_DIR


class Physionet2012Test(testing.DatasetBuilderTestCase):
    """Test Physionet 2012 dataset."""

    DATASET_CLASS = Physionet2012
    EXAMPLE_DIR = os.path.join(FAKE_DATA_DIR, 'physionet_2012')
    SPLITS = {
        "train": 4,
        "test": 2
    }
    DL_EXTRACT_RESULT = {
        'train-1-data': '',
        'train-1-outcome': 'Outcomes-a.txt',
        'train-2-data': '',
        'train-2-outcome': 'Outcomes-b.txt',
        'test-data': '',
        'test-outcome': 'Outcomes-c.txt'
    }


if __name__ == "__main__":
    testing.test_main()
