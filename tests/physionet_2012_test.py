"""TODO(PhysioNet_2012): Add a description here."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow_datasets import testing
from medical_ts_datasets.physionet_2012 import Physionet_2012

from .utils import FAKE_DATA_DIR


class Physionet2012Test(testing.DatasetBuilderTestCase):
    DATASET_CLASS = Physionet_2012
    EXAMPLE_DIR = os.path.join(FAKE_DATA_DIR, 'physionet_2012')
    SPLITS = {
        "train": 4,  # Number of fake train example
        "test": 2,  # Number of fake test example
    }

    # If you are calling `download/download_and_extract` with a dict, like:
    #   dl_manager.download({'some_key': 'http://a.org/out.txt', ...})
    # then the tests needs to provide the fake output paths relative to the
    # fake data directory
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

