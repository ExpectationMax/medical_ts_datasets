"""Adds the checksum directory to the search path of tensorflow datasets."""
import os
import tensorflow_datasets
tensorflow_datasets.download.add_checksums_dir(
    os.path.join(os.path.dirname(__file__), 'checksums'))
