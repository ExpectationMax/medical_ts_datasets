"""Module containing the PhysioNet/computing in cardidiology challenge 2019."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from collections.abc import Sequence

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds

_CITATION = """
@inproceedings{silva2012predicting,
  title={Predicting in-hospital mortality of icu patients: The physionet/computing in cardiology challenge 2012},
  author={Silva, Ikaro and Moody, George and Scott, Daniel J and Celi, Leo A and Mark, Roger G},
  booktitle={2012 Computing in Cardiology},
  pages={245--248},
  year={2012},
  organization={IEEE}
}
"""

_DESCRIPTION = """
The PhysioNet Computing in Cardiology Challenge 2012.
"""


class Physionet2019DataReader(Sequence):
    """Reader class for physionet 2019 dataset."""
    static_features = ['Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime']
    vital_features = [
        'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2']
    lab_features = [
        'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
        'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine',
        'Bilirubin_direct', 'Glucose', 'Lactate', 'Magnesium', 'Phosphate',
        'Potassium', 'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT',
        'WBC', 'Fibrinogen', 'Platelets'
    ]
    ts_features = vital_features + lab_features

    def __init__(self, data_path):
        self.data_path = data_path
        self.samples = tf.io.gfile.listdir(data_path)
        self.label_dtype = np.float32
        self.data_dtype = np.float32

    def __getitem__(self, index):
        sample_name = self.samples[index]
        filename = os.path.join(self.data_path, sample_name)
        with tf.io.gfile.GFile(filename, 'r') as f:
            data = pd.read_csv(f, sep='|', header=0)

        record_id = int(sample_name[1:].split('.')[0])
        time = data['ICULOS']
        sepsis_label = data['SepsisLabel']
        statics = data[self.static_features].iloc[0].values
        data = data[self.vital_features + self.lab_features]

        return {
            'statics': statics.astype(self.data_dtype),
            'time': time.astype(self.data_dtype),
            'values': data.astype(self.data_dtype),
            'targets': {'Sepsis': sepsis_label.astype(self.data_dtype)},
            'metadata': {
                'RecordID': record_id
            }
        }

    def __len__(self):
        return len(self.samples)


class Physionet2019(tfds.core.GeneratorBasedBuilder):
    """Dataset for the PhysioNet 2012 Computing in Cardiology Challenge 2012."""

    VERSION = tfds.core.Version('1.0.0')

    def _info(self):
        n_statics = len(Physionet2019DataReader.static_features)
        n_ts = len(Physionet2019DataReader.ts_features)
        return tfds.core.DatasetInfo(
            builder=self,
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # tfds.features.FeatureConnectors
            features=tfds.features.FeaturesDict({
                'statics': tfds.features.Tensor(shape=(n_statics,), dtype=tf.float32),
                'time': tfds.features.Tensor(shape=(None,), dtype=tf.float32),
                'values': tfds.features.Tensor(shape=(None, n_ts), dtype=tf.float32),
                'targets': {
                    'Sepsis': tfds.features.Tensor(shape=(None,), dtype=tf.float32)
                },
                'metadata': {
                    'RecordID': tf.uint32
                }
            }),
            # Homepage of the dataset for documentation
            homepage='https://physionet.org/content/challenge-2019/1.0.0/',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Return SplitGenerators."""
        paths = dl_manager.download_and_extract({
            'train-1': 'https://archive.physionet.org/users/shared/challenge-2019/training_setA.zip',
            'train-2': 'https://archive.physionet.org/users/shared/challenge-2019/training_setB.zip'
        })
        train_1_path = os.path.join(paths['train-1'], 'training')
        train_2_path = os.path.join(paths['train-2'], 'training_setB')

        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={
                    'data_dirs': [train_1_path, train_2_path]
                }
            )
        ]

    def _generate_examples(self, data_dirs):
        """Yield example."""
        index = 0
        for data_dir in data_dirs:
            reader = Physionet2019DataReader(data_dir)
            for instance in reader:
                yield index, instance
                index += 1
