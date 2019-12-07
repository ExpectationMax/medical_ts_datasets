"""TODO(PhysioNet_2012): Add a description here."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from collections.abc import Sequence
import logging

import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds

# TODO(PhysioNet_2012): BibTeX citation
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

# TODO(PhysioNet_2012):
_DESCRIPTION = """
The PhysioNet Computing in Cardiology Challenge 2012.
"""


class Physionet2012DataReader(Sequence):
    static_features = [
        'Age', 'Gender', 'Height', 'ICUType', 'Weight'
    ]
    ts_features = [
        'ALP', 'ALT', 'AST', 'Albumin', 'BUN', 'Bilirubin',
        'Cholesterol', 'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose',
        'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'MAP', 'MechVent', 'Mg',
        'NIDiasABP', 'NIMAP', 'NISysABP', 'Na', 'PaCO2', 'PaO2', 'Platelets',
        'RespRate', 'SaO2', 'SysABP', 'Temp', 'TroponinI', 'TroponinT',
        'Urine', 'WBC', 'pH'
    ]

    def __init__(self, data_path, endpoint_file):
        self.data_path = data_path
        self.endpoint_data = pd.read_csv(endpoint_file, header=0, sep=',')

    def convert_string_to_decimal_time(self, values):
        return values.str.split(':').apply(
            lambda a: float(a[0]) + float(a[1])/60
        )

    def __getitem__(self, index):
        example_row = self.endpoint_data.iloc[index, :]

        # Extract targets and record id
        targets = example_row.to_dict()
        record_id = targets['RecordID']
        del targets['RecordID']

        # Read data
        statics, timeseries = self.read_file(str(record_id))
        time = timeseries['Time']
        values = timeseries[self.ts_features]

        return {
            'statics': statics,
            'time': time,
            'values': values,
            'targets': targets,
            'metadata': {
                'RecordID': int(record_id)
            }
        }

    def read_file(self, record_id):
        filename = os.path.join(self.data_path, record_id + '.txt')
        with tf.io.gfile.GFile(filename, 'r') as f:
            data = pd.read_csv(f, sep=',', header=0)

        # Convert time to hours
        data['Time'] = self.convert_string_to_decimal_time(data['Time'])

        # Extract statics
        statics_indicator = data['Parameter'].isin(['RecordID'] + self.static_features)
        statics = data[statics_indicator]
        data = data[~statics_indicator]

        # Handle duplicates in statics
        duplicated_statics = statics[['Time', 'Parameter']].duplicated()
        if duplicated_statics.sum() > 0:
            logging.warning('Got duplicated statics: %s', statics)
            # Average over duplicate measurements
            statics = statics.groupby(['Time', 'Parameter'], as_index=False)\
                .mean().reset_index()
        statics = statics.pivot(
            index='Time', columns='Parameter', values='Value')
        statics = statics.reindex().reset_index()
        statics = statics.iloc[0]

        # Be sure we are loading the correct record
        assert str(int(statics['RecordID'])) != record_id
        # Drop RecordID
        statics = statics[self.static_features]

        # Sometimes the same value is observered twice for the same time,
        # potentially using different devices. In this case take the mean of
        # the observed values.
        duplicated_ts = data[['Time', 'Parameter']].duplicated()
        if duplicated_ts.sum() > 0:
            logging.debug(
                'Got duplicated time series variables for RecordID=%s',
                record_id
            )
            data = data.groupby(['Time', 'Parameter'], as_index=False)\
                .mean().reset_index()

        time_series = data.pivot(
            index='Time', columns='Parameter', values='Value')
        time_series = time_series\
            .reindex(columns=self.ts_features).reset_index()
        return statics, time_series

    def __len__(self):
        return len(self.endpoint_data)


class Physionet_2012(tfds.core.GeneratorBasedBuilder):
    """Dataset for the PhysioNet 2012 Computing in Cardiology Challenge 2012."""

    VERSION = tfds.core.Version('1.0.0')

    def _info(self):
        n_statics = len(Physionet2012DataReader.static_features)
        n_ts = len(Physionet2012DataReader.ts_features)
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
                    'In-hospital_death': tfds.features.ClassLabel(num_classes=2),
                    'SAPS-I': tfds.features.Tensor(shape=tuple(), dtype=tf.float32),
                    'SOFA': tfds.features.Tensor(shape=tuple(), dtype=tf.float32),
                    'Length_of_stay': tfds.features.Tensor(shape=tuple(), dtype=tf.float32),
                    'Survival': tfds.features.Tensor(shape=tuple(), dtype=tf.float32)
                },
                'metadata': {
                    'RecordID': tf.uint32
                }
            }),
            # Homepage of the dataset for documentation
            homepage='https://physionet.org/content/challenge-2012/1.0.0/',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        paths = dl_manager.download_and_extract({
            'train-1-data': 'http://physionet.org/files/challenge-2012/1.0.0/set-a.tar.gz',
            'train-1-outcome': 'http://physionet.org/files/challenge-2012/1.0.0/Outcomes-a.txt?download',
            'train-2-data': 'http://physionet.org/files/challenge-2012/1.0.0/set-b.tar.gz',
            'train-2-outcome': 'http://physionet.org/files/challenge-2012/1.0.0/Outcomes-b.txt?download',
            'test-data': 'http://physionet.org/files/challenge-2012/1.0.0/set-c.tar.gz',
            'test-outcome': 'http://physionet.org/files/challenge-2012/1.0.0/Outcomes-c.txt?download'
        })
        train_1_path = os.path.join(paths['train-1-data'], 'set-a')
        train_2_path = os.path.join(paths['train-2-data'], 'set-b')
        test_path = os.path.join(paths['test-data'], 'set-c')

        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={
                    'data_dirs': [train_1_path, train_2_path],
                    'outcome_files': [paths['train-1-outcome'], paths['train-2-outcome']]
                },
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs={
                    'data_dirs': [test_path],
                    'outcome_files': [paths['test-outcome']]
                }
            )
        ]

    def _generate_examples(self, data_dirs, outcome_files):
        """Yields examples."""
        index = 0
        for data_dir, outcome_file in zip(data_dirs, outcome_files):
            reader = Physionet2012DataReader(data_dir, outcome_file)
            for instance in reader:
                yield index, instance
                index += 1

