"""Module containing the PhysioNet/computing in cardidiology challenge 2012."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from collections.abc import Sequence
import logging

import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds

from .util import MedicalTsDatasetBuilder, MedicalTsDatasetInfo

_CITATION = """
@inproceedings{silva2012predicting,
  title={Predicting in-hospital mortality of icu patients:
         The physionet/computing in cardiology challenge 2012},
  author={Silva, Ikaro and Moody, George and Scott, Daniel J and Celi, Leo A
          and Mark, Roger G},
  booktitle={2012 Computing in Cardiology},
  pages={245--248},
  year={2012},
  organization={IEEE}
}
"""

_DESCRIPTION = """
The PhysioNet/Computing in Cardiology Challenge 2012.
"""


class Physionet2012DataReader(Sequence):
    """Reader class for physionet 2012 dataset."""

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

    def __init__(self, data_paths, endpoint_file):
        """Load instances from the Physionet 2012 challenge.

        Args:
            data_path: Path contiaing the patient records.
            endpoint_file: File containing the endpoint defentions for patient
                           records.

        """
        self.data_paths = data_paths
        self.endpoint_data = pd.read_csv(endpoint_file, header=0, sep=',')

    def _convert_string_to_decimal_time(self, values):
        return values.str.split(':').apply(
            lambda a: float(a[0]) + float(a[1])/60
        )

    def __getitem__(self, index):
        """Get instance at position index of endpoint file."""
        example_row = self.endpoint_data.iloc[index, :]

        # Extract targets and record id
        targets = example_row.to_dict()
        record_id = targets['RecordID']
        del targets['RecordID']

        # Read data
        statics, timeseries = self._read_file(str(record_id))
        time = timeseries['Time']
        values = timeseries[self.ts_features]

        return {
            'demographics': statics,
            'time': time,
            'vitals': values,
            'targets': targets,
            'metadata': {
                'patient_id': int(record_id)
            }
        }

    def _read_file(self, record_id):
        filename = None
        for path in self.data_paths:
            if tf.io.gfile.exists(path):
                filename = os.path.join(path, record_id + '.txt')
                break
        if filename is None:
            raise ValueError(f'Unable to find data for record: {record_id}.')

        with tf.io.gfile.GFile(filename, 'r') as f:
            data = pd.read_csv(f, sep=',', header=0)

        # Convert time to hours
        data['Time'] = self._convert_string_to_decimal_time(data['Time'])

        # Extract statics
        statics_indicator = data['Parameter'].isin(
            ['RecordID'] + self.static_features)
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
        assert str(int(statics['RecordID'])) == record_id
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
        """Return number of instances in the dataset."""
        return len(self.endpoint_data)


class Physionet2012(MedicalTsDatasetBuilder):
    """Dataset of the PhysioNet/Computing in Cardiology Challenge 2012."""

    VERSION = tfds.core.Version('1.0.0')
    has_demographics = True
    has_vitals = True
    has_lab_measurements = False
    has_interventions = False
    default_target = 'In-hospital_death'

    def _info(self):
        return MedicalTsDatasetInfo(
            builder=self,
            has_demographics=True,
            has_vitals=True,
            # TODO: Currently we treat all measurements as vitals. Should split
            # this.
            has_lab_measurements=False,
            has_interventions=False,
            targets={
                'In-hospital_death':
                    tfds.features.ClassLabel(num_classes=2),
                'SAPS-I':
                    tfds.features.Tensor(shape=tuple(), dtype=tf.float32),
                'SOFA':
                    tfds.features.Tensor(shape=tuple(), dtype=tf.float32),
                'Length_of_stay':
                    tfds.features.Tensor(shape=tuple(), dtype=tf.float32),
                'Survival':
                    tfds.features.Tensor(shape=tuple(), dtype=tf.float32)
            },
            default_target='In-hospital_death',
            demographics_names=Physionet2012DataReader.static_features,
            vitals_names=Physionet2012DataReader.ts_features,
            description=_DESCRIPTION,
            homepage='https://physionet.org/content/challenge-2012/1.0.0/',
            citation=_CITATION
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Return SplitGenerators."""
        paths = dl_manager.download_and_extract({
            'train-1-data': 'http://physionet.org/files/challenge-2012/1.0.0/set-a.tar.gz',  # noqa: E501
            'train-1-outcome': 'http://physionet.org/files/challenge-2012/1.0.0/Outcomes-a.txt?download',  # noqa: E501
            'train-2-data': 'http://physionet.org/files/challenge-2012/1.0.0/set-b.tar.gz',  # noqa: E501
            'train-2-outcome': 'http://physionet.org/files/challenge-2012/1.0.0/Outcomes-b.txt?download',  # noqa: E501
            'test-data': 'http://physionet.org/files/challenge-2012/1.0.0/set-c.tar.gz',  # noqa: E501
            'test-outcome': 'http://physionet.org/files/challenge-2012/1.0.0/Outcomes-c.txt?download'  # noqa: E501
            'train_listfile': '',
            'val_listfile': '',
            'test_listfile': ''
        })
        train_1_path = os.path.join(paths['train-1-data'], 'set-a')
        train_2_path = os.path.join(paths['train-2-data'], 'set-b')
        test_path = os.path.join(paths['test-data'], 'set-c')

        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={
                    'data_dirs': [train_1_path, train_2_path],
                    'outcome_file': paths['train_listfile']
                },
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.VALIDATION,
                gen_kwargs={
                    'data_dirs': [train_1_path, train_2_path],
                    'outcome_file': paths['val_listfile']
                },
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs={
                    'data_dirs': [test_path],
                    'outcome_file': paths['test_listfile']
                }
            )
        ]

    def _generate_examples(self, data_dirs, outcome_file):
        """Yield examples."""
        index = 0
        for data_dir, outcome_file in zip(data_dirs, outcome_files):
            reader = Physionet2012DataReader(data_dir, outcome_file)
            for instance in reader:
                yield index, instance
                index += 1
