"""Module containing the PhysioNet/computing in cardidiology challenge 2012."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from collections.abc import Sequence
import logging

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds

from .util import MedicalTsDatasetBuilder, MedicalTsDatasetInfo

RESOURCES = os.path.join(
    os.path.dirname(__file__), 'resources', 'physionet2012')

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
        'Age', 'Gender', 'Height', 'ICUType'
    ]
    ts_features = [
        'Weight', 'ALP', 'ALT', 'AST', 'Albumin', 'BUN', 'Bilirubin',
        'Cholesterol', 'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose',
        'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'MAP', 'MechVent', 'Mg',
        'NIDiasABP', 'NIMAP', 'NISysABP', 'Na', 'PaCO2', 'PaO2', 'Platelets',
        'RespRate', 'SaO2', 'SysABP', 'Temp', 'TroponinI', 'TroponinT',
        'Urine', 'WBC', 'pH'
    ]
    categorical_demographics = {
        'Gender': [0, 1],
        'ICUType': [1, 2, 3, 4]
    }
    expanded_static_features = [
        'Age', 'Gender=0', 'Gender=1', 'Height', 'ICUType=1', 'ICUType=2',
        'ICUType=3', 'ICUType=4'
    ]

    # Remove instances without any timeseries
    blacklist = [
        140501, 150649, 140936, 143656, 141264, 145611, 142998, 147514, 142731,
        150309, 155655, 156254
    ]

    def __init__(self, data_paths, endpoint_file):
        """Load instances from the Physionet 2012 challenge.

        Args:
            data_path: Path contiaing the patient records.
            endpoint_file: File containing the endpoint defentions for patient
                           records.

        """
        self.data_paths = data_paths
        endpoint_data = pd.read_csv(endpoint_file, header=0, sep=',')
        self.endpoint_data = endpoint_data[
            ~endpoint_data['RecordID'].isin(self.blacklist)]

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

        return record_id, {
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
            suggested_filename = os.path.join(path, record_id + '.txt')
            if tf.io.gfile.exists(suggested_filename):
                filename = suggested_filename
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

        # Do one hot encoding for categorical features
        for demo, values in self.categorical_demographics.items():
            cur_demo = statics[demo]
            # Transform categorical values into zero based index
            to_replace = {val: values.index(val) for val in values}
            # Ensure we dont have unexpected values
            if cur_demo in to_replace.keys():
                indicators = to_replace[cur_demo] #.replace(to_replace).values
                one_hot_encoded = np.eye(len(values))[indicators]
            else:
                # We have a few cases where the categorical variables are not
                # available. Then we should just return zeros for all
                # categories.
                one_hot_encoded = np.zeros(len(to_replace.values()))
            statics.drop(columns=demo, inplace=True)
            columns = [f'{demo}={val}' for val in values]
            statics = pd.concat([statics, pd.Series(one_hot_encoded, index=columns)])

        # Ensure same order
        statics = statics[self.expanded_static_features]

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
            .reindex(columns=self.ts_features).dropna(how='all').reset_index()
        return statics, time_series

    def __len__(self):
        """Return number of instances in the dataset."""
        return len(self.endpoint_data)


class Physionet2012(MedicalTsDatasetBuilder):
    """Dataset of the PhysioNet/Computing in Cardiology Challenge 2012."""

    VERSION = tfds.core.Version('1.0.10')
    has_demographics = True
    has_vitals = True
    has_lab_measurements = False
    has_interventions = False
    default_target = 'In-hospital_death'

    def _info(self):
        return MedicalTsDatasetInfo(
            builder=self,
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
            # TODO: Currently we treat all measurements as vitals. Should split
            # this.
            demographics_names=Physionet2012DataReader.expanded_static_features,
            vitals_names=Physionet2012DataReader.ts_features,
            description=_DESCRIPTION,
            homepage='https://physionet.org/content/challenge-2012/1.0.0/',
            citation=_CITATION
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Return SplitGenerators."""
        paths = dl_manager.download_and_extract({
            'set-a': 'http://physionet.org/files/challenge-2012/1.0.0/set-a.tar.gz',  # noqa: E501
            # 'train-1-outcome': 'http://physionet.org/files/challenge-2012/1.0.0/Outcomes-a.txt?download',  # noqa: E501
            'set-b': 'http://physionet.org/files/challenge-2012/1.0.0/set-b.tar.gz',  # noqa: E501
            # 'train-2-outcome': 'http://physionet.org/files/challenge-2012/1.0.0/Outcomes-b.txt?download',  # noqa: E501
            'set-c': 'http://physionet.org/files/challenge-2012/1.0.0/set-c.tar.gz',  # noqa: E501
            # 'test-outcome': 'http://physionet.org/files/challenge-2012/1.0.0/Outcomes-c.txt?download',  # noqa: E501
        })
        a_path = os.path.join(paths['set-a'], 'set-a')
        b_path = os.path.join(paths['set-b'], 'set-b')
        c_path = os.path.join(paths['set-c'], 'set-c')

        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={
                    'data_dirs': [a_path, b_path, c_path],
                    'outcome_file': os.path.join(RESOURCES, 'train_listfile.csv')
                },
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.VALIDATION,
                gen_kwargs={
                    'data_dirs': [a_path, b_path, c_path],
                    'outcome_file': os.path.join(RESOURCES, 'val_listfile.csv')
                },
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs={
                    'data_dirs': [a_path, b_path, c_path],
                    'outcome_file': os.path.join(RESOURCES, 'test_listfile.csv')
                }
            )
        ]

    def _generate_examples(self, data_dirs, outcome_file):
        """Yield examples."""
        reader = Physionet2012DataReader(data_dirs, outcome_file)
        for instance in reader:
            yield instance
