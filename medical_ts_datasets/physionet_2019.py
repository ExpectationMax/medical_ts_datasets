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

from .util import MedicalTsDatasetBuilder, MedicalTsDatasetInfo

RESOURCES = os.path.join(
    os.path.dirname(__file__), 'resources', 'physionet2019')

_CITATION = """
@article{reyna2019early,
  title={Early prediction of sepsis from clinical data:
         the PhysioNet/Computing in Cardiology Challenge 2019},
  author={Reyna, M and Josef, C and Jeter, R and Shashikumar, S and Westover, M
          and Nemati, S and Clifford, G and Sharma, A},
  journal={Critical Care Medicine},
  year={2019}
}
"""

_DESCRIPTION = """
The PhysioNet Computing in Cardiology Challenge 2019.
"""


class Physionet2019DataReader(Sequence):
    """Reader class for physionet 2019 dataset."""

    static_features = ['Age', 'Gender', 'HospAdmTime']
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
    categorical_demographics = {
        'Gender': [0, 1],
    }
    expanded_static_features = [
        'Age', 'Gender=0', 'Gender=1', 'HospAdmTime'
    ]

    def __init__(self, data_paths, listfile):
        """Load instances of PhysioNet 2019 challenge from data_path."""
        self.data_paths = data_paths

        # Ensure same order of instances
        with tf.io.gfile.GFile(listfile, 'r') as f:
            self.samples = pd.read_csv(f, header=0, sep=',')
        # self.samples = sorted(tf.io.gfile.listdir(data_path))
        self.label_dtype = np.float32
        self.data_dtype = np.float32

    def __getitem__(self, index):
        """Get instance at position index."""
        sample_name = self.samples.iloc[index].at['filename']
        filename = None
        for path in self.data_paths:
            suggested_filename = os.path.join(path, sample_name)
            if tf.io.gfile.exists(suggested_filename):
                filename = suggested_filename
                break
        if filename is None:
            raise ValueError(f'Unable to find data for record: {sample_name}.')

        with tf.io.gfile.GFile(filename, 'r') as f:
            data = pd.read_csv(f, sep='|', header=0)

        record_id = int(sample_name[1:].split('.')[0])
        time = data['ICULOS']
        sepsis_label = data['SepsisLabel']
        statics = data[self.static_features].iloc[0]

        # Do one hot encoding for categorical features
        for demo, values in self.categorical_demographics.items():
            cur_demo = statics[demo]
            # Transform categorical values into zero based index
            to_replace = {val: values.index(val) for val in values}
            # Ensure we dont have unexpected values
            if cur_demo in to_replace.keys():
                indicators = to_replace[cur_demo]
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

        vitals = data[self.vital_features]
        labs = data[self.lab_features]

        return sample_name, {
            'demographics': statics.values.astype(self.data_dtype),
            'time': time.values.astype(self.data_dtype),
            'vitals': vitals.values.astype(self.data_dtype),
            'lab_measurements': labs.values.astype(self.data_dtype),
            'targets': {
                'Sepsis':
                    sepsis_label.values.astype(np.int32)[:, np.newaxis]
            },
            'metadata': {
                'patient_id': record_id
            }
        }

    def __len__(self):
        """Return number of instances in the dataset."""
        return len(self.samples)


class Physionet2019(MedicalTsDatasetBuilder):
    """Dataset of the PhysioNet/Computing in Cardiology Challenge 2019."""

    VERSION = tfds.core.Version('1.0.3')

    has_demographics = True
    has_vitals = True
    has_lab_measurements = True
    has_interventions = False
    default_target = 'Sepsis'

    def _info(self):
        return MedicalTsDatasetInfo(
            builder=self,
            targets={
                'Sepsis': tfds.features.Tensor(
                    shape=(None, 1), dtype=tf.int32)
            },
            default_target='Sepsis',
            demographics_names=Physionet2019DataReader.expanded_static_features,
            vitals_names=Physionet2019DataReader.vital_features,
            lab_measurements_names=Physionet2019DataReader.lab_features,
            description=_DESCRIPTION,
            homepage='https://physionet.org/content/challenge-2019/1.0.0/',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Return SplitGenerators."""
        paths = dl_manager.download_and_extract({
            'train-1': 'https://archive.physionet.org/users/shared/challenge-2019/training_setA.zip',  # noqa: E501
            'train-2': 'https://archive.physionet.org/users/shared/challenge-2019/training_setB.zip'  # noqa: E501
        })
        train_1_path = os.path.join(paths['train-1'], 'training')
        train_2_path = os.path.join(paths['train-2'], 'training_setB')

        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={
                    'data_paths': [train_1_path, train_2_path],
                    'listfile': os.path.join(RESOURCES, 'train_listfile.csv')
                }
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.VALIDATION,
                gen_kwargs={
                    'data_paths': [train_1_path, train_2_path],
                    'listfile': os.path.join(RESOURCES, 'val_listfile.csv')
                }
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs={
                    'data_paths': [train_1_path, train_2_path],
                    'listfile': os.path.join(RESOURCES, 'test_listfile.csv')
                }
            )
        ]

    def _generate_examples(self, data_paths, listfile):
        """Yield example."""
        reader = Physionet2019DataReader(data_paths, listfile)
        for sample_id, instance in reader:
            yield sample_id, instance
