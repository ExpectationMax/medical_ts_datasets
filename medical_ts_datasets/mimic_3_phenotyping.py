"""Module containing phenotyping dataset of MIMIC-III benchmarks."""

from collections.abc import Sequence
from os.path import join

import numpy as np
import pandas as pd
import tensorflow_datasets as tfds
import tensorflow as tf

from .util import MedicalTsDatasetBuilder, MedicalTsDatasetInfo

_CITATION = """
@article{Harutyunyan2019,
  author={Harutyunyan, Hrayr and Khachatrian, Hrant and Kale, David C.
          and Ver Steeg, Greg and Galstyan, Aram},
  title={Multitask learning and benchmarking with clinical time series data},
  journal={Scientific Data},
  year={2019},
  volume={6},
  number={1},
  pages={96},
  issn={2052-4463},
  doi={10.1038/s41597-019-0103-9},
  url={https://doi.org/10.1038/s41597-019-0103-9}
}
@article{Johnson2016,
  title={MIMIC-III, a freely accessible critical care database},
  author={Johnson, Alistair EW and Pollard, Tom J and Shen, Lu and
          Li-wei, H Lehman and Feng, Mengling and Ghassemi, Mohammad and
          Moody, Benjamin and Szolovits, Peter and Celi, Leo Anthony and
          Mark, Roger G},
  journal={Scientific data},
  volume={3},
  pages={160035},
  year={2016},
  publisher={Nature Publishing Group}
}
"""

_DESCRIPTION = """
Phenotyping dataset of the MIMIC-III benchmarks.
"""


class MIMICPhenotypingReader(Sequence):
    """Reader for phenotyping dataset of the MIMIC-III benchmarks."""

    demographics = ['Height', 'Weight']
    vitals = [
        'Heart Rate', 'Mean blood pressure', 'Diastolic blood pressure',
        'Systolic blood pressure', 'Oxygen saturation', 'Respiratory rate'
    ]
    lab_measurements = [
        'Capillary refill rate', 'Glucose', 'pH', 'Temperature']
    # TODO: Figure out how to treat Glascow scale estimates
    interventions = [
        'Fraction inspired oxygen', 'Glascow coma scale eye opening',
        'Glascow coma scale motor response', 'Glascow coma scale total',
        'Glascow coma scale verbal response'
    ]
    coma_scale_eye_opening_replacements = {
        "1 No Response": 1,
        "None": 1,
        "2 To pain": 2,
        "To Pain": 2,
        "3 To speech": 3,
        "To Speech": 3,
        "4 Spontaneously": 4,
        "Spontaneously": 4,
    }
    coma_scale_motor_replacements = {
        "1 No Response": 1,
        "No response": 1,
        "2 Abnorm extensn": 2,
        "Abnormal extension": 2,
        "3 Abnorm flexion": 3,
        "Abnormal Flexion": 3,
        "4 Flex-withdraws": 4,
        "Flex-withdraws": 4,
        "5 Localizes Pain": 5,
        "Localizes Pain": 5,
        "6 Obeys Commands": 6,
        "Obeys Commands": 6
    }
    coma_scale_verbal_replacements = {
        "No Response-ETT": 0,
        "1.0 ET/Trach": 0,
        "1 No Response": 1,
        "No Response": 1,
        "2 Incomp sounds": 2,
        "Incomprehensible sounds": 2,
        "3 Inapprop words": 3,
        "Inappropriate Words": 3,
        "4 Confused": 4,
        "Confused": 4,
        "5 Oriented": 5,
        "Oriented": 5,
    }
    phenotypes = [
        'Acute and unspecified renal failure', 'Acute cerebrovascular disease',
        'Acute myocardial infarction', 'Cardiac dysrhythmias',
        'Chronic kidney disease',
        'Chronic obstructive pulmonary disease and bronchiectasis',
        'Complications of surgical procedures or medical care',
        'Conduction disorders', 'Congestive heart failure; nonhypertensive',
        'Coronary atherosclerosis and other heart disease',
        'Diabetes mellitus with complications',
        'Diabetes mellitus without complication',
        'Disorders of lipid metabolism', 'Essential hypertension',
        'Fluid and electrolyte disorders', 'Gastrointestinal hemorrhage',
        'Hypertension with complications and secondary hypertension',
        'Other liver diseases', 'Other lower respiratory disease',
        'Other upper respiratory disease',
        'Pleurisy; pneumothorax; pulmonary collapse',
        'Pneumonia (except that caused by tuberculosis or sexually transmitted disease)',  # noqa: E501
        'Respiratory failure; insufficiency; arrest (adult)',
        'Septicemia (except in labor)',
        'Shock'
    ]

    def __init__(self, dataset_dir, listfile):
        """Reader for phenotyping dataset of the MIMIC-III benchmarks."""
        self.dataset_dir = dataset_dir
        with tf.io.gfile.GFile(listfile, 'r') as f:
            self.instances = pd.read_csv(f, header=0, sep=',')

    def __getitem__(self, index):
        """Get instance with index."""
        instance = self.instances.iloc[index]
        length_of_stay = float(instance['period_length'])
        phenotype = instance.iloc[2:].values.astype(int)
        data_file = join(self.dataset_dir, instance['stay'])
        time, demographics, vitals, lab_measurements, intervensions = \
            self._read_data_for_instance(data_file)
        patient_id = int(instance['stay'].split('_')[0])

        return {
            'demographics': demographics,
            'time': time,
            'vitals': vitals,
            'lab_measurements': lab_measurements,
            'intervensions': intervensions,
            'targets': {
                'Length_of_stay': length_of_stay,
                'Phenotype': phenotype.astype(np.uint32)
            },
            'metadata': {
                'patient_id': patient_id
            }
        }

    def _read_data_for_instance(self, filename):
        with tf.io.gfile.GFile(filename, 'r') as f:
            data = pd.read_csv(f, header=0, sep=',')
        time = data['Hours']
        demographics = data[self.demographics].mean()
        vitals = data[self.vitals]
        lab_measurements = data[self.lab_measurements]
        interventions = data[self.interventions]
        interventions = self.__preprocess_coma_scales(interventions)
        return time, demographics, vitals, lab_measurements, interventions

    def __preprocess_coma_scales(self, data):
        to_replace = {
            "Glascow coma scale eye opening":
                self.coma_scale_eye_opening_replacements,
            "Glascow coma scale motor response":
                self.coma_scale_motor_replacements,
            "Glascow coma scale verbal response":
                self.coma_scale_verbal_replacements
        }
        coma_scale_columns = list(to_replace.keys())
        coma_scales = data[coma_scale_columns]
        coma_scales = coma_scales.astype(str)
        coma_scales = coma_scales.replace(
            to_replace=to_replace
        )
        coma_scales = coma_scales.astype(float)
        data = data.copy()
        data[coma_scale_columns] = coma_scales
        return data

    def __len__(self):
        """Get number of instances that can be read."""
        return len(self.instances)


class Mimic3Phenotyping(MedicalTsDatasetBuilder):
    """Dataset of the PhysioNet/Computing in Cardiology Challenge 2012."""

    VERSION = tfds.core.Version('1.0.0')
    MANUAL_DOWNLOAD_INSTRUCTIONS = """\
    manual_dir should contain the file `mimic_benchmarking_phenotyping.tar.gz`\
    """

    def _info(self):
        return MedicalTsDatasetInfo(
            builder=self,
            has_demographics=True,
            has_vitals=True,
            has_lab_measurements=True,
            has_interventions=True,
            targets={
                'Phenotype':
                    tfds.features.Tensor(
                        shape=(len(MIMICPhenotypingReader.phenotypes),),
                        dtype=tf.uint32
                    ),
                'Length_of_stay':
                    tfds.features.Tensor(shape=tuple(), dtype=tf.float32)
            },
            default_target='Phenotype',
            demographics_names=MIMICPhenotypingReader.demographics,
            vitals_names=MIMICPhenotypingReader.vitals,
            lab_measurements_names=MIMICPhenotypingReader.lab_measurements,
            interventions_names=MIMICPhenotypingReader.interventions,
            description=_DESCRIPTION,
            homepage='https://github.com/yerevann/mimic3-benchmarks',
            citation=_CITATION
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Return SplitGenerators."""
        data_file = join(
            dl_manager.manual_dir, 'mimic_benchmarking_phenotyping.tar.gz')
        extracted_path = dl_manager.extract(data_file)
        train_dir = join(extracted_path, 'train')
        train_listfile = join(extracted_path, 'train_listfile.csv')
        val_dir = train_dir
        val_listfile = join(extracted_path, 'val_listfile.csv')
        test_dir = join(extracted_path, 'test')
        test_listfile = join(extracted_path, 'test_listfile.csv')

        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                num_shards=20,
                gen_kwargs={
                    'data_dir': train_dir,
                    'listfile': train_listfile
                }
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.VALIDATION,
                num_shards=5,
                gen_kwargs={
                    'data_dir': val_dir,
                    'listfile': val_listfile
                }
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                num_shards=10,
                gen_kwargs={
                    'data_dir': test_dir,
                    'listfile': test_listfile
                }
            ),
        ]

    def _generate_examples(self, data_dir, listfile):
        """Yield examples."""
        index = 0
        reader = MIMICPhenotypingReader(data_dir, listfile)
        for instance in reader:
            yield index, instance
            index += 1
