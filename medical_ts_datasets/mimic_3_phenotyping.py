"""Module containing phenotyping dataset of MIMIC-III benchmarks."""

from os.path import join

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from .mimic_3_reader import MIMICReader
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


class MIMICPhenotypingReader(MIMICReader):
    """Reader for phenotyping dataset of the MIMIC-III benchmarks."""

    # Blacklisted instances due to unusually many observations compared to the
    # overall distribution.
    blacklist = [
        # Criterion for exclusion: more than 2000 distinct timepoints
        # Train data
        '50883_episode1_timeseries.csv', '70492_episode1_timeseries.csv',
        '711_episode3_timeseries.csv', '24915_episode1_timeseries.csv',
        '73129_episode2_timeseries.csv', '3932_episode1_timeseries.csv',
        '24597_episode1_timeseries.csv', '31123_episode1_timeseries.csv',
        '99383_episode1_timeseries.csv', '16338_episode1_timeseries.csv',
        '48123_episode2_timeseries.csv', '1785_episode1_timeseries.csv',
        '56854_episode1_timeseries.csv', '76151_episode2_timeseries.csv',
        '72908_episode1_timeseries.csv', '26277_episode3_timeseries.csv',
        '77614_episode1_timeseries.csv', '6317_episode3_timeseries.csv',
        '82609_episode1_timeseries.csv', '79645_episode1_timeseries.csv',
        '12613_episode1_timeseries.csv', '77617_episode1_timeseries.csv',
        '41861_episode1_timeseries.csv', '55205_episode1_timeseries.csv',
        '45910_episode1_timeseries.csv', '80927_episode1_timeseries.csv',
        '49555_episode1_timeseries.csv', '19911_episode3_timeseries.csv',
        '43459_episode1_timeseries.csv', '21280_episode2_timeseries.csv',
        '90776_episode1_timeseries.csv', '51078_episode2_timeseries.csv',
        '65565_episode1_timeseries.csv', '41493_episode1_timeseries.csv',
        '10694_episode2_timeseries.csv', '54073_episode1_timeseries.csv',
        '12831_episode2_timeseries.csv', '89223_episode1_timeseries.csv',
        '46156_episode1_timeseries.csv', '58242_episode4_timeseries.csv',
        '55639_episode1_timeseries.csv', '89840_episode1_timeseries.csv',
        # Validation data
        '67906_episode1_timeseries.csv', '59268_episode1_timeseries.csv',
        '78251_episode1_timeseries.csv', '32476_episode1_timeseries.csv',
        '96924_episode2_timeseries.csv', '96686_episode10_timeseries.csv',
        '5183_episode1_timeseries.csv', '58723_episode1_timeseries.csv',
        '78515_episode1_timeseries.csv', '40187_episode1_timeseries.csv',
        '62239_episode2_timeseries.csv', '79337_episode1_timeseries.csv',
        # Testing data
        '29105_episode2_timeseries.csv', '69745_episode4_timeseries.csv',
        '59726_episode1_timeseries.csv', '81786_episode1_timeseries.csv',
        '12805_episode1_timeseries.csv', '6145_episode1_timeseries.csv',
        '54353_episode2_timeseries.csv', '58854_episode1_timeseries.csv',
        '98994_episode1_timeseries.csv', '19223_episode2_timeseries.csv',
        '80345_episode1_timeseries.csv', '48935_episode1_timeseries.csv',
        '48380_episode1_timeseries.csv', '70698_episode1_timeseries.csv',
        '51177_episode1_timeseries.csv'
    ]

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
        """Initialize MIMIC-III phenotyping reader."""
        super().__init__(dataset_dir, listfile, self.blacklist)

    def __getitem__(self, index):
        """Get instance with index."""
        instance = self.instances.iloc[index]
        length_of_stay = float(instance['period_length'])
        phenotype = instance.iloc[2:].values.astype(int)
        data_file = join(self.dataset_dir, instance['stay'])
        time, demographics, vitals, lab_measurements, intervensions = \
            self._read_data_for_instance(data_file)
        patient_id = int(instance['stay'].split('_')[0])

        return instance['stay'], {
            'demographics': demographics,
            'time': time,
            'vitals': vitals,
            'lab_measurements': lab_measurements,
            'interventions': intervensions,
            'targets': {
                'Length_of_stay': length_of_stay,
                'Phenotype': phenotype.astype(np.int32)
            },
            'metadata': {
                'patient_id': patient_id
            }
        }


class Mimic3Phenotyping(MedicalTsDatasetBuilder):
    """Phenotyping task dataset of the MIMIC-III benchmarks."""

    VERSION = tfds.core.Version('1.0.2')
    MANUAL_DOWNLOAD_INSTRUCTIONS = """\
    manual_dir should contain the file `mimic_benchmarking_phenotyping.tar.gz`\
    """

    reader = MIMICPhenotypingReader
    has_demographics = True
    has_vitals = True
    has_lab_measurements = True
    has_interventions = True
    default_target = 'Phenotype'

    def _info(self):
        return MedicalTsDatasetInfo(
            builder=self,
            targets={
                'Phenotype':
                    tfds.features.Tensor(
                        shape=(len(MIMICPhenotypingReader.phenotypes),),
                        dtype=tf.int32
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
                gen_kwargs={
                    'data_dir': train_dir,
                    'listfile': train_listfile
                }
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.VALIDATION,
                gen_kwargs={
                    'data_dir': val_dir,
                    'listfile': val_listfile
                }
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs={
                    'data_dir': test_dir,
                    'listfile': test_listfile
                }
            ),
        ]

    def _generate_examples(self, data_dir, listfile):
        """Yield examples."""
        reader = MIMICPhenotypingReader(data_dir, listfile)
        for patient_id, instance in reader:
            yield patient_id, instance
