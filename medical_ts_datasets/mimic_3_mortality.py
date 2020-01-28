"""Module containing mortality prediction dataset of MIMIC-III benchmarks."""
from os.path import join

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
In hospital mortality prediction task of the MIMIC-III benchmarks.
"""


class MIMICMortalityReader(MIMICReader):
    """Reader for mortality prediction of the MIMIC-III benchmarks."""

    # Blacklisted instances due to unusually many observations compared to the
    # overall distribution.
    blacklist = [
        # Criterion for exclusion: more than 1000 distinct timepoints
        # In training data
        '73129_episode2_timeseries.csv', '48123_episode2_timeseries.csv',
        '76151_episode2_timeseries.csv', '41493_episode1_timeseries.csv',
        '65565_episode1_timeseries.csv', '55205_episode1_timeseries.csv',
        '41861_episode1_timeseries.csv', '58242_episode4_timeseries.csv',
        '54073_episode1_timeseries.csv', '46156_episode1_timeseries.csv',
        '55639_episode1_timeseries.csv', '89840_episode1_timeseries.csv',
        '43459_episode1_timeseries.csv', '10694_episode2_timeseries.csv',
        '51078_episode2_timeseries.csv', '90776_episode1_timeseries.csv',
        '89223_episode1_timeseries.csv', '12831_episode2_timeseries.csv',
        '80536_episode1_timeseries.csv',
        # In validation data
        '78515_episode1_timeseries.csv', '62239_episode2_timeseries.csv',
        '58723_episode1_timeseries.csv', '40187_episode1_timeseries.csv',
        '79337_episode1_timeseries.csv',
        # In testing data
        '51177_episode1_timeseries.csv', '70698_episode1_timeseries.csv',
        '48935_episode1_timeseries.csv', '54353_episode2_timeseries.csv',
        '19223_episode2_timeseries.csv', '58854_episode1_timeseries.csv',
        '80345_episode1_timeseries.csv', '48380_episode1_timeseries.csv'
    ]

    def __init__(self, dataset_dir, listfile):
        """Initialize MIMIC-III mortality reader."""
        super().__init__(dataset_dir, listfile, self.blacklist)

    def __getitem__(self, index):
        """Get instance with index."""
        instance = self.instances.iloc[index]
        mortality = int(instance['y_true'])
        data_file = join(self.dataset_dir, instance['stay'])
        time, demographics, vitals, lab_measurements, interventions = \
            self._read_data_for_instance(data_file)
        patient_id = int(instance['stay'].split('_')[0])

        return instance['stay'], {
            'demographics': demographics,
            'time': time,
            'vitals': vitals,
            'lab_measurements': lab_measurements,
            'interventions': interventions,
            'targets': {
                'In_hospital_mortality': mortality
            },
            'metadata': {
                'patient_id': patient_id
            }
        }


class Mimic3Mortality(MedicalTsDatasetBuilder):
    """In hospital mortality task dataset of the MIMIC-III benchmarks."""

    VERSION = tfds.core.Version('1.0.1')
    MANUAL_DOWNLOAD_INSTRUCTIONS = """\
    manual_dir should contain the file `mimic_benchmarking_mortality.tar.gz`\
    """

    has_demographics = True
    has_vitals = True
    has_lab_measurements = True
    has_interventions = True
    default_target = 'In_hospital_mortality'

    def _info(self):
        return MedicalTsDatasetInfo(
            builder=self,
            targets={
                'In_hospital_mortality':
                    tfds.features.ClassLabel(num_classes=2)
            },
            default_target='In_hostpital_mortality',
            demographics_names=MIMICMortalityReader.demographics,
            vitals_names=MIMICMortalityReader.vitals,
            lab_measurements_names=MIMICMortalityReader.lab_measurements,
            interventions_names=MIMICMortalityReader.interventions,
            description=_DESCRIPTION,
            homepage='https://github.com/yerevann/mimic3-benchmarks',
            citation=_CITATION
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Return SplitGenerators."""
        data_file = join(
            dl_manager.manual_dir, 'mimic_benchmarking_mortality.tar.gz')
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
        reader = MIMICMortalityReader(data_dir, listfile)
        for patient_id, instance in reader:
            yield patient_id, instance
