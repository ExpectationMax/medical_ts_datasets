"""Utility functions for MIMIC-III benchmarking datasets."""
import tensorflow as tf
import pandas as pd


class MIMICReader:
    """Reader base class for MIMIC-III benchmarks."""

    demographics = ['Height']
    vitals = [
        'Weight', 'Heart Rate', 'Mean blood pressure',
        'Diastolic blood pressure', 'Systolic blood pressure',
        'Oxygen saturation', 'Respiratory rate'
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

    def __init__(self, dataset_dir, listfile, blacklist=None):
        """Reader for phenotyping dataset of the MIMIC-III benchmarks."""
        self.dataset_dir = dataset_dir
        with tf.io.gfile.GFile(listfile, 'r') as f:
            self.instances = pd.read_csv(f, header=0, sep=',')

        if blacklist is not None:
            # Remove instances which are on the blacklist
            self.instances = self.instances[
                ~self.instances['stay'].isin(blacklist)
            ]

    def _read_data_for_instance(self, filename):
        """Read a single instance from file.

        Args:
            filename: Filename from which to read data.

        """
        with tf.io.gfile.GFile(filename, 'r') as f:
            data = pd.read_csv(f, header=0, sep=',')
        time = data['Hours']
        # Sometimes the demographics might be NaN.
        # Thus we might need to replace those with a placeholder number (in our
        # case -1)
        demographics = (
            data[self.demographics]
            .replace({-1: float('NaN')})
            .mean()
        )
        demographics = demographics.fillna(value=-1)
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
