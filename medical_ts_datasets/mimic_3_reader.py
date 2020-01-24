"""Utility functions for MIMIC-III benchmarking datasets."""
import tensorflow as tf
import pandas as pd


class MIMICReader:
    """Reader base class for MIMIC-III benchmarks."""

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
        "1 No Response": 0,
        "None": 0,
        "2 To pain": 1,
        "To Pain": 1,
        "3 To speech": 2,
        "To Speech": 2,
        "4 Spontaneously": 3,
        "Spontaneously": 3,
    }
    coma_scale_motor_replacements = {
        "1 No Response": 0,
        "No response": 0,
        "2 Abnorm extensn": 1,
        "Abnormal extension": 1,
        "3 Abnorm flexion": 2,
        "Abnormal Flexion": 2,
        "4 Flex-withdraws": 3,
        "Flex-withdraws": 3,
        "5 Localizes Pain": 4,
        "Localizes Pain": 4,
        "6 Obeys Commands": 5,
        "Obeys Commands": 5
    }
    coma_scale_verbal_replacements = {
        "No Response-ETT": 5,
        "1.0 ET/Trach": 5,
        "1 No Response": 0,
        "No Response": 0,
        "2 Incomp sounds": 1,
        "Incomprehensible sounds": 1,
        "3 Inapprop words": 2,
        "Inappropriate Words": 2,
        "4 Confused": 3,
        "Confused": 3,
        "5 Oriented": 4,
        "Oriented": 4,
    }

    coma_scale_total_replacements: {
        "3": 0,
        "4": 1,
        "5": 2,
        "6": 3,
        "7": 4,
        "8": 5,
        "9": 6,
        "10": 7,
        "11": 8,
        "12": 9,
        "13": 10,
        "14": 11,
        "15": 12,
    }

    categorical_channels = {
        "Glascow coma scale eye opening":
            max(coma_scale_eye_opening_replacements.values()) + 1,
        "Glascow coma scale motor response":
            max(coma_scale_motor_replacements.values()) +1,
        "Glascow coma scale verbal response":
            max(coma_scale_verbal_replacements.values()) + 1,
        "Glascow coma scale total":
            max(coma_scale_total_replacements.values()) + 1
    }

    def __init__(self, dataset_dir, listfile, blacklist=None):
        """Reader for phenotyping dataset of the MIMIC-III benchmarks."""
        self.dataset_dir = dataset_dir
        with tf.io.gfile.GFile(listfile, 'r') as f:
            self.instances = pd.read_csv(f, header=0, sep=',')

        if blacklist is not None:
            # Remove instances which are on the blacklist
            self.instances = self.instances[
                self.instances['stay'].isin(blacklist)
            ]

    def _read_data_for_instance(self, filename):
        """Read a single instance from file.

        Args:
            filename: Filename from which to read data.

        """
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
            "Glascow coma scale total":
                self.coma_scale_total_replacements
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
