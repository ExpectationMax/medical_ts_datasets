"""Utility functions and classes used by medical ts datasets."""
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_datasets.core.features import FeaturesDict, Tensor


class MedicalTsDatasetInfo(tfds.core.DatasetInfo):
    """DatasetINfo for medical time series datasets."""

    time_dtype = tf.float32
    demographics_dtype = tf.float32
    vitals_dtype = tf.float32
    lab_measurements_dtype = tf.float32
    interventions_dtype = tf.float32
    patient_id_dtype = tf.uint32

    def __init__(self, builder, targets, default_target,
                 demographics_names=None, vitals_names=None,
                 lab_measurements_names=None, interventions_names=None,
                 description=None, homepage=None, citation=None):
        """Dataset info for medical time series datasets.

        Ensures all datasets follow a similar structure and can be used
        (almost) interchangably.

        Args:
            builder: Builder class associated with this dataset info.
            targets: Dictionary of endpoints.
            demographics_names: Names of the demographics.
            vitals_names: Names of the vital measurements.
            lab_measurements_names: Names of the lab measurements.
            interventions_names: Names of the intervensions.
            description: Dataset description.
            homepage: Homepage of dataset.
            citation: Citation of dataset.

        """
        self.has_demographics = demographics_names is not None
        self.has_vitals = vitals_names is not None
        self.has_lab_measurements = lab_measurements_names is not None
        self.has_interventions = interventions_names is not None
        self.default_target = default_target

        metadata = tfds.core.MetadataDict()
        features_dict = {
            'time': Tensor(shape=(None,), dtype=self.time_dtype)
        }
        demo_is_categorical = []
        combined_is_categorical = []
        if self.has_demographics:
            metadata['demographics_names'] = demographics_names
            demo_is_categorical.extend(
                ['=' in demo_name for demo_name in demographics_names])
            features_dict['demographics'] = Tensor(
                shape=(len(demographics_names),),
                dtype=self.demographics_dtype)
        if self.has_vitals:
            metadata['vitals_names'] = vitals_names
            combined_is_categorical.extend(
                ['=' in name for name in vitals_names])
            features_dict['vitals'] = Tensor(
                shape=(None, len(vitals_names),),
                dtype=self.vitals_dtype)
        if self.has_lab_measurements:
            metadata['lab_measurements_names'] = lab_measurements_names
            combined_is_categorical.extend(
                ['=' in name for name in lab_measurements_names])
            features_dict['lab_measurements'] = Tensor(
                shape=(None, len(lab_measurements_names),),
                dtype=self.lab_measurements_dtype)
        if self.has_interventions:
            metadata['interventions_names'] = interventions_names
            combined_is_categorical.extend(
                ['=' in name for name in interventions_names])
            features_dict['interventions'] = Tensor(
                shape=(None, len(interventions_names),),
                dtype=self.interventions_dtype)

        metadata['demographics_categorical_indicator'] = demo_is_categorical
        metadata['combined_categorical_indicator'] = combined_is_categorical

        features_dict['targets'] = targets
        features_dict['metadata'] = {'patient_id': self.patient_id_dtype}
        features_dict = FeaturesDict(features_dict)
        # TODO: If we are supposed to return raw values, we cannot make
        # a supervised dataset
        if builder.output_raw:
            supervised_keys = None
        else:
            supervised_keys = ("combined", "target")

        super().__init__(
            builder=builder,
            description=description, homepage=homepage, citation=citation,
            features=features_dict,
            supervised_keys=supervised_keys,
            metadata=metadata
        )


class MedicalTsDatasetBuilder(tfds.core.GeneratorBasedBuilder):
    """Builder class for medical time series datasets."""

    def __init__(self, output_raw=False, add_measurements_and_lengths=True,
                 **kwargs):
        self.output_raw = output_raw
        self.add_measurements_and_lengths = add_measurements_and_lengths
        super().__init__(**kwargs)

    def _as_dataset(self, **kwargs):
        """Evtl. transform categorical covariates into one-hot encoding."""
        dataset = super()._as_dataset(**kwargs)
        if self.output_raw:
            return dataset

        has_demographics = self.info.has_demographics
        collect_ts = []
        if self.has_vitals:
            collect_ts.append('vitals')
        if self.has_lab_measurements:
            collect_ts.append('lab_measurements')
        if self.has_interventions:
            collect_ts.append('interventions')

        def preprocess_output(instance):
            if has_demographics:
                demographics = instance['demographics']
            else:
                demographics = None

            time = instance['time']
            time_series = tf.concat(
                [instance[mod_type] for mod_type in collect_ts], axis=-1)

            if self.add_measurements_and_lengths:
                measurements = tf.math.is_finite(time_series)
                length = tf.shape(time)[0]
                return {
                    'combined': (
                        demographics,
                        time,
                        time_series,
                        measurements,
                        length
                    ),
                    'target': instance['targets'][self.default_target]
                }
            else:
                return {
                    'combined': (demographics, time, time_series),
                    'target': instance['targets'][self.default_target]
                }

        return dataset.map(
            preprocess_output,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
