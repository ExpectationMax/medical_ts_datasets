"""Utility functions and classes used by medical ts datasets."""
import logging

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

    @tfds.core.api_utils.disallow_positional_args
    def __init__(self, builder, has_demographics, has_vitals,
                 has_lab_measurements, has_interventions, targets,
                 default_target, demographics_names=None, vitals_names=None,
                 lab_measurements_names=None, interventions_names=None,
                 description=None, homepage=None, citation=None):
        """Dataset info for medical time series datasets.

        Ensures all datasets follow a similar structure and can be used
        (almost) interchangably.

        Args:
            builder: Builder class associated with this dataset info.
            has_demographics: The dataset has information on demographics.
            has_vitals: The dataset has vital measurements.
            has_lab_measurements: The dataset has lab measurements.
            has_interventions: The dataset has information on interventions.
            targets: Dictionary of endpoints.
            demographics_names: Names of the demographics.
            vitals_names: Names of the vital measurements.
            lab_measurements_names: Names of the lab measurements.
            interventions_names: Names of the intervensions.
            description: Dataset description.
            homepage: Homepage of dataset.
            citation: Citation of dataset.

        """
        self.has_demographics = has_demographics
        self.has_vitals = has_vitals
        self.has_lab_measurements = has_lab_measurements
        self.has_intervensions = has_interventions
        self.default_target = default_target

        metadata = tfds.core.MetadataDict()
        features_dict = {
            'time': Tensor(shape=(None,), dtype=self.time_dtype)
        }
        if has_demographics:
            metadata['demographics_names'] = demographics_names
            features_dict['demographics'] = Tensor(
                shape=(len(demographics_names),),
                dtype=self.demographics_dtype)
        if has_vitals:
            metadata['vitals_names'] = vitals_names
            features_dict['vitals'] = Tensor(
                shape=(None, len(vitals_names),),
                dtype=self.vitals_dtype)
        if has_lab_measurements:
            metadata['lab_measurements_names'] = lab_measurements_names
            features_dict['lab_measurements'] = Tensor(
                shape=(None, len(lab_measurements_names),),
                dtype=self.lab_measurements_dtype)
        if has_interventions:
            metadata['interventions_names'] = interventions_names
            features_dict['intervensions'] = Tensor(
                shape=(None, len(interventions_names),),
                dtype=self.interventions_dtype)

        features_dict['targets'] = targets
        features_dict['metadata'] = {'patient_id': self.patient_id_dtype}
        features_dict = FeaturesDict(features_dict)
        super().__init__(
            builder=builder,
            description=description, homepage=homepage, citation=citation,
            features=features_dict,
            supervised_keys=("combined", self.default_target),
            metadata=metadata
        )


class MedicalTsDatasetBuilder(tfds.core.GeneratorBasedBuilder):
    """Builder class for medical time series datasets."""

    combine_values = ['vitals', 'lab_measurements', 'interventions']

    def as_dataset(self, split=None, batch_size=None, combine_timeseries=False,
                   drop_demographics=False, replace_nans=None):
        """Get a tensorflow dataset from this builder.

        Args:
            split: Which split to return.
            batch_size: Batch size.
            combine_timeseries: Whether to combine fields with time series
                measurements into a single tensor.
            drop_demographics: If we should drop the demographics.

        Returns:
            A tensorflow dataset.

        """
        dataset = super().as_dataset(
            split=split, batch_size=None, as_supervised=False)
        if combine_timeseries or drop_demographics:
            # Combine vitals, labs and interventions to single tensor.
            def apply_transforms(d):
                if combine_timeseries:
                    d['values'] = tf.concat(
                        [
                            d[val]
                            for val in self.combine_values
                            if val in d.keys()
                        ],
                        axis=-1
                    )
                    for val in self.combine_values.keys():
                        if val in d.keys():
                            del d[val]

                    if replace_nans:
                        # TODO: Implement replacement of NaNs
                        raise NotImplementedError()

                if drop_demographics:
                    del d['demographics']

                if batch_size is not None:
                    # Add a length tensor if we are padding the batches
                    d['length'] = tf.shape(d['time'])[0]
                return d
            dataset = dataset.map(
                apply_transforms, n_parallel=tf.data.experimental.AUTOTUNE)
        if batch_size:
            dataset = dataset.padded_batch(
                batch_size, tf.compat.v1.data.get_output_shapes(dataset))

        return dataset
