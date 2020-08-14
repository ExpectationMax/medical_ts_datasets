============================
Medical time series datasets
============================

This module contains the implementation of multiple medical time series datasets
following the tensorflow dataset API.

Currently implemented datasets are:

- ``physionet2012`` (mortality prediction)
- ``mimic3_mortality`` (mortality prediction)
- ``mimic3_phenotyping`` (mortality prediction)
- ``physionet2019`` (online sepsis early prediction)


Example usage
-------------

In order to get a tensorflow dataset representation of one of the datasets simply
import ``tensorflow_datasets`` and this module.  The datasets can then be accessed
like any other tensorflow dataset.

.. code-block:: python

    import tensorflow_datasets as tfds
    import medical_ts_datasets

    physionet_dataset = tfds.load(name='physionet2012', split='train')


Instance structure
------------------

Each instance in the dataset is represented as a nested directory of the following
structure:

- ``statics``: Static variables such as demographics or the unit the patient was
  admitted to
- ``time``: Scalar time variable containing the time since admission in hours
- ``values``: Observation values of time series, these by default contain `NaN` for
  modalities which were not observed for the given timepoint.
- ``targets``: Directory of potential target values, the available endpoints are
  dataset specific.
- ``metadata``: Directory of metadata on an individual patient, such as the
  identifier using in the dataset.

Supervised dataset
------------------

If the load method is called with the flag ``as_supervised=True``, it will
return a dataset which can readily be used together with keras. Here each
instance is represented by a (X, y) tuple and the X tuple contains the
following 4 elements: ``time``, ``values``, ``measurements`` (indicators if
a value was measured or not) and ``length``.

Citing
------

When using datasets from this repository, please do not forget to cite the following references besides a reference to the original dataset:

.. code-block:: bibtex

    @InProceedings{horn2020set,
        title = {{S}et {F}unctions for {T}ime {S}eries},
        author = {Max Horn and Michael Moor and Christian Bock and Bastian Rieck and Karsten Borgwardt},
        booktitle = {Proceedings of the 37th International Conference on Machine Learning},
        year = {2020},
        volume = {119},
        series = {Proceedings of Machine Learning Research},
        publisher = {PMLR}
    }

    @misc{TFDS,
      title = {{TensorFlow Datasets}, A collection of ready-to-use datasets},
      howpublished = {\url{https://www.tensorflow.org/datasets}},
    }
