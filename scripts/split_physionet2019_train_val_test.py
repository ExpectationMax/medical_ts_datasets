"""Split physionet2019 dataset into train validation and test."""
import argparse
from glob import glob
import os

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('training', type=str)
    parser.add_argument('training_setB', type=str)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    # From random.org
    random_seed = 16742963
    np.random.seed(random_seed)

    # Read labels (if patient ever has sepsis)
    ids = []
    labels = []
    for filename in tqdm(sorted(os.listdir(args.training))):
        if not filename.endswith('psv'):
            continue
        timepoint_labels = pd.read_csv(
            os.path.join(args.training, filename),
            sep='|',
            usecols=['SepsisLabel'])
        timepoint_labels = timepoint_labels['SepsisLabel'].values
        ids.append(filename)
        labels.append(int(timepoint_labels.any()))

    for filename in tqdm(sorted(os.listdir(args.training_setB))):
        if not filename.endswith('psv'):
            continue
        timepoint_labels = pd.read_csv(
            os.path.join(args.training_setB, filename),
            sep='|',
            usecols=['SepsisLabel'])
        timepoint_labels = timepoint_labels['SepsisLabel'].values
        ids.append(filename)
        labels.append(int(timepoint_labels.any()))

    data = pd.DataFrame(
        zip(ids, labels), columns=['filename', 'SepsisLabel'])

    train_data, test_data, = train_test_split(
        data, stratify=data['SepsisLabel'], test_size=0.2)
    test_data.to_csv(
        os.path.join(args.output, 'test_listfile.csv'), index=False)

    train_data, val_data = train_test_split(
        train_data, stratify=train_data['SepsisLabel'], test_size=0.2)

    train_data.to_csv(
        os.path.join(args.output, 'train_listfile.csv'), index=False)
    val_data.to_csv(
        os.path.join(args.output, 'val_listfile.csv'), index=False)


if __name__ == '__main__':
    main()
