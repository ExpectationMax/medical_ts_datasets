"""Split physionet 2012 dataset into train, validation and test."""
import argparse
import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('outcome_a', type=str)
    parser.add_argument('outcome_b', type=str)
    parser.add_argument('outcome_c', type=str)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    # From random.org
    random_seed = 145469037
    np.random.seed(random_seed)

    a = pd.read_csv(args.outcome_a)
    b = pd.read_csv(args.outcome_b)
    c = pd.read_csv(args.outcome_c)
    all_outcomes = pd.concat([a, b, c], axis=0)
    y = all_outcomes['In-hospital_death'].values
    all_train_data, test_data = train_test_split(
        all_outcomes, stratify=y, test_size=0.2)
    test_data.to_csv(
        os.path.join(args.output, 'test_listfile.csv'), index=False)

    y = all_train_data['In-hospital_death'].values
    train_data, val_data = train_test_split(
        all_train_data, stratify=y, test_size=0.2)

    train_data.to_csv(
        os.path.join(args.output, 'train_listfile.csv'), index=False)
    val_data.to_csv(
        os.path.join(args.output, 'val_listfile.csv'), index=False)


if __name__ == '__main__':
    main()
