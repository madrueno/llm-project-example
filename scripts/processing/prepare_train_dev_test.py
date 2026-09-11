"""Script to split interim data into train, dev, and test sets."""

import pandas as pd
from sklearn.model_selection import train_test_split

from intent_classification.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR


# Capped for educational purposes to reduce runtime and cost
TRAIN_SIZE, DEV_SIZE, TEST_SIZE = 500, 250, 250


def load_interim_splits() -> tuple:
    """
    Load the interim train/test intent files.

    Returns
    -------
    tuple
        (train_df, test_df) DataFrames with text and label columns
    """
    all_data = []
    for split_name in ['train', 'test']:
        filepath = INTERIM_DATA_DIR / f'intents_{split_name}.csv'
        print(f'Loading {filepath}...')
        df = pd.read_csv(filepath)
        print(f'  Rows: {len(df)}')
        all_data.append(df)
    print('-' * 40)

    return tuple(all_data)


def split_train_dev_test(
    train_df: pd.DataFrame, test_df: pd.DataFrame, random_state: int = 42
) -> tuple:
    """
    Split data into train, dev, and test sets with stratification.

    Dev is carved from the official train set. All three splits are stratified
    and capped at TRAIN_SIZE/DEV_SIZE/TEST_SIZE for educational purposes.

    Parameters
    ----------
    train_df : pd.DataFrame
        Official train DataFrame with text and label columns
    test_df : pd.DataFrame
        Official test DataFrame with text and label columns
    random_state : int
        Random seed for reproducibility (default: 42)

    Returns
    -------
    tuple
        (train_df, dev_df, test_df) DataFrames
    """
    train_df, dev_df = train_test_split(
        train_df, test_size=DEV_SIZE, random_state=random_state, stratify=train_df['label']
    )

    _, train_df = train_test_split(
        train_df, test_size=TRAIN_SIZE, random_state=random_state, stratify=train_df['label']
    )
    _, test_df = train_test_split(
        test_df, test_size=TEST_SIZE, random_state=random_state, stratify=test_df['label']
    )

    print(f'Train set: {len(train_df)} rows')
    print(f'Dev set: {len(dev_df)} rows')
    print(f'Test set: {len(test_df)} rows')
    print('-' * 40)

    return train_df, dev_df, test_df


def main() -> None:
    """
    Load the interim intent files, then split into train/dev/test.

    Each split is saved to data/processed/{split}.csv.
    """
    train_df, test_df = load_interim_splits()

    # Split into train/dev/test
    train_df, dev_df, test_df = split_train_dev_test(train_df, test_df)

    # Save train set
    train_path = PROCESSED_DATA_DIR / 'train.csv'
    print(f'Saving train set to {train_path}...')
    train_df.to_csv(train_path, index=False)

    # Save dev set
    dev_path = PROCESSED_DATA_DIR / 'dev.csv'
    print(f'Saving dev set to {dev_path}...')
    dev_df.to_csv(dev_path, index=False)

    # Save test set
    test_path = PROCESSED_DATA_DIR / 'test.csv'
    print(f'Saving test set to {test_path}...')
    test_df.to_csv(test_path, index=False)

    print('-' * 40)
    print('Done!')


if __name__ == '__main__':
    main()
