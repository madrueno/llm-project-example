"""Dataset loader for banking intent classification."""

import pandas as pd

from intent_classification.config import PROCESSED_DATA_DIR


class IntentsDataset:
    """
    Loader for the intent classification datasets.

    Attributes
    ----------
    X_train : list[str]
        Training features (query text)
    y_train : list[str]
        Training labels (intent names)
    X_dev : list[str]
        Dev/validation features
    y_dev : list[str]
        Dev/validation labels
    X_test : list[str]
        Test features (query text)
    y_test : list[str]
        Test labels (intent names)
    """

    def __init__(self):
        """Load train/dev/test splits from precomputed files."""
        train_df = pd.read_csv(PROCESSED_DATA_DIR / 'train.csv')
        dev_df = pd.read_csv(PROCESSED_DATA_DIR / 'dev.csv')
        test_df = pd.read_csv(PROCESSED_DATA_DIR / 'test.csv')

        self.X_train = train_df['text'].to_list()
        self.y_train = train_df['label'].to_list()

        self.X_dev = dev_df['text'].to_list()
        self.y_dev = dev_df['label'].to_list()

        self.X_test = test_df['text'].to_list()
        self.y_test = test_df['label'].to_list()
