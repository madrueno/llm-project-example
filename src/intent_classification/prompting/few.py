"""Static few-shot prompt for intent classification."""

import pandas as pd

from intent_classification.prompting.base import BasePrompt


class FewShotPrompt(BasePrompt):
    """Few-shot prompt builder with static examples sampled from training data"""

    def __init__(self, X_train: list[str], y_train: list[str], n_per_class: int = 5) -> None:
        """
        Sample examples from training data and build the system prompt.

        Parameters
        ----------
        X_train : list of str
            Training texts
        y_train : list of str
            Training labels
        n_per_class : int, default=5
            Number of examples to sample per class
        """
        self.labels_block = self.load('labels.md')
        self.examples = self.sample_examples(X_train, y_train, n_per_class)

    def build(self, text: str) -> list[dict]:
        """
        Build multi-turn chat messages with examples as user/assistant pairs.

        Parameters
        ----------
        text : str
            Input text to classify

        Returns
        -------
        list of dict
            System message, interleaved example user/assistant turns, and
            the final user query
        """
        system_content = '\n\n'.join([self.TASK, self.labels_block])
        messages = [{'role': 'system', 'content': system_content}]
        for row in self.examples.itertuples():
            messages.append({'role': 'user', 'content': row.text})
            messages.append({'role': 'assistant', 'content': row.label})
        messages.append({'role': 'user', 'content': text})
        return messages

    @staticmethod
    def sample_examples(X_train: list[str], y_train: list[str], n_per_class: int) -> pd.DataFrame:
        """
        Return n_per_class examples per class from the training pool.

        Parameters
        ----------
        X_train : list of str
            Training texts
        y_train : list of str
            Training labels
        n_per_class : int
            Number of examples to sample per class

        Returns
        -------
        pandas.DataFrame
            Sampled examples, sorted by label
        """
        return (
            pd.DataFrame({'text': X_train, 'label': y_train})
            .sample(frac=1, random_state=42)
            .groupby('label')
            .head(n_per_class)
            .sort_values('label')
            .reset_index(drop=True)
        )
