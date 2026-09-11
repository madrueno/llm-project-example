"""RAG prompt: dynamic example selection by semantic similarity."""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from intent_classification.prompting.base import BasePrompt


class RAGPrompt(BasePrompt):
    """RAG prompt builder that retrieves similar examples per query"""

    def __init__(
        self,
        X_train: list[str],
        y_train: list[str],
        k: int = 5,
        model_name: str = 'all-MiniLM-L6-v2',
    ) -> None:
        """
        Build the index over the training pool.

        Parameters
        ----------
        X_train : list of str
            Training texts to index
        y_train : list of str
            Training labels
        k : int, default=5
            Number of examples retrieved per query
        model_name : str, default='all-MiniLM-L6-v2'
            Sentence-transformers model used for the embeddings
        """
        self.labels_block = self.load('labels.md')
        self.build_index(X_train, y_train, model_name)
        self.k = k

    def build_index(self, X_train: list[str], y_train: list[str], model_name: str) -> None:
        """
        Embed the training pool for similarity search.

        Parameters
        ----------
        X_train : list of str
            Training texts to index
        y_train : list of str
            Training labels
        model_name : str
            Sentence-transformers model used for the embeddings
        """
        pool = pd.DataFrame({'text': X_train, 'label': y_train})
        self.pool = pool.reset_index(drop=True)

        self.encoder = SentenceTransformer(model_name)
        self.embeddings = self.encoder.encode(
            self.pool['text'].tolist(), show_progress_bar=True, convert_to_numpy=True
        )

    def retrieve(self, query: str) -> pd.DataFrame:
        """
        Return the k most semantically similar examples to the query.

        Parameters
        ----------
        query : str
            Query text to retrieve examples for

        Returns
        -------
        pandas.DataFrame
            The k most similar training examples
        """
        query_emb = self.encoder.encode([query], convert_to_numpy=True)
        scores = (self.embeddings @ query_emb.T).squeeze()
        top_k = np.argsort(scores)[::-1][: self.k]
        return self.pool.iloc[top_k].reset_index(drop=True)

    def build(self, text: str) -> list[dict]:
        """
        Build multi-turn chat messages with retrieved examples as user/assistant pairs.

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
        for row in self.retrieve(text).itertuples():
            messages.append({'role': 'user', 'content': row.text})
            messages.append({'role': 'assistant', 'content': row.label})
        messages.append({'role': 'user', 'content': text})
        return messages
