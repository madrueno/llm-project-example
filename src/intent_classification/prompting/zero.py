"""Zero-shot prompt for intent classification."""

from intent_classification.prompting.base import BasePrompt


class ZeroShotPrompt(BasePrompt):
    """Zero-shot prompt builder"""

    def __init__(self) -> None:
        """Build the valid-intents block from labels.md."""
        self.labels_block = self.load('labels.md')

    def build(self, text: str) -> list[dict]:
        """
        Build the chat messages for one text.

        Parameters
        ----------
        text : str
            Input text to classify

        Returns
        -------
        list of dict
            System and user chat messages for the text
        """
        content = '\n\n'.join([self.TASK, self.labels_block])
        return [{'role': 'system', 'content': content}, {'role': 'user', 'content': text}]
