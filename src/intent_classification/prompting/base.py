"""Shared chat-prompt base class."""

from abc import ABC, abstractmethod

from intent_classification.config import PROMPTS_DIR


class BasePrompt(ABC):
    """
    Base chat-prompt builder.

    Subclasses implement ``build`` to return the full chat message list.
    """

    @staticmethod
    def load(name: str) -> str:
        """
        Read and strip a template file from the prompts directory.

        Parameters
        ----------
        name : str
            Template file name under the prompts directory

        Returns
        -------
        str
            File contents with surrounding whitespace stripped
        """
        return (PROMPTS_DIR / name).read_text(encoding='utf-8').strip()

    TASK = load('task.md')

    @abstractmethod
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
            Chat messages ready for apply_chat_template
        """
