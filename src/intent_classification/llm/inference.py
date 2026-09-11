"""LLM intent classifier served by a local Ollama instance."""

from typing import Literal

from langchain_ollama import ChatOllama
from pydantic import BaseModel
from tqdm import tqdm


class Intent(BaseModel):
    label: Literal['Account', 'Cards', 'Disputes', 'Payments', 'Topups', 'Transfers']


class LLMClient:
    """
    LLM intent classifier over a local Ollama model with structured output.

    Parameters
    ----------
    prompt : ZeroShotPrompt, FewShotPrompt or RAGPrompt
        Prompt builder exposing build(text) -> chat messages
    model_name : str
        Ollama model tag
    temperature : float, default=0.0
        Sampling temperature
    """

    def __init__(self, prompt, model_name: str, temperature: float = 0.0) -> None:
        self.prompt = prompt
        self.model_name = model_name

        self.llm = ChatOllama(model=model_name, temperature=temperature)
        self.llm = self.llm.with_structured_output(Intent)

    def predict(self, texts: list[str]) -> list[str]:
        """
        Predict labels for input texts.

        Structured output constrains each response to an Intent whose label
        is one of the valid labels, so every prediction is a valid intent.

        Parameters
        ----------
        texts : list of str
            Texts to classify

        Returns
        -------
        list of str
            Predicted labels, one per text
        """
        return [self.llm.invoke(self.prompt.build(text)).label for text in tqdm(texts)]
