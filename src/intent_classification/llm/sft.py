"""LLM supervised fine-tuning with QLoRA for intent classification."""

import outlines
import torch
from datasets import Dataset
from outlines import Generator
from outlines.inputs import Chat
from outlines.types import Choice
from peft import LoraConfig, PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from intent_classification.config import CUSTOM_MODELS_DIR


class SFTClassifier:
    """
    Fine-tune a causal LLM with QLoRA adapters for intent classification.

    Parameters
    ----------
    system_prompt : str
        System prompt used in the train and inference templates
    """

    def __init__(
        self, system_prompt: str = '', model_name: str = 'Qwen/Qwen2.5-0.5B-Instruct'
    ) -> None:
        """
        Set up paths and system prompt.

        Call load() or train() before predicting.

        Parameters
        ----------
        system_prompt : str, optional
            System prompt used in the train and inference templates
        model_name : str, optional
            HuggingFace model ID to fine-tune (default: Qwen/Qwen2.5-0.5B-Instruct)
        """
        self.model_name = model_name
        self.system_prompt = system_prompt

        self.adapter_dir = CUSTOM_MODELS_DIR / 'llm-sft'
        self.model, self.tokenizer = None, None

    def build_messages(self, text: str, label: str | None = None) -> list[dict[str, str]]:
        """
        Build a chat messages list for one sample.

        Parameters
        ----------
        text : str
            Input text to classify
        label : str, optional
            Training label (intent name) for the input text
        """
        messages = []

        if self.system_prompt:
            messages.append({'role': 'system', 'content': self.system_prompt})

        messages.append({'role': 'user', 'content': text})

        if label is not None:
            messages.append({'role': 'assistant', 'content': label})

        return messages

    def load(self) -> None:
        """Load the saved LoRA adapter from disk."""
        print('Loading base model + LoRA adapter...')

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map='auto', dtype=torch.bfloat16
        )
        self.model = PeftModel.from_pretrained(model, str(self.adapter_dir))
        self.model.generation_config.max_new_tokens = 5
        self.model.generation_config.do_sample = False
        self.model.eval()

        print('Model loaded successfully.')

    def train(
        self, X_train: list[str], y_train: list[str], num_epochs: int = 3, batch_size: int = 4
    ) -> None:
        """
        Fine-tune a new QLoRA adapter and save it.

        Parameters
        ----------
        X_train : list of str
            Training texts
        y_train : list of str
            Training labels (intent names)
        num_epochs : int, default=3
            Number of training epochs
        batch_size : int, default=4
            Training batch size
        """
        print(f'Training QLoRA adapter over {self.model_name}...')

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, quantization_config=bnb_config, device_map='auto'
        )
        hf_dataset = Dataset.from_dict(
            {
                'messages': [
                    self.build_messages(text, label) for text, label in zip(X_train, y_train)
                ]
            }
        )

        lora_config = LoraConfig(task_type='CAUSAL_LM', target_modules='all-linear')

        sft_config = SFTConfig(
            output_dir=str(self.adapter_dir),
            max_length=256,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            save_strategy='epoch',
            assistant_only_loss=True,
        )

        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=hf_dataset,
            processing_class=tokenizer,
            peft_config=lora_config,
        )

        trainer.train()
        trainer.save_model(str(self.adapter_dir))

        print(f'Saved adapter to {self.adapter_dir}')

        self.load()

    def predict(self, texts: list[str], labels: list[str]) -> list[str]:
        """
        Predict labels for input texts with constrained decoding.

        Generation is restricted to the token sequences of the valid
        labels, so every prediction is a valid intent.

        Parameters
        ----------
        texts : list of str
            Texts to classify
        labels : list of str
            Valid labels the generation is restricted to

        Returns
        -------
        list of str
            Predicted labels, one per text
        """
        outlines_model = outlines.from_transformers(self.model, self.tokenizer)
        generator = Generator(outlines_model, Choice(labels))
        return [generator(Chat(self.build_messages(text))) for text in tqdm(texts)]
