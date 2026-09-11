"""Training pipeline for supervised fine-tuning banking intent classification.

This experiment fine-tunes a causal language model with LoRA adapters.

Trained adapter is stored in models/custom directory.

WARNING: Test set results should ONLY be examined after selecting the best
approach based on dev set performance.
"""

import argparse

from intent_classification import IntentsDataset, SFTClassifier, evaluation
from intent_classification.config import PROMPTS_DIR


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Fine-tune LLM with LoRA for intent classification'
    )
    parser.add_argument(
        '--load', action='store_true', help='Load an existing adapter instead of training a new one'
    )
    return parser.parse_args()


def main(args):
    """Load dataset, fine-tune SFT adapter, and evaluate on train and dev."""
    # ===== DATASET / PIPELINE =====
    dataset = IntentsDataset()

    labels = sorted(set(dataset.y_train))
    system_prompt = (PROMPTS_DIR / 'task.md').read_text(encoding='utf-8').strip()
    sft = SFTClassifier(system_prompt=system_prompt)

    if args.load:
        print('Loading existing adapter...')
        sft.load()
    else:
        print('Training new adapter (will overwrite if exists)...')
        sft.train(dataset.X_train, dataset.y_train)

    # ========== TRAIN SET ==========
    print('=' * 70 + '\n' + 'PROCESSING TRAIN SET' + '\n' + '=' * 70)
    y_train_pred = sft.predict(dataset.X_train, labels)
    train_metrics = evaluation.evaluate_metrics(dataset.y_train, y_train_pred)

    # ========== DEV SET ==========
    print('=' * 70 + '\n' + 'PROCESSING DEV SET (for model selection)' + '\n' + '=' * 70)
    y_dev_pred = sft.predict(dataset.X_dev, labels)
    dev_metrics = evaluation.evaluate_metrics(dataset.y_dev, y_dev_pred)

    # ========== FINAL SUMMARY ==========
    experiment = f'llm-sft-local-{evaluation.model_slug(sft.model_name)}'
    evaluation.save_predictions(
        dataset.X_train, dataset.y_train, y_train_pred, experiment, 'train', sft.model_name
    )
    evaluation.save_predictions(
        dataset.X_dev, dataset.y_dev, y_dev_pred, experiment, 'dev', sft.model_name
    )
    evaluation.print_experiment_summary(train_metrics, dev_metrics, experiment_name=experiment)


if __name__ == '__main__':
    args = parse_args()
    main(args)
