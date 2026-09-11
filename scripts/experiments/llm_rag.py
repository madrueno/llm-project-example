"""RAG-based banking intent classification with a local Ollama model.

This experiment retrieves similar training examples dynamically for each query.

WARNING: Test set results should ONLY be examined after selecting the best
approach based on dev set performance. Using test results for model selection
leads to overfitting and invalidates your evaluation.
"""

from intent_classification import IntentsDataset, LLMClient, RAGPrompt, evaluation


def main():
    """Load dataset, run RAG inference with the Ollama model, and evaluate."""
    dataset = IntentsDataset()

    prompt = RAGPrompt(dataset.X_train, dataset.y_train, k=5)
    client = LLMClient(prompt, model_name='qwen2.5:3b-instruct-q4_K_M')

    print('=' * 70 + '\n' + 'PROCESSING TRAIN SET' + '\n' + '=' * 70)
    y_train_pred = client.predict(dataset.X_train)
    train_metrics = evaluation.evaluate_metrics(dataset.y_train, y_train_pred)

    print('=' * 70 + '\n' + 'PROCESSING DEV SET' + '\n' + '=' * 70)
    y_dev_pred = client.predict(dataset.X_dev)
    dev_metrics = evaluation.evaluate_metrics(dataset.y_dev, y_dev_pred)

    experiment = f'llm-rag-{evaluation.model_slug(client.model_name)}'
    evaluation.save_predictions(
        dataset.X_train, dataset.y_train, y_train_pred, experiment, 'train', client.model_name
    )
    evaluation.save_predictions(
        dataset.X_dev, dataset.y_dev, y_dev_pred, experiment, 'dev', client.model_name
    )
    evaluation.print_experiment_summary(train_metrics, dev_metrics, experiment_name=experiment)


if __name__ == '__main__':
    main()
