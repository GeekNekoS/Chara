from project_logging import setup_logger
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np

logger = setup_logger("evaluate_model")


def evaluate_model(model: tf.keras.Model, test: tf.data.Dataset):
    logger.info("function starts")
    try:
        y_true = []
        y_pred = []

        for x, y in test:
            y_true.extend(y.numpy())
            predictions = model.predict(x)
            y_pred.extend(np.argmax(predictions, axis=1))

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        cm = confusion_matrix(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')

        print("Confusion Matrix:")
        print(cm)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        history = model.history.history

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # График accuracy
        ax1.plot(history['accuracy'], label='Train Accuracy')
        ax1.plot(history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Accuracy over Epochs')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Accuracy')
        ax1.legend()

        # График loss
        ax2.plot(history['loss'], label='Train Loss')
        ax2.plot(history['val_loss'], label='Validation Loss')
        ax2.set_title('Loss over Epochs')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss')
        ax2.legend()

        plt.show()
        logger.info("successful ended")
    except Exception as exc:
        logger.error(f"{exc}")

