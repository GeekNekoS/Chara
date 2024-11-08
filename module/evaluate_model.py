from module.project_logging import setup_logger
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np

logger = setup_logger("evaluate_model")


def evaluate_model(model: tf.keras.Model, test: tf.data.Dataset):
    logger.info("function starts")
    # Прогноз на основе тестовых данных
    y_pred = model.predict(test, batch_size=64)

    y_true = []
    # Проход по всему датасету test
    for x, y in test:
        # Собираем истинные метки из датасета test
        y_true.extend(y.numpy())  # Преобразуем тензор меток в numpy и добавляем в список

    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_true, axis=1)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    logger.info("Confusion Matrix:")
    logger.info(cm)
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")

    # history = model.history.history
    #
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    #
    # # График accuracy
    # ax1.plot(history['accuracy'], label='Train Accuracy')
    # ax1.plot(history['val_accuracy'], label='Validation Accuracy')
    # ax1.set_title('Accuracy over Epochs')
    # ax1.set_xlabel('Epochs')
    # ax1.set_ylabel('Accuracy')
    # ax1.legend()
    #
    # # График loss
    # ax2.plot(history['loss'], label='Train Loss')
    # ax2.plot(history['val_loss'], label='Validation Loss')
    # ax2.set_title('Loss over Epochs')
    # ax2.set_xlabel('Epochs')
    # ax2.set_ylabel('Loss')
    # ax2.legend()
    #
    # plt.show()
    logger.info("successful ended")


