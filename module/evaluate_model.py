from module.project_logging import setup_logger
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


logger = setup_logger("evaluate_model")


def evaluate_model(y_pred: np.array, y_true: np.array):
    logger.info("function starts")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    class_names = [i for i in range(10)]
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(y_true, y_pred, class_names)

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

def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Отображает матрицу ошибок в виде таблицы.

    :param y_true: истинные метки классов
    :param y_pred: предсказанные метки классов
    :param class_names: список с названиями классов
    """
    # Создаём матрицу ошибок
    cm = confusion_matrix(y_true, y_pred)

    # Визуализируем матрицу ошибок с помощью heatmap из seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)

    # Оформляем график
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()


