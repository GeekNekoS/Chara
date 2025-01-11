"""
Модуль, прогнозирующий классы
"""
import tensorflow as tf
import numpy as np
from module.project_logging import setup_logger
from module.load_train_test import load_train_test


logger = setup_logger('predict_class')


def predict_class_infer(test_dataset, batch_size, image_size):
    logger.info("Predicting class of images started")
    model = tf.saved_model.load('models/model')
    logger.info("Сигнатуры модели", model.signatures)
    infer = model.signatures["serving_default"]

    # Разворачиваем тестовый датасет (анбатчинг)
    unbatched_data = test_dataset.unbatch()

    # Список для хранения предсказаний
    inputs = []
    labels = []

    # Проходим по всем примерам в распакованном датасете и собираем их в списки
    for image, label in unbatched_data:
        inputs.append(image.numpy())
        labels.append(label.numpy())

    labels = np.array(labels)
    # Преобразуем список предсказаний в тензор
    inputs_tensor = tf.convert_to_tensor(inputs)
    # Получаем предсказания
    output = infer(inputs=inputs_tensor)
    # Извлекаем массив вероятностей
    probabilities = output['output_0']
    print("Предсказанные вероятности:", np.round(np.array(probabilities[0]), 2), "по классам")
    max_probability = np.round(np.max(probabilities) * 100, 2)
    # Находим индекс максимального значения вдоль оси 1 (для каждого примера)
    predicted_classes = np.argmax(probabilities, axis=1)
    labels = np.argmax(labels, axis=1)
    print("Предсказанный класс на первом фото:", predicted_classes[0], "с вероятностью", max_probability, "%")
    logger.info("Predicting class of images finished")
    return predicted_classes, labels


def predict_class(test_dataset, batch_size, image_size):
    logger.info("Predicting class of images started")

    # Загружаем модель .keras
    model = tf.keras.models.load_model('models/1/model.keras')
    logger.info("Модель загружена")

    # Разворачиваем тестовый датасет (анбатчинг)
    unbatched_data = test_dataset.unbatch()

    # Список для хранения изображений и меток
    inputs = []
    labels = []

    # Проходим по всем примерам в распакованном датасете и собираем их в списки
    for image, label in unbatched_data:
        inputs.append(image.numpy())
        labels.append(label.numpy())

    labels = np.array(labels)
    inputs_tensor = np.array(inputs)

    # Получаем предсказания
    predictions = model.predict(inputs_tensor, batch_size=batch_size)

    # Получаем вероятности для каждого класса
    probabilities = predictions

    print("Предсказанные вероятности:", np.round(probabilities[0], 2), "по классам")

    # Находим индекс максимального значения вдоль оси 1 (для каждого примера)
    predicted_classes = np.argmax(probabilities, axis=1)
    labels = np.argmax(labels, axis=1)

    print("Предсказанный класс на первом фото:", predicted_classes[0], "с вероятностью",
          np.round(np.max(probabilities[0]) * 100, 2), "%")
    logger.info("Predicting class of images finished")

    return predicted_classes, labels
