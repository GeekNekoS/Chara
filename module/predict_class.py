"""
Модуль, прогнозирующий классы
"""
import tensorflow as tf
import numpy as np
from module.project_logging import setup_logger
from module.load_train_test_val import load_train_test_val


logger = setup_logger('predict_class')


def predict_class(test_dataset, batch_size, image_size):
    logger.info("Predicting class of images started")
    model = tf.saved_model.load('models/model')
    logger.info("Сигнатуры модели", model.signatures)
    # logger.info("Inputs:", infer.inputs)
    # logger.info("Outputs:", infer.outputs)
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
