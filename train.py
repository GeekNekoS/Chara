"""
Модуль для тренировки нейронной сети/модели.
"""
import tensorflow as tf
import numpy as np
from module.project_logging import setup_logger
from module.train_model import train_model
from module.evaluate_model import evaluate_model
from module.load_train_test_val import load_train_test_val


logger = setup_logger("Learning model started")


def predict_class(directory, batch_size, image_size):
    train_dataset, test_dataset = load_train_test_val(directory, batch_size, image_size)
    model = tf.saved_model.load('models/model')
    logger.info("Сигнатуры модели", model.signatures)
    infer = model.signatures["serving_default"]
    # logger.info("Inputs:", infer.inputs)
    # logger.info("Outputs:", infer.outputs)
    infer = model.signatures["serving_default"]

    # Разворачиваем тестовый датасет (анбатчинг)
    unbatched_data = test_dataset.unbatch()

    # Список для хранения предсказаний
    inputs = []
    labels = []

    # Проходим по всем примерам в распакованном датасете
    for image, label in unbatched_data:
        # Добавляем изображение в модель для предсказания
        # pred = model(image[None, ...])  # Поскольку каждый image - это одна картинка, добавляем размерность
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
    predicted_classes = np.argmax(probabilities, axis=1) + 1 # так как индексы начинаются с нуля, надо увеличить на 1
    print("Предсказанный класс на первом фото:", predicted_classes[0], "с вероятностью", max_probability, "%")
    return predicted_classes, labels


if __name__ == '__main__':
    # directory = '/mnt/d/project_practicum/dataset'
    directory = '/mnt/d/project_practicum/mnist_images'
    batch_size = 128
    image_size = (28, 28)
    num_classes = 10
    epochs = 100
    learning_rate = 1e-04

    # train_model(
    #     directory=directory,
    #     batch_size=batch_size,
    #     image_size=image_size,
    #     num_classes = num_classes,
    #     epochs=epochs,
    #     learning_rate=learning_rate
    # )

    predicted_classes, labels = predict_class(directory, batch_size, image_size)
    evaluate_model(predicted_classes, labels)