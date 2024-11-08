import os
os.environ["KERAS_BACKEND"] = "tensorflow"
from module.project_logging import setup_logger
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import keras_cv

logger = setup_logger("create_model")


def create_model(input_shape: tuple, num_classes: int = 58) -> tf.keras.Model:
    """
      Создаёт и возвращает нейронную сеть на основе модели TensorFlow Keras для классификации изображений.

      Модель имеет сверточные слои с нормализацией, пулинговые слои, а также плотные (Dense) слои для
      завершения классификации. Она предназначена для обработки входных данных заданной формы и может
      классифицировать данные на указанное число классов.

      Параметры:
      ----------
      input_shape : tuple
          Форма входных данных (высота, ширина, каналы), которая указывается в первом слое `Input`. Это
          должна быть трёхмерная структура (например, `(128, 128, 3)` для изображений 128x128 с тремя
          каналами, например, RGB).

      num_classes : int
          Количество классов для классификации. Число, которое указывает, сколько различных категорий
          будет на выходе модели. Это количество нейронов в последнем Dense-слое, что позволяет
          прогнозировать вероятность каждого класса.

      Возвращаемое значение:
      ----------------------
      tf.keras.Model
          Объект модели Keras, готовый для компиляции и тренировки. Модель включает в себя
          входной слой, несколько сверточных слоёв с пулингом, и плотные слои для конечной классификации.

      Логика работы:
      --------------
      - **Сверточные слои** (`Conv2D`): используются для извлечения признаков изображения. Первый слой с 64 фильтрами и
        размером ядра 3x3 применяется к входному изображению, затем идет нормализация и пулинг, что позволяет уменьшить
        размерность изображения и повысить устойчивость к изменению положения объектов.
      - **MaxPooling2D**: операции понижающей дискретизации для уменьшения пространственных размеров.
        Они уменьшают количество параметров и вычислений в сети, что помогает избежать переобучения.
      - **BatchNormalization**: нормализация данных в свёрточном слое для ускорения обучения и повышения стабильности.
      - **Flatten**: преобразует двумерные данные в одномерный вектор для ввода в плотные слои.
      - **Dense слои**: два полносвязных слоя, последний из которых соответствует количеству классов. Слои с функцией
        активации `relu` позволяют модели обучаться сложным нелинейным зависимостям, а финальный слой выдаёт прогноз для
        каждого класса.

      Логирование:
      ------------
      - Логирует начало и завершение создания модели, включая параметры `input_shape` и `num_classes`.
      - При возникновении ошибки в процессе создания модели логирует подробное сообщение об ошибке.
      """
    logger.info(f"start of creating model, shape: {input_shape}, num_classes: {num_classes}")

    inputs = keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = layers.Rescaling(1. / 255)(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    # x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.MaxPooling2D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="model")

    # # Create a model using a pretrained backbone
    # backbone = keras_cv.models.EfficientNetV2Backbone.from_preset(
    #     "efficientnetv2_b0_imagenet"
    # )
    # model = keras_cv.models.ImageClassifier(
    #     backbone=backbone,
    #     num_classes=num_classes,
    #     activation="softmax",
    # )

    logger.info("model created successfully")
    return model


def data_augmentation(images):
    data_augmentation_layers = [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ]
    for layer in data_augmentation_layers:
        images = layer(images)
    return images
