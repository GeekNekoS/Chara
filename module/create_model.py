import os
os.environ["KERAS_BACKEND"] = "tensorflow"
from module.project_logging import setup_logger
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers


logger = setup_logger("create_model")


def create_model(input_shape: tuple = (256, 256, 3), num_classes: int = 58) -> tf.keras.Model:
    """
    Создаёт модель на основе предобученной сети MobileNetV2 с изменением финальных слоёв под задачу классификации.
    """
    logger.info(f"start of creating model, shape: {input_shape}, num_classes: {num_classes}")

    inputs = tf.keras.Input(shape=input_shape)

    # Блок 1
    x = layers.Conv2D(32, (3, 3), padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(64, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.5)(x)

    # Блок 2
    x = layers.Conv2D(64, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(128, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.5)(x)

    # Блок 3
    x = layers.Conv2D(128, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(128, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.5)(x)

    # Полносвязный слой
    x = layers.Flatten()(x)
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    # Создаем модель, которая будет извлекать признаки из изображений
    model = keras.Model(inputs=inputs, outputs=outputs)

    logger.info("model created successfully")
    return model


# logger.info(
    #     f"Start creating model with pretrained MobileNetV2, input_shape: {input_shape}, num_classes: {num_classes}")
    #
    # # Загружаем предобученную сеть MobileNetV2 с ImageNet
    # base_model = tf.keras.applications.MobileNet(
    #     input_shape=input_shape,
    #     include_top=False,  # Не включаем верхние слои, добавим свои
    #     weights='imagenet'  # Используем предобученные веса ImageNet
    # )
    #
    # # Замораживаем веса предобученной модели
    # base_model.trainable = True
    #
    # # Входной слой
    # inputs = tf.keras.Input(shape=input_shape)
    # # Слой нормализации изображений (например, нормализуем в диапазон [0, 1])
    # x = layers.Rescaling(1./255)(inputs)  # Нормализация пикселей, делим на 255
    #
    # # Нормализация входных данных (если не требуется, можно убрать)
    # x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    #
    # # Проход через базовую модель
    # x = base_model(x, training=False)
    #
    # # Добавляем собственные слои
    # x = layers.GlobalAveragePooling2D()(x)  # Усреднение признаков
    # x = layers.Dense(512, activation='relu')(x)
    # x = layers.Dropout(0.3)(x)
    # x = layers.Dense(256, activation='relu')(x)
    # x = layers.Dropout(0.3)(x)
    #
    # # Финальный слой классификации
    # outputs = layers.Dense(num_classes, activation='softmax')(x)
    #
    # # Создаём модель
    # model = tf.keras.Model(inputs, outputs)
    #
    # logger.info("Model created successfully")
    # return model


def data_augmentation(images):
    data_augmentation_layers = [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ]
    for layer in data_augmentation_layers:
        images = layer(images)
    return images
