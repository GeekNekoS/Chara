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
    logger.info(
        f"Start creating model with pretrained MobileNetV2, input_shape: {input_shape}, num_classes: {num_classes}")

    # Загружаем предобученную сеть MobileNetV2 с ImageNet
    base_model = tf.keras.applications.MobileNet(
        input_shape=input_shape,
        include_top=False,  # Не включаем верхние слои, добавим свои
        weights='imagenet'  # Используем предобученные веса ImageNet
    )

    # Замораживаем веса предобученной модели
    base_model.trainable = False

    # Входной слой
    inputs = tf.keras.Input(shape=input_shape)

    # Нормализация входных данных (если не требуется, можно убрать)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

    # Проход через базовую модель
    x = base_model(x, training=False)

    # Добавляем собственные слои
    x = layers.GlobalAveragePooling2D()(x)  # Усреднение признаков
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    # Финальный слой классификации
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    # Создаём модель
    model = tf.keras.Model(inputs, outputs)

    logger.info("Model created successfully")
    return model


def data_augmentation(images):
    data_augmentation_layers = [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ]
    for layer in data_augmentation_layers:
        images = layer(images)
    return images
