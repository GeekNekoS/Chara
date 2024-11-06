from project_logging import setup_logger
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers

logger = setup_logger("create_model")


def create_model(input_shape: tuple, num_classes: int) -> tf.keras.Model:
    logger.info(f"start of creating model, shape: {input_shape}, num_classes: {num_classes}")
    try:
        inputs = keras.Input(shape=input_shape)
        x = layers.Conv2D(64, 3, 1, activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Conv2D(128, 3, 1, activation='relu', padding='same')(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dense(32, activation="relu")(x)
        outputs = layers.Dense(num_classes)(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name="model")
        logger.info("model created successfully")
        return model
    except Exception as exc:
        logger.error("Error creating model: %s", str(exc))
