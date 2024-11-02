import tensorflow.keras as keras
from tensorflow.keras import layers

def create_model(input_shape: tuple, num_classes: int):

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
    return model
