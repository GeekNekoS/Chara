from module.project_logging import setup_logger
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers

logger = setup_logger("create_model")


def create_model(input_shape: tuple, num_classes: int) -> tf.keras.Model:
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

      Пример использования:
      ----------------------
      >>> model = create_model(input_shape=(128, 128, 3), num_classes=10)
      >>> model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
      >>> model.summary()

      В примере создается модель для классификации изображений размером 128x128 с 3 цветовыми каналами на 10 классов.
      После создания модель компилируется и отображается её структура.

      """
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
