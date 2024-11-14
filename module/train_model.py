import os
import tensorflow as tf
import keras
from sklearn.metrics import f1_score
from tensorflow.keras import backend as K
import numpy as np
from tensorflow.keras.callbacks import Callback
from module.project_logging import setup_logger
from module.load_train_test_val import load_train_test_val
from module.create_model import create_model
from module.evaluate_model import evaluate_model


logger = setup_logger("train_model")


def train_model(train_dataset,
                test_dataset,
                batch_size: int,
                image_size: tuple[int, int],
                num_classes: int = 58,
                epochs: int = 10,
                learning_rate: float = 1e-3
                ):
    """
    Обучает модель на основе изображений из заданного тренировочного и тестового датасетов, выполняя их загрузку,
    создание модели, её компиляцию и обучение. Завершает процесс оценкой модели на тестовом наборе данных.

    Параметры:
    ----------
    train_dataset : tf.data.Dataset
        Датасет для обучения, содержащий батчи изображений и соответствующих меток.

    test_dataset : tf.data.Dataset
        Датасет для тестирования модели, содержащий батчи изображений и соответствующих меток.

    batch_size : int
        Размер батча для обучения модели.

    image_size : tuple[int, int]
        Размер изображений, на которых будет обучаться модель (в формате (ширина, высота)).

    num_classes : int, по умолчанию 58
        Количество классов для классификации. Определяется по количеству поддиректорий в директории с данными.

    epochs : int, по умолчанию 10
        Количество эпох для обучения модели.

    learning_rate : float, по умолчанию 1e-3
        Значение скорости обучения для оптимизатора Adam.

    Возвращаемое значение:
    ----------------------
    None

    Логика работы:
    --------------
    1. Функция создаёт модель с помощью функции `create_model`, передавая размер входного изображения и количество классов.
    2. Компилирует модель с оптимизатором Adam, используя функцию потерь `categorical_crossentropy` и метрику точности.
    3. Настроены колбеки для ранней остановки и сохранения лучшей модели на основе валидационной точности.
    4. Модель обучается на тренировочном датасете, а затем загружается лучшая модель для её оценки на тестовом наборе данных.
    5. Логирует процесс обучения и ошибки, если они возникнут.

    Примечание:
    -----------
    Функция ожидает, что функция `create_model` создаёт модель для классификации с количеством классов, соответствующим
    числу подкатегорий в директории с данными.
    """
    logger.info(f"training starts with batch_size: {batch_size}, image_size: {image_size}, num_classes: {num_classes}")
    try:
        model = create_model(input_shape=image_size + (3,), num_classes=num_classes)
        # Показать модель
        model.summary()
        metrics = [
            keras.metrics.CategoricalAccuracy(),
        ]
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=metrics
        )
        # Определите колбек для сохранения модели
        # checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        #     filepath='models/model.keras',  # Путь для сохранения модели
        #     save_weights_only=False,  # Сохранять всю модель, а не только веса
        #     save_best_only=True,  # Сохранять только лучшую модель (по метрике)
        #     monitor='val_accuracy',  # Мониторить метрику (например, валидационную потерю)
        #     mode='max',  # Сохранять, если метрика уменьшается
        #     verbose=1  # Логирование процесса
        # )
        saved_model_callback = SavedModelCallback(
            save_dir='models/model',
            monitor='val_categorical_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1

        )
        # Определите колбек для ранней остановки
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_categorical_accuracy',  # Мониторим валидационную потерю
            patience=20,  # Количество эпох без улучшения перед остановкой
            verbose=1,  # Логирование ранней остановки
            restore_best_weights=True  # Восстановить лучшие веса после остановки
        )
        callbacks = [early_stopping_callback, saved_model_callback]
        history = model.fit(x=train_dataset,
                  # batch_size=128,
                  callbacks=callbacks,
                  validation_data=test_dataset,
                  verbose=1,
                  epochs=epochs
                  )
        model = tf.keras.models.load_model('models/model')
        evaluate_model(model, history, test_dataset)
        logger.info("training completed successfully")
    except Exception as exc:
        logger.error(f"An error occurred during training: {exc}")


class SavedModelCallback(Callback):
    def __init__(self, save_dir, monitor='val_loss', mode='min', save_best_only=True, verbose=1):
        """
        Колбек для сохранения модели в формате SavedModel, только если метрика улучшается.

        :param save_dir: Путь к директории для сохранения модели.
        :param monitor: Метрика для мониторинга (например, 'val_loss').
        :param mode: Режим мониторинга ('min' или 'max').
        :param save_best_only: Если True, сохраняется только лучшая модель.
        :param verbose: Уровень вывода сообщений (0 или 1).
        """
        super(SavedModelCallback, self).__init__()
        self.save_dir = save_dir
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.verbose = verbose

        # Установка начального значения метрики
        self.best = float('inf') if mode == 'min' else -float('inf')

    def on_epoch_end(self, epoch, logs=None):
        """
        Сохраняет модель в формате SavedModel, только если метрика улучшается.

        :param epoch: Номер текущей эпохи.
        :param logs: Логирования метрик обучения.
        """
        # Получение текущего значения метрики
        current = logs.get(self.monitor)
        if current is None:
            if self.verbose > 0:
                print(f"Warning: Metric '{self.monitor}' is not available in logs.")
            return

        # Определение, нужно ли сохранять модель
        if (self.mode == 'min' and current < self.best) or (self.mode == 'max' and current > self.best):
            self.best = current
            if self.verbose > 0:
                print(f"\n Epoch {epoch+1}: {self.monitor} improved to {current:.4f}, saving model to {self.save_dir}/model_epoch_{epoch+1}")

            # Создаем путь для сохранения модели
            model_save_path = os.path.join(self.save_dir, f"model_epoch_{epoch+1}")
            # Сохраняем модель в формате SavedModel
            tf.saved_model.save(self.model, model_save_path)
            print(f"\n Model saved to: {model_save_path}")

        elif self.verbose > 0:
            print(f"\n Epoch {epoch+1}: {self.monitor} did not improve from {self.best:.4f}")



