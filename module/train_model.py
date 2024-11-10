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


def train_model(directory: str,
                batch_size: int,
                image_size: tuple[int, int],
                num_classes: int = 58,
                epochs: int = 10,
                learning_rate: float = 1e-3
                ):
    """
      Обучает модель на основе изображений из заданной директории, выполняя загрузку данных,
      создание модели, её компиляцию и обучение. Завершает процесс оценкой модели на
      валидационном наборе данных.

      Параметры:
      ----------
      directory : str
          Путь к директории, содержащей изображения, организованные по подкаталогам для каждой
          категории (метки). Директория должна иметь подкаталоги, где каждая папка соответствует
          отдельному классу для классификации.

      batch_size : int
          Размер пакета (batch) для загрузки данных. Задает количество изображений, передаваемых
          в сеть за один шаг, что влияет на скорость и эффективность обучения.

      image_size : tuple[int, int]
          Размер загружаемых изображений в формате (высота, ширина). Все изображения масштабируются
          до этого размера, обеспечивая согласованность данных.
      num_classes : int
          Количество классов для распознавания

      Возвращаемое значение:
      ----------------------
      None
          Функция не возвращает значений, но записывает информацию в лог. В ходе работы функция
          выводит сообщения о начале и успешном завершении обучения. В случае возникновения ошибок
          информация также записывается в лог.

      Логика работы:
      --------------
      1. **Загрузка данных** — функция `load_train_test_val` загружает данные из `directory`, разбивая
         их на обучающий, тестовый и валидационный наборы данных.
      2. **Создание модели** — функция `create_model` создает сверточную нейронную сеть, подходящую для
         указанной задачи классификации, определяя входную форму и количество классов на основе загруженных
         данных.
      3. **Компиляция модели** — модель компилируется с использованием стандартных параметров.
      4. **Обучение модели** — модель обучается с использованием метода `fit` и колбэка `EarlyStopping`,
         который завершает обучение при отсутствии улучшений, предотвращая переобучение.
      5. **Оценка модели** — модель оценивается на валидационном наборе с использованием функции `evaluate_model`,
         чтобы проверить её точность после обучения.
      6. **Логирование** — начало, успешное завершение и ошибки в процессе обучения записываются в лог.

      Примечание:
      -----------
      Функция `create_model` определяет количество классов на основе количества поддиректорий в `directory`,
      предполагая, что каждая поддиректория представляет один класс.
      """
    logger.info(f"training starts with directory: {directory}, batch_size: {batch_size}, image_size: {image_size}")
    try:
        train_dataset, test_dataset = load_train_test_val(directory, batch_size, image_size)
        print(train_dataset)
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
        f1_score = []
        # Определите колбек для ранней остановки
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_categorical_accuracy',  # Мониторим валидационную потерю
            patience=50,  # Количество эпох без улучшения перед остановкой
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



